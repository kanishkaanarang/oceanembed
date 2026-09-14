"""Run: python -m src.models.train_oceanembed_v3 --help (CPU or CUDA)."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .oceanembed_v3 import OceanEmbedNet_v3, OceanEmbedV3Loss, STANDARD_DEPTHS
from .v3_features import engineer_features, fit_preprocessor, transform_features


def split_rows(frame, mode="temporal", seed=42, gap=0, single_year=None):
    """Whole-date holdouts or seeded 2-degree spatial-block holdouts; no row shuffle split."""
    if mode not in ("temporal", "spatial") or gap < 0:
        raise ValueError("Use temporal or spatial splits with a nonnegative gap.")
    if mode == "temporal":
        if "date" in frame:
            labels = pd.to_datetime(frame.date, errors="raise", utc=True).dt.floor("D")
        elif single_year is not None and "day_of_year" in frame:
            labels = frame.day_of_year
            if labels.isna().any() or not labels.between(1, 366).all():
                raise ValueError("Invalid day_of_year values.")
        else:
            raise ValueError("Temporal split needs a date column, or --single-year with day_of_year.")
        if labels.isna().any():
            raise ValueError("Every temporal observation must have a valid date.")
        groups = np.sort(labels.unique())
    else:
        labels = (np.floor(frame.lat / 2).astype(int).astype(str) + ":" +
                  np.floor(frame.lon / 2).astype(int).astype(str))
        groups = np.random.default_rng(seed).permutation(np.sort(labels.unique()))
    n = len(groups)
    a, b = int(.7*n), int(.85*n)
    partitions = (groups[:a], groups[a+gap:b], groups[b+gap:])
    if any(len(p) == 0 for p in partitions):
        raise ValueError(f"Only {n} groups: reduce gap or supply more observations for three disjoint splits.")
    return tuple(np.flatnonzero(labels.isin(p).to_numpy()) for p in partitions)


@torch.inference_mode()
def evaluate(model, loader, mean, scale, device):
    model.eval()
    errors = []
    for x, target in loader:
        output = model(x.to(device))
        pred = torch.stack([output["temperature"], output["salinity"]], dim=1).float()*scale + mean
        errors.append((pred.cpu() - target).numpy())
    error = np.concatenate(errors).astype(np.float64)
    if not np.isfinite(error).all():
        raise FloatingPointError("Non-finite validation predictions.")
    return {name: {"rmse": float(np.sqrt(np.mean(error[:, i]**2))),
                   "mae": float(np.mean(np.abs(error[:, i]))),
                   "rmse_by_depth": np.sqrt(np.mean(error[:, i]**2, axis=0)).tolist()}
            for i, name in enumerate(("temperature", "salinity"))}


def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train(args):
    if min(args.epochs, args.batch_size, args.patience, args.threads) < 1 or args.gap_groups < 0:
        raise ValueError("Epochs, batch size, patience and threads must be positive; gap cannot be negative.")
    if args.physics_warmup <= 0 or args.lr <= 0 or args.workers < 0 or args.max_rows < 0:
        raise ValueError("Warmup and learning rate must be positive; workers and max rows cannot be negative.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    data_path, out = Path(args.data), Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a new output directory; {out} is not empty.")
    frame = pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)
    frame["_source_row"] = np.arange(len(frame))
    if args.max_rows and len(frame) > args.max_rows:
        frame = frame.sample(args.max_rows, random_state=args.seed).sort_index()
    last = args.last_depth or (1000 if {"temp_1000m", "sal_1000m"} <= set(frame) else 900)
    depths = list(STANDARD_DEPTHS[:-1]) + [last]
    columns = [[f"{prefix}_{d}m" for d in depths] for prefix in ("temp", "sal")]
    missing = set(sum(columns, [])) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing targets (900 m is never relabeled 1000 m): {sorted(missing)}")
    target = np.stack([frame[c].to_numpy(dtype=np.float32) for c in columns], axis=1)
    bad = ((target[:, 0] < -3) | (target[:, 0] > 45) |
           (target[:, 1] < 0) | (target[:, 1] > 50)).any(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} profiles contain implausible T/S targets. "
                         "Rebuild from raw supported depths; do not clip extrapolated labels.")
    # Complete-column supervision; record exclusions rather than inventing labels.
    valid = np.isfinite(target).all(axis=(1, 2)) & np.isfinite(frame[["lat", "lon"]]).all(axis=1).to_numpy()
    dropped = int((~valid).sum())
    frame, target = frame.loc[valid].reset_index(drop=True), target[valid]
    indices = split_rows(frame, args.split, args.seed, args.gap_groups, args.single_year)
    features = engineer_features(frame)
    processor = fit_preprocessor(features.iloc[indices[0]])
    x = transform_features(features, processor)
    mean_cpu = torch.from_numpy(target[indices[0]].mean(axis=0))
    scale_cpu = torch.from_numpy(np.maximum(target[indices[0]].std(axis=0), np.array([[.5], [.05]]))).float()
    mean, scale = mean_cpu.to(device), scale_cpu.to(device)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = [DataLoader(TensorDataset(torch.from_numpy(x[idx]), torch.from_numpy(target[idx])),
                          batch_size=args.batch_size, shuffle=(i == 0), num_workers=args.workers,
                          pin_memory=device.type == "cuda", generator=generator if i == 0 else None)
               for i, idx in enumerate(indices)]
    config = {"input_dim": x.shape[1], "latent_dim": 128, "depths": depths, "dropout": args.dropout}
    model = OceanEmbedNet_v3(**config).to(device)
    loss_config = {"depths": depths, "alpha": args.alpha, "beta": args.beta, "gamma": args.gamma,
                   "monotonic_weight": args.monotonic_weight}
    criterion = OceanEmbedV3Loss(**loss_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {"args": vars(args), "model_config": config, "loss_config": loss_config,
                "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "discarded_rows": dropped, "split_rows": [len(i) for i in indices],
                "device": str(device), "fp16": amp, "torch_version": str(torch.__version__),
                "target_convention": "potential temperature theta0, practical salinity PSS-78",
                "metrics_scope": "SMOKE TEST ONLY" if args.max_rows else "held-out groups"}
    (out / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out / "split_rows.json").write_text(json.dumps({name: frame.iloc[idx]._source_row.tolist()
        for name, idx in zip(("train", "validation", "test"), indices)}), encoding="utf-8")
    print(json.dumps(metadata), flush=True)
    best, stale, history = float("inf"), 0, []
    for epoch in range(args.epochs):
        model.train()
        start, totals, seen = time.perf_counter(), {}, 0
        for step, (batch_x, batch_y) in enumerate(loaders[0]):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                prediction = model(batch_x)
            # Explicit float32 de-normalization and EOS-80 evaluation under AMP.
            physical = torch.stack([prediction["temperature"], prediction["salinity"]], dim=1).float()*scale + mean
            losses = criterion(physical[:, 0], physical[:, 1], batch_y[:, 0], batch_y[:, 1],
                               physics_weight=min(1., (epoch+1)/args.physics_warmup))
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch+1}, batch {step}.")
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=not amp)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step(epoch + (step+1)/len(loaders[0]))
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.) + float(value.detach())*len(batch_x)
            seen += len(batch_x)
        validation = evaluate(model, loaders[1], mean, scale, device)
        # Both tasks contribute equally relative to their requested error targets.
        score = .5*(validation["temperature"]["rmse"]/.65 + validation["salinity"]["rmse"]/.25)
        row = {"epoch": epoch+1, "train": {k: v/seen for k, v in totals.items()},
               "validation": validation, "score": score, "seconds": time.perf_counter()-start}
        history.append(row)
        print(json.dumps(row), flush=True)
        improved = score < best - 1e-4
        if improved:
            best, stale = score, 0
        else:
            stale += 1
        checkpoint = {"model_config": config, "model_state": model.state_dict(), "preprocessor": processor,
                      "target_mean": mean_cpu, "target_scale": scale_cpu, "epoch": epoch+1,
                      "best_score": best, "validation": validation, "metadata": metadata,
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict()}
        atomic_save(checkpoint, out / "last.pt")
        if improved:
            atomic_save(checkpoint, out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stale >= args.patience:
            break
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test = evaluate(model, loaders[2], mean, scale, device)
    baseline_error = target[indices[2]].astype(float) - mean_cpu.numpy()
    report = {"best_epoch": checkpoint["epoch"], "validation": checkpoint["validation"], "test": test,
              "train_mean_baseline_test_rmse": np.sqrt(np.mean(baseline_error**2, axis=(0, 2))).tolist(),
              "depths_m": depths, "scope": metadata["metrics_scope"],
              "targets_met_on_test": test["temperature"]["rmse"] < .65 and test["salinity"]["rmse"] < .25}
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    return report


def predict_from_checkpoint(path, frame, device="cpu"):
    """Reload the exact feature schema, normalization and physical depth labels."""
    saved = torch.load(path, map_location=device, weights_only=True)
    model = OceanEmbedNet_v3(**saved["model_config"]).to(device).eval()
    model.load_state_dict(saved["model_state"])
    x = transform_features(engineer_features(frame), saved["preprocessor"])
    with torch.inference_mode():
        output = model(torch.from_numpy(x).to(device))
        physical = torch.stack([output["temperature"], output["salinity"]], dim=1).cpu()
        physical = physical*saved["target_scale"].cpu() + saved["target_mean"].cpu()
    return {"temperature_c": physical[:, 0].numpy(), "salinity_psu": physical[:, 1].numpy(),
            "depths_m": saved["model_config"]["depths"]}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/processed/multimodal_training_dataset_v3_qc.parquet")
    p.add_argument("--output", default="results/oceanembed_v3")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--alpha", type=float, default=4.)
    p.add_argument("--beta", type=float, default=.01)
    p.add_argument("--gamma", type=float, default=.1)
    p.add_argument("--monotonic-weight", type=float, default=.05)
    p.add_argument("--physics-warmup", type=float, default=10.)
    p.add_argument("--split", choices=["temporal", "spatial"], default="temporal")
    p.add_argument("--gap-groups", type=int, default=1)
    p.add_argument("--single-year", type=int, help="Explicit year assertion when the source has day_of_year but no date.")
    p.add_argument("--last-depth", choices=[900, 1000], type=int)
    p.add_argument("--workers", type=int, default=0, help="Zero is portable on Windows; CUDA can use 2–4.")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=0, help="Subsample for smoke tests only.")
    p.add_argument("--no-amp", action="store_true")
    return p


if __name__ == "__main__":
    train(parser().parse_args())
