# OceanEmbed v3: joint physical water-column reconstruction

## Data correction comes first

The local legacy table has 72,280 profiles but only five days (1–5 January 2023),
15 targets ending at **900 m**, and no winds. Its surface inputs and profile
targets come from the same GLORYS reanalysis. This is not independent satellite
versus Argo validation. The legacy builder extrapolated shallow columns to 900 m,
producing temperatures from -96.691 to 273.225 C and salinity up to 561.481 PSU.
These labels invalidate direct comparisons with new scores.

`build_training_dataset.py` now rejects columns that do not reach the deepest
target, rejects interior missing samples and implausible interpolated targets,
and retains the full observation date. The uppermost valid sample within 5 m is
held constant to the surface; there is no extrapolation below the deepest sample.
The separately regenerated `multimodal_training_dataset_v3_qc.parquet` has 71,380
profiles. Existing datasets and deployed model weights were not overwritten.
Full-column selection excludes shallow shelves; scores apply to supported deep
columns. Masked partial-column supervision is needed for a future shelf model.

Rebuild this separate dataset from the repository root:

```powershell
.venv/Scripts/python.exe -c "from src.preprocessing.build_training_dataset import build_training_table; build_training_table(output_path='data/processed/multimodal_training_dataset_v3_qc.parquet', output_csv='data/processed/multimodal_training_dataset_v3_qc.csv')"
```

## Architecture and physical conventions

`src/models/oceanembed_v3.py` implements `OceanEmbedNet_v3`. A normalized surface
feature vector passes through a 256-unit MLP to a shared 128-dimensional latent
representation. Continuous physical-depth coordinates and learned depth embeddings
expand that representation across 15 layers. Three residual Conv1d blocks with
dilations 1, 2, 4 and squeeze-and-excitation channel attention couple the water
column. GroupNorm avoids batch-size-dependent running statistics. Independent
convolutional temperature and salinity heads each return 15 standardized outputs.

The training/inference wrapper restores physical units with training-only,
per-task, per-depth means and scales. The model defaults to 1000 m; the trainer
detects actual column names and preserves 900 m labels when that is the source.
`--last-depth 1000` requires real `temp_1000m` and `sal_1000m` targets. Never rename
900 m labels or extrapolate them to claim a 1000 m model.

Loss in physical units:

```
L = MSE(T) + alpha*MSE(S) + ramp*(beta*L_stability + gamma*L_thermocline)
L_stability = mean(relu(-100*delta(rho_theta)/delta(z))**2)
L_thermocline = mean(w*(delta(T_pred)/delta(z)-delta(T_true)/delta(z))**2)/0.05**2
                + monotonic_weight*L_cooling
```

Defaults: alpha=4, beta=0.01, gamma=0.1, monotonic_weight=0.05; ramp reaches one
over ten epochs. These are starting weights to tune on validation data. `w=4`
for interval midpoints from 50 through 150 m and one elsewhere. Actual metre
spacing enters all derivatives. MSE is otherwise equally weighted across tiers.

The differentiable EOS-80 polynomial comes from the existing
`src/models/physics_losses.py`. Inputs must be **potential temperature referenced
to 0 dbar** and **practical salinity**. Density increasing with positive-down depth
is a static stability prior, not the hydrostatic pressure equation `dp/dz=rho*g`.
Using in-situ density would mix compressibility with stratification. For operational
validation, evaluate TEOS-10 buoyancy frequency using the correct salinity,
temperature and pressure conversions; this regularizer is an approximation.
See [TEOS-10 conventions](https://www.teos-10.org/pubs/gsw/v3_06_11/pdf/Getting_Started.pdf).

MLD is estimated during supervised training from the first target potential-density
increase of 0.03 kg/m3 relative to 10 m. Below it, the model receives a soft penalty
for predicted warming exceeding observed positive warming by 0.001 C/m. No crossing
means this cooling term is inactive. Target MLD prevents the prediction moving the
boundary to evade the loss. Observed warming inversions are allowed; tropical
barrier layers make strict monotonic temperature constraints inappropriate.

## Training, evaluation and inference

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-training-v3.txt
# For the test command below: python -m pip install pytest

# Current five-day dataset: train days 1–3, validate day 4, test day 5.
.venv/Scripts/python.exe -m src.models.train_oceanembed_v3 --gap-groups 0 --epochs 150 --batch-size 256 --patience 25 --output results/oceanembed_v3_experiment01

# Quick wiring check; its scores are explicitly marked SMOKE TEST ONLY.
.venv/Scripts/python.exe -m src.models.train_oceanembed_v3 --gap-groups 0 --epochs 2 --max-rows 1024 --output results/oceanembed_v3_smoke_new

# A separate spatial-generalization experiment, not an independent temporal test.
.venv/Scripts/python.exe -m src.models.train_oceanembed_v3 --split spatial --gap-groups 0 --output results/oceanembed_v3_spatial01

.venv/Scripts/python.exe -m pytest tests -q
```

Use a fresh output directory for every run. For a longer time series, retain the
default one-group gap and increase it to match decorrelation time. Groups are
whole observed days, not guaranteed consecutive calendar days. Spatial mode holds
out seeded 2-degree cells; it does not implement a geographic buffer, and the same
dates can occur in each spatial partition. Historical inputs need additional
purging equal to their lookback duration. A legacy table without `date` requires
an explicit `--single-year 2023` assertion.

The trainer supplies DataLoader, AdamW, cosine warm restarts, gradient clipping,
CUDA fp16 GradScaler, full-precision EOS/loss calculations, and validation-based
early stopping. CPU falls back to float32. CUDA fp16 is implemented but was not
exercised on this CPU-only machine. See [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp)
and [the warm-restart scheduler](https://docs.pytorch.org/docs/main/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html).

Preprocessing fits only training rows: median imputation, missing flags, feature
normalization and target scaling. Nonfinite complete profiles are excluded and
counted; physically implausible targets stop training rather than being clipped.
Early stopping minimizes `0.5*(validation_RMSE_T/0.65 + validation_RMSE_S/0.25)`.
This balances both targets but does not guarantee either threshold individually.
The test set is evaluated only after selecting the best validation checkpoint.

Outputs: `best.pt`, `last.pt`, `config.json` including source SHA256 and conventions,
`split_rows.json` with original row positions, `history.json`, and `metrics.json`
with per-depth RMSE, overall RMSE/MAE and a training-mean baseline. Checkpoints
include optimizer/scheduler/scaler states; automatic resume is not exposed by the CLI.

```python
import pandas as pd
from src.models.train_oceanembed_v3 import predict_from_checkpoint

surface = pd.read_parquet('data/processed/multimodal_training_dataset_v3_qc.parquet').head(32)
profiles = predict_from_checkpoint('results/oceanembed_v3_experiment01/best.pt', surface)
print(profiles['temperature_c'].shape, profiles['salinity_psu'].shape)
print(profiles['depths_m'])
```

Inference uses surface feature columns only. Batch large regional tables explicitly
to control memory. This is a separate candidate pipeline; promote a checkpoint into
the serving engine only after validating its feature sources and depth convention.

## Feature engineering

Implemented pointwise features in `v3_features.py`: current speed and kinetic energy,
current u*v, SST*SSH, SSS*SSH, centred SST*SSS, positive SST excess above 26 C,
Coriolis parameter, coordinate trigonometric features, and SST/SSS/SSH interactions
with seasonal sine/cosine. Optional paired `wind_u`, `wind_v` enable wind speed,
constant-drag wind-stress proxies and wind-current alignment/cross products.
Wind speed cubed is also included. Precomputed columns named `wind_stress_curl`,
`ekman_pumping`, `geostrophic_vorticity`, `sst_gradient_magnitude`,
`sss_gradient_magnitude`, `thermal_advection`, `sla`, `mld_climatology` and
`mixed_layer_heat_proxy` are included when supplied; inference must supply the
same feature schema used for training.
These are hypotheses to ablate, not guaranteed improvements.

Compute these additional fields on collocated, masked grids before table sampling:

| Feature | Definition / requirement |
| --- | --- |
| Wind stress | tau = rho_air*C_D*abs(U10)*U10; prefer supplied stress or a validated bulk algorithm over the constant-drag proxy. |
| Stress curl | d(tau_y)/dx - d(tau_x)/dy, using metre distances and ocean-only stencils. |
| Ekman pumping | w_E = d(tau_y/(rho0*f))/dx - d(tau_x/(rho0*f))/dy, positive upward; accounts for latitude-dependent f. Mask an equatorial band, e.g. abs(lat)<5 degrees, and test sensitivity. |
| Geostrophic relative vorticity | d(v_g)/dx - d(u_g)/dy; useful with strain and eddy polarity. |
| Thermal advection | -(u*dSST/dx + v*dSST/dy), plus SST/SSS gradient magnitude. |
| Wind mixing proxy | abs(U10)**3, with antecedent 7/14/30-day summaries using only information available at inference. |
| Mixed-layer heat proxy | rho0*Cp*h_clim*max(SST-26,0), if an independent MLD climatology h_clim is supplied; label as proxy, not measured TCHP. |
| Sea-level anomaly | SSH minus a consistent independent reference climatology; raw SSH and SLA are not interchangeable. |

Never compute curl/vorticity from unrelated rows or use degrees as metres. At the
equator, standard Ekman division by f is singular. See [NOAA's Ekman dynamics review](https://www.pmel.noaa.gov/pubs/outstand/kess2580/dynamics.shtml).
Do not use target-derived MLD, density or subsurface heat content as input features.

## Credible accuracy claims

Completed local run: `results/oceanembed_v3_qc_full`, seed 42, 20 epochs, batch 256,
CPU float32. Training took approximately 351 seconds. The best validation
checkpoint was epoch 20; 42,828 profiles trained on January 1–3, 14,276 validated
on January 4, and 14,276 tested on January 5, 2023. All 15 layers end at 900 m.

| Held-out day-5 metric | Temperature (C) | Salinity (PSU) |
| --- | ---: | ---: |
| RMSE | 0.270772 | 0.102692 |
| MAE | 0.175176 | 0.057181 |
| Training-mean baseline RMSE | 1.000574 | 0.613772 |

Both requested overall thresholds are met **on this limited split**. This does
not measure independent satellite-to-Argo, seasonal or new-region performance.
The baseline above is the training mean, not a retrained HistGradientBoosting
comparison. Raw per-depth metrics are in `metrics.json`; the complete inference
checkpoint is `best.pt`. The initial `oceanembed_v3_smoke` run used the invalid
legacy labels and is superseded by the corrected-data runs; do not cite it.

First compare HistGradientBoosting and v3 on the **same corrected targets and saved
split**, then compare v3 with `--beta 0 --gamma 0`, and finally tune physical weights
on validation only. Report temperature and salinity independently, depthwise and
for 50–150 m, across multiple seeds. Quantify uncertainty using day/region blocks,
not independent-row bootstraps. Five adjacent dates cannot establish seasonal or
geographic generalization. More dates, causal forcing histories, and independent
satellite-to-Argo matchup evaluation matter more than adding attention alone.

Neither requested threshold is guaranteed. Report measured test scores with their
sampling/depth limitations, and do not claim the change from the legacy score is
an architectural improvement: target quality and the validation protocol changed.
