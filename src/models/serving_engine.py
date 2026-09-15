"""Checkpoint-backed v3 serving with its original feature schema and depth labels."""
from collections import OrderedDict
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from .oceanembed_net import get_engine as get_data_engine, COMMON_LATS, COMMON_LONS
from .oceanembed_v3 import OceanEmbedNet_v3
from .v3_features import engineer_features, transform_features

CHECKPOINT = Path(__file__).resolve().parents[2] / 'results/oceanembed_v3_qc_full/best.pt'
STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 900]


class V3ServingEngine:
    def __init__(self):
        saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        if saved['model_config']['depths'] != STANDARD_DEPTHS:
            raise ValueError('Serving depth schema does not match the checkpoint.')
        self.model = OceanEmbedNet_v3(**saved['model_config']).eval()
        self.model.load_state_dict(saved['model_state'])
        self.processor = saved['preprocessor']
        self.mean, self.scale = saved['target_mean'], saved['target_scale']
        self.model_path, self.model_loaded = CHECKPOINT, True
        self.model_name = 'OceanEmbedNet v3 · SE-ResNet · dual T/S heads'
        self.report = json.loads(CHECKPOINT.with_name('metrics.json').read_text())
        self.data = get_data_engine()
        self.raw_inputs, self.ocean_mask = self.data.raw_inputs, self.data.ocean_mask
        self.dataset_mode, self.is_compact = self.data.dataset_mode, self.data.is_compact
        self.num_days = self.data.num_days
        self._cache = OrderedDict()
        torch.set_num_threads(min(torch.get_num_threads(), 4))

    @torch.inference_mode()
    def predict_features(self, frame):
        x = transform_features(engineer_features(frame), self.processor)
        physical, latent = [], []
        for start in range(0, len(x), 256):
            output = self.model(torch.from_numpy(x[start:start+256]))
            physical.append((torch.stack([output['temperature'], output['salinity']], 1)
                             * self.scale + self.mean).numpy())
            latent.append(output['latent'].numpy())
        return np.concatenate(physical), np.concatenate(latent)

    def predict_day(self, day_index):
        idx = int(np.clip(day_index, 0, self.num_days-1))
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        surface = self.raw_inputs[idx]
        lon, lat = np.meshgrid(COMMON_LONS, COMMON_LATS)
        mask = self.ocean_mask
        doy = (date(2023, 2, 1) + timedelta(days=idx)).timetuple().tm_yday
        frame = pd.DataFrame(dict(lat=lat[mask], lon=lon[mask],
            doy_sin=np.sin(2*np.pi*doy/365), doy_cos=np.cos(2*np.pi*doy/365),
            sst=surface[0][mask], sss=surface[1][mask], ssh=surface[2][mask],
            current_u=surface[5][mask], current_v=surface[6][mask]))
        pred, latent = self.predict_features(frame)
        if not np.isfinite(pred).all():
            raise FloatingPointError('V3 produced non-finite ocean predictions.')
        t = np.full((15, *mask.shape), np.nan, dtype=np.float32)
        s = t.copy()
        embedding = np.full((128, *mask.shape), np.nan, dtype=np.float32)
        t[:, mask], s[:, mask], embedding[:, mask] = pred[:, 0].T, pred[:, 1].T, latent.T
        actual = self.data.raw_targets[idx].copy()
        # Reference has 700/1000 m; interpolate 900 m without relabeling 1000 m.
        actual[-1] = actual[-2] + (actual[-1]-actual[-2]) * (200/300)
        result = dict(day_index=idx, pred_celsius=t, pred_salinity=s, actual_celsius=actual,
                      embedding=embedding, surface_inputs=surface, ocean_mask=mask)
        self._cache[idx] = result
        if len(self._cache) > 8:
            self._cache.popitem(last=False)
        return result


@lru_cache(maxsize=1)
def get_engine():
    return V3ServingEngine()
