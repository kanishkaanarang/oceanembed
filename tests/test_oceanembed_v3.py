import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch

from src.models.oceanembed_v3 import OceanEmbedNet_v3, OceanEmbedV3Loss, STANDARD_DEPTHS
from src.models.train_oceanembed_v3 import split_rows, train, parser
from src.models.v3_features import engineer_features, fit_preprocessor, transform_features
from src.preprocessing.build_training_dataset import interpolate_complete_column


class OceanEmbedV3Tests(unittest.TestCase):
    def test_both_heads_and_shared_encoder_receive_gradients(self):
        torch.set_num_threads(2)
        model = OceanEmbedNet_v3(9, dropout=0)
        output = model(torch.randn(3, 9))
        self.assertEqual(output['latent'].shape, (3, 128))
        self.assertEqual(output['temperature'].shape, (3, 15))
        self.assertEqual(output['salinity'].shape, (3, 15))
        (output['temperature'].square().mean() + output['salinity'].square().mean()).backward()
        for module in (model.encoder, model.temperature_head, model.salinity_head):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))

    def test_density_inversion_penalty_and_exact_loss_composition(self):
        loss = OceanEmbedV3Loss()
        cool = torch.linspace(28, 4, 15)[None]
        sal = torch.full_like(cool, 35)
        stable = loss(cool, sal, cool, sal)
        self.assertAlmostEqual(stable['total'].item(), 0.0, places=6)
        warm = cool.flip(-1).clone().requires_grad_()
        inverted = loss(warm, sal, cool, sal)
        self.assertGreater(inverted['hydrostatic'].item(), 0)
        self.assertTrue(torch.allclose(
            inverted['total'],
            inverted['mse_temperature'] + 4 * inverted['mse_salinity']
            + 0.01 * inverted['hydrostatic'] + 0.1 * inverted['thermocline']
        ))
        inverted['total'].backward()
        self.assertTrue(torch.isfinite(warm.grad).all())

    def test_observed_warm_inversions_are_not_monotonicity_errors(self):
        t = torch.linspace(28, 4, 15)[None]
        t[:, 8] = t[:, 7] + 1
        s = torch.full_like(t, 35)
        result = OceanEmbedV3Loss()(t, s, t, s)
        self.assertEqual(result['monotonic'].item(), 0)
        self.assertEqual(result['thermocline'].item(), 0)

    def test_gradient_loss_uses_real_metres(self):
        z = torch.tensor(STANDARD_DEPTHS, dtype=torch.float32)[None]
        truth = 28 - 0.02 * z
        prediction = truth + 0.001 * z
        sal = torch.full_like(truth, 35)
        result = OceanEmbedV3Loss()(prediction, sal, truth, sal)
        mid = (z[:, 1:] + z[:, :-1]) / 2
        expected = ((0.001 ** 2 / 0.05 ** 2) * (1 + 3 * ((mid >= 50) & (mid <= 150)).float())).mean()
        self.assertAlmostEqual(result['thermocline'].item(), expected.item(), places=4)

    def test_split_keeps_days_disjoint_and_chronological(self):
        frame = pd.DataFrame({'date': np.repeat(pd.date_range('2023-01-01', periods=20), 3)})
        train_idx, val_idx, test_idx = split_rows(frame, gap=1)
        self.assertLess(frame.iloc[train_idx].date.max(), frame.iloc[val_idx].date.min())
        self.assertLess(frame.iloc[val_idx].date.max(), frame.iloc[test_idx].date.min())
        self.assertFalse(bool(set(train_idx) & set(val_idx)))
        self.assertFalse(bool(set(val_idx) & set(test_idx)))
        frame.loc[0, 'date'] = pd.NaT
        with self.assertRaises(ValueError):
            split_rows(frame)

    def test_imputation_is_train_only_and_records_missingness(self):
        train_df = pd.DataFrame({'x': [1.0, 3.0, np.nan]})
        state = fit_preprocessor(train_df)
        result = transform_features(pd.DataFrame({'x': [100.0, np.nan]}), state)
        self.assertEqual(state['median'], [2.0])
        self.assertEqual(state['mean'], [2.0])
        self.assertEqual(result[1].tolist(), [0.0, 1.0])
        self.assertGreater(result[0, 0], 100)

    def test_optional_winds_must_be_paired(self):
        frame = pd.DataFrame({key: [1.0] for key in
            ['lat', 'lon', 'doy_sin', 'doy_cos', 'sst', 'sss', 'ssh', 'current_u', 'current_v', 'wind_u']})
        with self.assertRaises(ValueError):
            engineer_features(frame)
        frame['wind_v'] = 2.0
        self.assertIn('tau_x_proxy', engineer_features(frame))

    def test_interpolation_never_extrapolates_below_bottom(self):
        z = np.array([0.5, 10.0, 100.0])
        self.assertIsNone(interpolate_complete_column(z, np.array([28.0, 27.0, 20.0]), np.full(3, 35.0)))
        z = np.array([0.5, 10.0, 1000.0])
        t = np.array([28.0, 27.0, 4.0])
        ti, si = interpolate_complete_column(z, t, np.full(3, 35.0))
        self.assertEqual(ti[0], 28)
        self.assertAlmostEqual(ti[-1], float(np.interp(900, z, t)), places=5)
        self.assertTrue((si == 35).all())
        self.assertIsNone(interpolate_complete_column(z, np.array([28.0, np.nan, 4.0]), np.full(3, 35.0)))

    def test_invalid_depths_rejected(self):
        with self.assertRaises(ValueError):
            OceanEmbedNet_v3(9, depths=[0] * 15)

    def test_trainer_rejects_extrapolated_labels_before_training(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            columns = {f'{prefix}_{depth}m': [value] for prefix, value in [('temp', 20.0), ('sal', 35.0)]
                       for depth in STANDARD_DEPTHS}
            columns.update({'lat': [15.0], 'lon': [85.0], 'temp_1000m': [273.0]})
            path = tmp_path / 'invalid.csv'
            pd.DataFrame(columns).to_csv(path, index=False)
            output = tmp_path / 'run'
            args = parser().parse_args(['--data', str(path), '--output', str(output)])
            with self.assertRaises(ValueError):
                train(args)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
