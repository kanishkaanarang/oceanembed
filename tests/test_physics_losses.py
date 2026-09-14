"""Focused tests for differentiable ocean-profile physics."""

import unittest

import torch

from src.models.physics_losses import (
    STANDARD_DEPTHS,
    PhysicsInformedProfileLoss,
    density_stability_loss,
    eos80_potential_density,
    mackenzie_sound_speed,
    soft_mixed_layer_depth,
)


class PhysicsLossTests(unittest.TestCase):
    def setUp(self):
        self.depths = torch.tensor(STANDARD_DEPTHS, dtype=torch.float32)

    def test_standard_depths_match_problem_statement(self):
        self.assertEqual(
            STANDARD_DEPTHS,
            (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000),
        )

    def test_eos80_reference_density(self):
        density = eos80_potential_density(
            torch.tensor([35.0]),
            torch.tensor([0.0]),
        )
        self.assertAlmostEqual(density.item(), 1028.1063, places=3)

    def test_mackenzie_surface_reference(self):
        sound_speed = mackenzie_sound_speed(
            torch.tensor([0.0]),
            torch.tensor([35.0]),
            torch.tensor([0.0]),
        )
        self.assertAlmostEqual(sound_speed.item(), 1448.96, places=2)

    def test_stability_loss_detects_density_inversion(self):
        stable_density = 1024.0 + 0.002 * self.depths
        inverted_density = stable_density.clone()
        inverted_density[8] = inverted_density[7] - 0.5

        stable_loss = density_stability_loss(
            stable_density.unsqueeze(0),
            self.depths,
        )
        inverted_loss = density_stability_loss(
            inverted_density.unsqueeze(0),
            self.depths,
        )

        self.assertEqual(stable_loss.item(), 0.0)
        self.assertGreater(inverted_loss.item(), 0.0)

    def test_soft_mld_tracks_first_threshold_crossing(self):
        density = torch.full((1, len(STANDARD_DEPTHS)), 1024.0)
        density[:, 5:] += 0.08
        mld, crossing = soft_mixed_layer_depth(
            density,
            self.depths,
            transition_width=0.001,
        )

        self.assertAlmostEqual(mld.item(), 50.0, places=2)
        self.assertGreater(crossing[0, 2].item(), 0.99)

    def test_combined_loss_backpropagates(self):
        target_theta = torch.linspace(28.0, 4.0, len(STANDARD_DEPTHS)).repeat(2, 1)
        target_salinity = torch.linspace(33.0, 35.0, len(STANDARD_DEPTHS)).repeat(2, 1)
        predicted_theta = (target_theta + 0.2).clone().requires_grad_(True)
        predicted_salinity = (target_salinity - 0.05).clone().requires_grad_(True)
        surface_anchor = torch.stack(
            [target_theta[:, 0], target_salinity[:, 0]],
            dim=-1,
        )
        loss_function = PhysicsInformedProfileLoss()

        losses = loss_function(
            predicted_theta,
            predicted_salinity,
            target_theta,
            target_salinity,
            surface_anchor=surface_anchor,
        )
        losses["total"].backward()

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertIsNotNone(predicted_theta.grad)
        self.assertIsNotNone(predicted_salinity.grad)
        self.assertTrue(torch.isfinite(predicted_theta.grad).all())
        self.assertTrue(torch.isfinite(predicted_salinity.grad).all())


if __name__ == "__main__":
    unittest.main()
