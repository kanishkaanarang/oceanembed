"""Differentiable ocean-profile physics and losses for SIH26066."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


STANDARD_DEPTHS = (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000)


def eos80_potential_density(practical_salinity: Tensor, theta0: Tensor) -> Tensor:
    """Return EOS-80 potential density at zero pressure in kg m-3.

    ``theta0`` must be potential temperature referenced to 0 dbar and
    ``practical_salinity`` must use PSS-78. Both tensors must have equal shape.
    """
    if practical_salinity.shape != theta0.shape:
        raise ValueError("practical_salinity and theta0 must have equal shape")

    salinity = practical_salinity.clamp_min(0.0)
    temperature = theta0

    pure_water_density = (
        999.842594
        + 6.793952e-2 * temperature
        - 9.095290e-3 * temperature.square()
        + 1.001685e-4 * temperature.pow(3)
        - 1.120083e-6 * temperature.pow(4)
        + 6.536332e-9 * temperature.pow(5)
    )
    salinity_linear_coefficient = (
        0.824493
        - 4.0899e-3 * temperature
        + 7.6438e-5 * temperature.square()
        - 8.2467e-7 * temperature.pow(3)
        + 5.3875e-9 * temperature.pow(4)
    )
    salinity_power_coefficient = (
        -5.72466e-3
        + 1.0227e-4 * temperature
        - 1.6546e-6 * temperature.square()
    )

    return (
        pure_water_density
        + salinity_linear_coefficient * salinity
        + salinity_power_coefficient * salinity.pow(1.5)
        + 4.8314e-4 * salinity.square()
    )


def mackenzie_sound_speed(temperature: Tensor, salinity: Tensor, depth: Tensor) -> Tensor:
    """Return Mackenzie (1981) seawater sound speed in m s-1."""
    return (
        1448.96
        + 4.591 * temperature
        - 5.304e-2 * temperature.square()
        + 2.374e-4 * temperature.pow(3)
        + 1.340 * (salinity - 35.0)
        + 1.630e-2 * depth
        + 1.675e-7 * depth.square()
        - 1.025e-2 * temperature * (salinity - 35.0)
        - 7.139e-13 * temperature * depth.pow(3)
    )


def density_stability_loss(
    density: Tensor,
    depths: Tensor,
    tolerance: float = 0.0,
) -> Tensor:
    """Penalize negative potential-density gradients for depth-positive-down data."""
    _validate_profile_shape(density, depths)
    density_gradient = torch.diff(density, dim=-1) / torch.diff(depths)
    unstable_gradient = F.relu(-density_gradient - tolerance)
    return (1000.0 * unstable_gradient).square().mean()


def buoyancy_frequency_squared(
    density: Tensor,
    depths: Tensor,
    gravity: float = 9.80665,
) -> Tensor:
    """Approximate layer-centred N-squared from potential density."""
    _validate_profile_shape(density, depths)
    density_gradient = torch.diff(density, dim=-1) / torch.diff(depths)
    layer_density = 0.5 * (density[..., 1:] + density[..., :-1])
    return gravity * density_gradient / layer_density


def soft_mixed_layer_depth(
    density: Tensor,
    depths: Tensor,
    reference_depth: float = 10.0,
    density_threshold: float = 0.03,
    transition_width: float = 0.005,
) -> tuple[Tensor, Tensor]:
    """Estimate differentiable MLD using first density-threshold crossing."""
    _validate_profile_shape(density, depths)
    reference_matches = torch.isclose(
        depths,
        depths.new_tensor(reference_depth),
        atol=1e-6,
        rtol=0.0,
    )
    if not reference_matches.any():
        raise ValueError(f"reference depth {reference_depth} is absent")

    reference_index = int(reference_matches.nonzero(as_tuple=False)[0].item())
    if reference_index >= depths.numel() - 1:
        raise ValueError("reference depth must not be the deepest level")

    candidate_depths = depths[reference_index + 1 :]
    density_change = (
        density[..., reference_index + 1 :]
        - density[..., reference_index, None]
    )
    crossing_probability = torch.sigmoid(
        (density_change - density_threshold) / transition_width
    )
    preceding_survival = torch.cumprod(
        torch.cat(
            [
                torch.ones_like(crossing_probability[..., :1]),
                1.0 - crossing_probability[..., :-1] + 1e-7,
            ],
            dim=-1,
        ),
        dim=-1,
    )
    first_crossing_probability = crossing_probability * preceding_survival
    no_crossing_probability = torch.prod(
        1.0 - crossing_probability + 1e-7,
        dim=-1,
    )
    expected_depth = (
        first_crossing_probability * candidate_depths
    ).sum(dim=-1) + no_crossing_probability * candidate_depths[-1]
    return expected_depth, crossing_probability


def thermocline_gradient_loss(
    predicted_temperature: Tensor,
    target_temperature: Tensor,
    depths: Tensor,
    maximum_depth: float = 300.0,
) -> Tensor:
    """Match observed upper-ocean gradients while emphasizing the thermocline."""
    _validate_profile_shape(predicted_temperature, depths)
    _validate_profile_shape(target_temperature, depths)
    depth_intervals = torch.diff(depths)
    predicted_gradient = torch.diff(predicted_temperature, dim=-1) / depth_intervals
    target_gradient = torch.diff(target_temperature, dim=-1) / depth_intervals
    midpoint_depths = 0.5 * (depths[:-1] + depths[1:])
    upper_ocean = (midpoint_depths <= maximum_depth).to(target_temperature.dtype)
    target_strength = (100.0 * target_gradient.abs()).detach()
    normalizer = target_strength.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    thermocline_weight = (1.0 + 4.0 * target_strength / normalizer).clamp_max(10.0)
    thermocline_weight = thermocline_weight * upper_ocean
    gradient_error = F.smooth_l1_loss(
        100.0 * predicted_gradient,
        100.0 * target_gradient,
        reduction="none",
    )
    return (gradient_error * thermocline_weight).sum() / thermocline_weight.sum().clamp_min(1.0)


@dataclass(frozen=True)
class PhysicsLossWeights:
    temperature: float = 1.0
    salinity: float = 1.0
    stability: float = 0.05
    thermocline: float = 0.15
    salinity_gradient: float = 0.08
    mixed_layer_depth: float = 0.10
    surface_anchor: float = 0.03


class PhysicsInformedProfileLoss(nn.Module):
    """Joint T/S profile objective with differentiable water-column physics."""

    def __init__(
        self,
        temperature_scale: float | Sequence[float] = 1.0,
        salinity_scale: float | Sequence[float] = 1.0,
        depths: Sequence[float] = STANDARD_DEPTHS,
        weights: PhysicsLossWeights = PhysicsLossWeights(),
    ) -> None:
        super().__init__()
        depth_tensor = torch.as_tensor(depths, dtype=torch.float32)
        if depth_tensor.ndim != 1 or depth_tensor.numel() < 2:
            raise ValueError("depths must be a one-dimensional sequence")
        if not torch.all(torch.diff(depth_tensor) > 0):
            raise ValueError("depths must be strictly increasing")

        self.register_buffer("depths", depth_tensor)
        self.register_buffer(
            "temperature_scale",
            _profile_scale(temperature_scale, depth_tensor.numel()),
        )
        self.register_buffer(
            "salinity_scale",
            _profile_scale(salinity_scale, depth_tensor.numel()),
        )
        self.weights = weights

    def forward(
        self,
        predicted_theta0: Tensor,
        predicted_salinity: Tensor,
        target_theta0: Tensor,
        target_salinity: Tensor,
        surface_anchor: Tensor | None = None,
        physics_weight: float = 1.0,
    ) -> dict[str, Tensor]:
        for profile in (
            predicted_theta0,
            predicted_salinity,
            target_theta0,
            target_salinity,
        ):
            _validate_profile_shape(profile, self.depths)

        temperature_loss = F.smooth_l1_loss(
            (predicted_theta0 - target_theta0) / self.temperature_scale,
            torch.zeros_like(predicted_theta0),
        )
        salinity_loss = F.smooth_l1_loss(
            (predicted_salinity - target_salinity) / self.salinity_scale,
            torch.zeros_like(predicted_salinity),
        )

        predicted_density = eos80_potential_density(
            predicted_salinity,
            predicted_theta0,
        )
        target_density = eos80_potential_density(target_salinity, target_theta0)
        stability_loss = density_stability_loss(predicted_density, self.depths)
        thermocline_loss = thermocline_gradient_loss(
            predicted_theta0,
            target_theta0,
            self.depths,
        )

        depth_intervals = torch.diff(self.depths)
        predicted_salinity_gradient = torch.diff(
            predicted_salinity,
            dim=-1,
        ) / depth_intervals
        target_salinity_gradient = torch.diff(
            target_salinity,
            dim=-1,
        ) / depth_intervals
        salinity_gradient_loss = F.smooth_l1_loss(
            100.0 * predicted_salinity_gradient,
            100.0 * target_salinity_gradient,
        )

        predicted_mld, predicted_crossing = soft_mixed_layer_depth(
            predicted_density,
            self.depths,
        )
        with torch.no_grad():
            target_mld, target_crossing = soft_mixed_layer_depth(
                target_density,
                self.depths,
            )
        mixed_layer_depth_loss = F.smooth_l1_loss(
            predicted_mld / 100.0,
            target_mld / 100.0,
        ) + 0.25 * F.mse_loss(predicted_crossing, target_crossing)

        surface_loss = predicted_theta0.new_zeros(())
        if surface_anchor is not None:
            if surface_anchor.shape != predicted_theta0.shape[:-1] + (2,):
                raise ValueError("surface_anchor must have shape [..., 2]")
            surface_loss = F.smooth_l1_loss(
                (predicted_theta0[..., 0] - surface_anchor[..., 0]) / 0.5,
                torch.zeros_like(predicted_theta0[..., 0]),
            ) + F.smooth_l1_loss(
                (predicted_salinity[..., 0] - surface_anchor[..., 1]) / 0.2,
                torch.zeros_like(predicted_salinity[..., 0]),
            )

        data_loss = (
            self.weights.temperature * temperature_loss
            + self.weights.salinity * salinity_loss
        )
        physics_loss = (
            self.weights.stability * stability_loss
            + self.weights.thermocline * thermocline_loss
            + self.weights.salinity_gradient * salinity_gradient_loss
            + self.weights.mixed_layer_depth * mixed_layer_depth_loss
            + self.weights.surface_anchor * surface_loss
        )
        total_loss = data_loss + physics_weight * physics_loss

        return {
            "total": total_loss,
            "data": data_loss,
            "physics": physics_loss,
            "temperature": temperature_loss,
            "salinity": salinity_loss,
            "stability": stability_loss,
            "thermocline": thermocline_loss,
            "salinity_gradient": salinity_gradient_loss,
            "mixed_layer_depth": mixed_layer_depth_loss,
            "surface_anchor": surface_loss,
        }


def _profile_scale(scale: float | Sequence[float], depth_count: int) -> Tensor:
    scale_tensor = torch.as_tensor(scale, dtype=torch.float32)
    if scale_tensor.ndim == 0:
        scale_tensor = scale_tensor.repeat(depth_count)
    if scale_tensor.shape != (depth_count,):
        raise ValueError(f"scale must be scalar or contain {depth_count} values")
    if not torch.all(scale_tensor > 0):
        raise ValueError("all scale values must be positive")
    return scale_tensor


def _validate_profile_shape(profile: Tensor, depths: Tensor) -> None:
    if profile.ndim < 1 or profile.shape[-1] != depths.numel():
        raise ValueError(
            f"profile final dimension must contain {depths.numel()} depth levels"
        )
