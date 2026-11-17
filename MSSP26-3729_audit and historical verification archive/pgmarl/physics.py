from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import get_window


@dataclass(frozen=True)
class Mechanics:
    gear_ratio_n: float = 121.0
    motor_inertia_kg_m2: float = 8.70e-4
    stiffness_output_Nm_per_rad: float = 3.71277e5
    damping_output_Nm_s_per_rad: float = 219.173
    coulomb_friction_Nm: float = 0.45
    static_friction_Nm: float = 0.70
    stribeck_velocity_rad_s: float = 0.10
    viscous_friction_Nm_s_per_rad: float = 0.010
    moving_trend_samples: int = 11


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width % 2 != 1:
        raise ValueError("moving-trend width must be odd")
    pad = width // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def motor_side_residual(theta: np.ndarray, theta_dot: np.ndarray,
                        theta_ddot: np.ndarray, tau_true: np.ndarray,
                        mechanics: Mechanics) -> Dict[str, np.ndarray]:
    trend_theta = moving_average(theta, mechanics.moving_trend_samples)
    trend_speed = moving_average(theta_dot, mechanics.moving_trend_samples)
    deformation = theta - trend_theta
    deformation_rate = theta_dot - trend_speed
    reflected = mechanics.gear_ratio_n ** 2
    tau_flex = (
        mechanics.stiffness_output_Nm_per_rad * deformation
        + mechanics.damping_output_Nm_s_per_rad * deformation_rate
    ) / reflected
    ratio = np.clip(
        np.abs(theta_dot) / max(mechanics.stribeck_velocity_rad_s, 1e-8),
        0.0, 50.0)
    tau_friction = (
        mechanics.coulomb_friction_Nm
        + (mechanics.static_friction_Nm - mechanics.coulomb_friction_Nm)
        * np.exp(-(ratio ** 2))
    ) * np.sign(theta_dot) + mechanics.viscous_friction_Nm_s_per_rad * theta_dot
    tau_inertia = mechanics.motor_inertia_kg_m2 * theta_ddot
    tau_theory = tau_inertia + tau_flex + tau_friction
    residual = tau_true - tau_theory
    return {
        "tau_flex": tau_flex,
        "tau_friction": tau_friction,
        "tau_inertia": tau_inertia,
        "tau_theory": tau_theory,
        "residual": residual,
    }


def residual_shape_features(residual: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64)
    abs_mean = np.mean(np.abs(residual)) + epsilon
    rms = np.sqrt(np.mean(residual ** 2) + epsilon)
    peak = np.max(np.abs(residual))
    crest = peak / rms
    impulse = peak / abs_mean
    margin = peak / (np.mean(np.sqrt(np.abs(residual) + 1e-6)) ** 2 + epsilon)
    shape = rms / abs_mean
    centered = residual - residual.mean()
    std = centered.std() + epsilon
    normalized = np.clip(centered / std, -20.0, 20.0)
    magnitude = np.abs(np.fft.rfft(residual / rms))
    return np.asarray([
        residual.mean(), abs_mean, residual.var(), std, rms, peak,
        residual.max() - residual.min(), crest, impulse, margin, shape,
        np.mean(normalized ** 4), np.mean(normalized ** 3),
        magnitude.mean(), magnitude.max(),
    ], dtype=np.float32)


def angular_segments(theta_dot: np.ndarray, vibration: np.ndarray,
                     internal_rate_hz: float = 100.0,
                     vibration_rate_hz: float = 1000.0,
                     speed_threshold_rad_s: float = 0.05,
                     reversal_guard_s: float = 0.05) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
    """Return valid monotone motor-travel segments without bridging gaps."""
    speed = np.asarray(theta_dot, dtype=np.float64)
    vib = np.asarray(vibration, dtype=np.float64)
    if vib.ndim == 1:
        vib = vib[:, None]
    internal_time = np.arange(speed.size) / internal_rate_hz
    vibration_time = np.arange(vib.shape[0]) / vibration_rate_hz
    speed_vibration = np.interp(vibration_time, internal_time, speed)
    moving = np.abs(speed_vibration) > speed_threshold_rad_s
    sign = np.sign(speed_vibration)
    reversals = np.flatnonzero(sign[1:] * sign[:-1] < 0) + 1
    guard = int(round(reversal_guard_s * vibration_rate_hz))
    for center in reversals:
        moving[max(0, center - guard):min(moving.size, center + guard + 1)] = False
    segments = []
    start = None
    for index, valid in enumerate(np.r_[moving, False]):
        if valid and start is None:
            start = index
        elif not valid and start is not None:
            if index - start > 1:
                local_speed = speed_vibration[start:index]
                travel = np.cumsum(np.abs(local_speed)) / (
                    2.0 * np.pi * vibration_rate_hz)
                travel -= travel[0]
                if travel[-1] >= 1.0:
                    segments.append((travel, vib[start:index]))
            start = None
    return segments


def order_band_features(theta_dot: np.ndarray, vibration: np.ndarray,
                        characteristic_orders: Sequence[float],
                        internal_rate_hz: float = 100.0,
                        vibration_rate_hz: float = 1000.0,
                        samples_per_revolution: int = 128,
                        harmonics: Iterable[int] = (1, 2, 3),
                        invalid_fill: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
    segments = angular_segments(
        theta_dot, vibration, internal_rate_hz, vibration_rate_hz)
    if not segments:
        if invalid_fill is None:
            size = len(characteristic_orders) * len(tuple(harmonics)) + 3
            return np.full(size, np.nan, dtype=np.float32), False
        return np.asarray(invalid_fill, dtype=np.float32), False
    harmonic_values = []
    travel_weights = []
    for travel, signal in segments:
        total_travel = float(travel[-1])
        count = max(int(np.floor(total_travel * samples_per_revolution)), 2)
        uniform = np.arange(count, dtype=np.float64) / samples_per_revolution
        resampled = np.column_stack([
            np.interp(uniform, travel, signal[:, channel])
            for channel in range(signal.shape[1])
        ])
        windowed = resampled * get_window("hann", count, fftbins=True)[:, None]
        power = np.abs(np.fft.rfft(windowed, axis=0)) ** 2
        power = power.mean(axis=1)
        orders = np.fft.rfftfreq(count, d=1.0 / samples_per_revolution)
        denominator = power[orders > 0].sum()
        if denominator <= 1e-12:
            continue
        half_width = max(0.12, 1.5 / total_travel)
        values = []
        for base in characteristic_orders:
            for harmonic in harmonics:
                center = float(base) * int(harmonic)
                mask = np.abs(orders - center) <= half_width
                values.append(float(power[mask].sum() / denominator))
        first = np.asarray(values, dtype=np.float64)
        ratio_21 = first[1::3].sum() / (first[0::3].sum() + 1e-12)
        ratio_31 = first[2::3].sum() / (first[0::3].sum() + 1e-12)
        impulsiveness = float(np.max(np.abs(resampled)) / (
            np.sqrt(np.mean(resampled ** 2)) + 1e-12))
        harmonic_values.append(np.r_[first, ratio_21, ratio_31, impulsiveness])
        travel_weights.append(total_travel)
    if not harmonic_values:
        if invalid_fill is None:
            size = len(characteristic_orders) * len(tuple(harmonics)) + 3
            return np.full(size, np.nan, dtype=np.float32), False
        return np.asarray(invalid_fill, dtype=np.float32), False
    return np.average(
        np.asarray(harmonic_values), axis=0,
        weights=np.asarray(travel_weights)).astype(np.float32), True


def mechanism_feature_vector(theta: np.ndarray, theta_dot: np.ndarray,
                             theta_ddot: np.ndarray, tau_true: np.ndarray,
                             vibration: np.ndarray,
                             characteristic_orders: Sequence[float],
                             invalid_order_fill: Optional[np.ndarray] = None,
                             mechanics: Mechanics = Mechanics()) -> Tuple[np.ndarray, bool]:
    residual = motor_side_residual(
        theta, theta_dot, theta_ddot, tau_true, mechanics)["residual"]
    residual_features = residual_shape_features(residual)
    order_features, available = order_band_features(
        theta_dot, vibration, characteristic_orders,
        invalid_fill=invalid_order_fill)
    friction = motor_side_residual(
        theta, theta_dot, theta_ddot, tau_true, mechanics)["tau_friction"]
    friction_features = np.asarray([
        np.mean(friction), np.std(friction), np.sqrt(np.mean(friction ** 2)),
        np.max(np.abs(friction)),
    ], dtype=np.float32)
    return np.concatenate([
        residual_features, friction_features, order_features]).astype(np.float32), available

