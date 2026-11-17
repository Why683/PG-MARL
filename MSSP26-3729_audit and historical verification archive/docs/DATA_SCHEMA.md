# Fixed window record schema

Each `.npz` record referenced by `window_manifest.csv` contains one immutable
2.0 s classification unit:

- `sequence`: float32 `[200, 6, C]`, synchronized 100 Hz joint/internal/IMU
  features in the manuscript channel order;
- `theta`: float32 `[200]`, joint-2 motor encoder angle in rad;
- `theta_dot`: float32 `[200]`, motor angular velocity in rad/s;
- `theta_ddot`: float32 `[200]`, derivative used inside MMPS only;
- `tau_true`: float32 `[200]`, motor-side measured torque in N m;
- `vibration`: float32 `[2000, 3]`, cleaned effective 1 kHz vibration;
- `mechanism_features`: float32 `[F]`, descriptors generated solely from the
  preceding arrays and fixed source-fitted invalid-order medians;
- `residual_sequence`: float32 `[200]`, the motor-side torque-related residual
  used only by the offline MMPS peak-preserving morphology CNN;
- `label`: scalar int64 in the public class order Wear, Crack, Broken, Normal,
  Pitting;
- `domain`, `trial_id`, `window_index`, `start_s`, `end_s`: immutable identity
  metadata.

The manifest, rather than a random crop generator, defines every window used in
training, validation and testing.  Mini-batch randomness changes only the order
in which fixed source-training records are sampled.
