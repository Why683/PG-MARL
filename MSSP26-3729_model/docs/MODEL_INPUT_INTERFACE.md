# Model input interface

Preprocessing and sensor-rate synchronization code are intentionally outside
this model-only repository.  Each manifest row points to one already prepared
2.0-s `.npz` classification record containing:

- `sequence`: float32 `[200, 6, C]`, the manuscript-defined 100-Hz TASN input;
- `mechanism_features`: float32 `[37]`, comprising 15 residual statistics,
  four friction descriptors and 18 order-domain descriptors;
- `residual_sequence`: float32 `[200]`, used only by the offline MMPS;
- `label`: scalar int64 in the order Wear, Crack, Broken, Normal, Pitting.

The architecture never concatenates the 1-kHz vibration waveform with the
100-Hz joint sequence into a common raw tensor.  The vibration-dependent
order descriptors are calculated by the manuscript-defined angular-domain
procedure in `pgmarl/physics.py` before entering MMPS.
