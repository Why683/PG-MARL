# Historical raw adapter

This retained parser documents the actual BORUNTE CSV/channel names. Its sampling and window defaults belong to the historical archive. For the 11.16 protocol, use it only to read raw signals, then resample internal/IMU values to exactly 200 states, retain 2,000 vibration samples, and write the immutable records defined in `docs/DATA_SCHEMA.md`.
