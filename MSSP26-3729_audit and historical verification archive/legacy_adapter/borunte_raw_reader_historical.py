import os
import re
import random
import numpy as np
import pandas as pd


# The order-domain descriptor is deliberately richer than the legacy five
# single-line matched-chirp coefficients.  A tooth defect modulates several
# harmonics and sidebands; retaining only the centre line made the physical
# representation low-rank and strongly trajectory dependent.
PHYSICS_PRIOR_NAMES = (
    'planetary_rotation_h1', 'planetary_rotation_h2', 'planetary_rotation_h3',
    'first_stage_mesh_h1', 'first_stage_mesh_h2', 'first_stage_mesh_h3',
    'second_stage_mesh_h1', 'second_stage_mesh_h2', 'second_stage_mesh_h3',
    'first_mesh_minus_planet_h1', 'first_mesh_minus_planet_h2',
    'first_mesh_minus_planet_h3', 'first_mesh_plus_planet_h1',
    'first_mesh_plus_planet_h2', 'first_mesh_plus_planet_h3',
    'first_mesh_harmonic_ratio', 'second_mesh_harmonic_ratio',
    'order_domain_impulsiveness',
)

# Keep dataset/protocol inspection usable on machines without a PyTorch
# environment. Tensor conversion functions below still require PyTorch.
try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - used by the protocol audit script
    torch = None
    F = None


class RealBorunteTrial:
    def __init__(self, trial_dir, window_time=2.0):
        self.trial_dir = trial_dir
        self.window_time = window_time
        self.fs_int = None
        files = os.listdir(trial_dir)
        self.int_path = next((os.path.join(trial_dir, f) for f in files if 'internal' in f.lower()), None)
        self.imu_path = next((os.path.join(trial_dir, f) for f in files if 'imu' in f.lower()), None)
        cleaned_vib = next((f for f in files if 'vib' in f.lower() and '_cleaned' in f.lower()), None)
        self.is_cleaned_vib = bool(cleaned_vib)
        self.vib_path = os.path.join(trial_dir, cleaned_vib) if cleaned_vib else os.path.join(trial_dir, next(
            (f for f in files if 'vib' in f.lower()), None))
        if not all([self.int_path, self.imu_path, self.vib_path]): raise FileNotFoundError(f"缺少文件: {trial_dir}")
        self._cached_data = None

    def _read_csv_robust(self, filepath, **kwargs):
        # Dataset CSV files are UTF-8. Trying GB18030 first can occasionally
        # decode them without raising while corrupting Chinese column names.
        for enc in ['utf-8-sig', 'utf-8', 'gb18030', 'gbk']:
            try:
                return pd.read_csv(filepath, encoding=enc, **kwargs)
            except Exception:
                continue
        return pd.read_csv(filepath, encoding='gb18030', encoding_errors='ignore', **kwargs)

    def _read_lines_robust(self, filepath, max_lines=None):
        """Read text without silently losing timestamps because of encoding."""
        last_error = None
        for enc in ['utf-8-sig', 'utf-8', 'gb18030', 'gbk']:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    if max_lines is None:
                        return f.readlines()
                    return [f.readline() for _ in range(max_lines)]
            except (UnicodeDecodeError, OSError) as exc:
                last_error = exc
        raise RuntimeError(f"Unable to read text file {filepath}: {last_error}")

    def _parse_time_to_seconds(self, time_str, mode='int'):
        try:
            time_str = str(time_str).strip().replace('"', '')
            if mode == 'int':
                m = re.search(r'_(\d+)-(\d+)-(\d+)_(\d+)', time_str)
                if m: return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + float(
                    m.group(4)) / 1000.0
            elif mode == 'imu':
                m = re.search(r'(\d+):(\d+):(\d+(?:\.\d+)?)', time_str)
                if m: return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        except Exception:
            pass
        return 0.0

    def _load_data(self):
        if self._cached_data is not None: return self._cached_data
        df_int = pd.read_excel(self.int_path) if self.int_path.endswith(('.xlsx', '.xls')) else self._read_csv_robust(
            self.int_path)
        df_int.columns = [str(c).strip().replace('"', '') for c in df_int.columns]
        time_int = np.array([self._parse_time_to_seconds(t, 'int') for t in
                             df_int['timestamp']]) if 'timestamp' in df_int.columns else np.zeros(len(df_int))
        valid_int_dt = np.diff(time_int)
        valid_int_dt = valid_int_dt[(valid_int_dt > 0) & (valid_int_dt < 1.0)]
        self.fs_int = 1.0 / np.median(valid_int_dt) if len(valid_int_dt) else 81.0

        int_data = [df_int[[f'axis-{i}', f'curSpeed-{i}', f'curTorque-{i}']].values if len([c for c in df_int.columns if
                                                                                            f'axis-{i}' in c or f'curSpeed-{i}' in c or f'curTorque-{i}' in c]) >= 3 else np.zeros(
            (len(df_int), 3)) for i in range(6)]
        internal_array = np.stack(int_data, axis=1)

        imu_data_list, imu_time_list = [[] for _ in range(6)], [[] for _ in range(6)]
        lines = self._read_lines_robust(self.imu_path)

        if 'lines' in locals() and lines:
            for line in lines[1:]:
                chunks = [c.strip() for c in line.replace('\t', ',').replace('"', '').split(',')]
                dev_idx = next((idx for idx, c in enumerate(chunks[:5]) if 'COM' in c), -1)
                if dev_idx != -1:
                    m = re.search(r'5(\d)$', chunks[dev_idx])
                    if m and 0 <= int(m.group(1)) <= 5:
                        dev_id = int(m.group(1))
                        start_idx = dev_idx + 1 + (1 if dev_idx + 1 < len(chunks) and (
                                    '2026' in chunks[dev_idx + 1] or '-' in chunks[dev_idx + 1]) else 0)
                        raw_feats = [float(x) for x in chunks[start_idx:] if x != ''][:9]
                        if len(raw_feats) == 9:
                            imu_data_list[dev_id].append(raw_feats[6:9] + raw_feats[3:6] + raw_feats[0:3])
                            imu_time_list[dev_id].append(
                                self._parse_time_to_seconds(chunks[dev_idx - 1] if dev_idx > 0 else chunks[0], 'imu'))

        min_len = min([len(d) for d in imu_data_list if len(d) > 0] or [0])
        if min_len == 0:
            imu_array, time_imu, self.fs_imu = np.zeros((1000, 6, 9)), np.zeros(1000), 500.0
        else:
            imu_array = np.stack(
                [np.array(d[:min_len] if len(d) > 0 else np.zeros((min_len, 9)).tolist()) for d in imu_data_list],
                axis=1)
            time_imu = next((np.array(t[:min_len]) for t in imu_time_list if len(t) > 0), np.zeros(min_len))
            if len(time_imu) > 1 and time_imu[-1] > time_imu[0]:
                self.fs_imu = (len(time_imu) - 1) / (time_imu[-1] - time_imu[0])
            else:
                self.fs_imu = 50.0

        vib_base_time, self.fs_vib = 0.0, 10000.0
        try:
            raw_vib_path = self.vib_path.replace('_cleaned', '') if self.is_cleaned_vib else self.vib_path
            for line in self._read_lines_robust(raw_vib_path, max_lines=10):
                if '起始时间' in line or 'Start' in line:
                    m = re.search(r'(\d+):(\d+):(\d+(?:\.\d+)?)', line)
                    if m:
                        vib_base_time = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
                if '采样率' in line or 'Rate' in line:
                    m = re.search(r'(\d+(?:\.\d+)?)', line)
                    if m:
                        self.fs_vib = float(m.group(1))
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to parse vibration metadata in {self.trial_dir}: {exc}") from exc

        df_vib = self._read_csv_robust(self.vib_path) if getattr(self, 'is_cleaned_vib',
                                                                 False) else self._read_csv_robust(self.vib_path,
                                                                                                   skiprows=6)
        # Cleaned files retain the original sample index. This correctly
        # identifies both 10 kHz -> 1 kHz decimation and native 1 kHz files.
        sample_col = next((c for c in df_vib.columns if '采样点' in str(c)), None)
        if sample_col is not None:
            sample_indices = pd.to_numeric(df_vib[sample_col], errors='coerce').to_numpy()
            if np.isnan(sample_indices).any():
                sample_indices = np.arange(len(df_vib), dtype=float)
        else:
            sample_indices = np.arange(len(df_vib), dtype=float)
        sample_steps = np.diff(sample_indices)
        sample_steps = sample_steps[sample_steps > 0]
        index_step = float(np.median(sample_steps)) if len(sample_steps) else 1.0
        raw_fs_vib = self.fs_vib
        self.fs_vib = raw_fs_vib / index_step
        time_vib = vib_base_time + sample_indices / raw_fs_vib

        vib2_cols = [c for c in df_vib.columns if 'AI0' in c or 'AI1' in c or 'AI2' in c][:3]
        vib4_cols = [c for c in df_vib.columns if 'AI4' in c or 'AI5' in c or 'AI6' in c][:3]
        vib2_array = df_vib[vib2_cols].values if len(vib2_cols) == 3 else np.zeros(
            (len(df_vib) if len(df_vib) > 0 else 1000, 3))
        vib4_array = df_vib[vib4_cols].values if len(vib4_cols) == 3 else np.zeros(
            (len(df_vib) if len(df_vib) > 0 else 1000, 3))

        self.len_int, self.len_imu, self.len_vib = round(self.fs_int * self.window_time), round(
            self.fs_imu * self.window_time), round(self.fs_vib * self.window_time)
        self.global_start = max(time_int[0] if len(time_int) > 0 else 0, time_imu[0] if len(time_imu) > 0 else 0,
                                time_vib[0] if len(time_vib) > 0 else 0)
        self.global_end = min([t[-1] for t in [time_int, time_imu, time_vib] if len(t) > 0] or [0])
        if self.global_end <= self.global_start + self.window_time:
            raise ValueError(
                f"No valid synchronized {self.window_time:.1f}s interval in {self.trial_dir}: "
                f"start={self.global_start:.6f}, end={self.global_end:.6f}"
            )
        self._cached_data = (internal_array, time_int, imu_array, time_imu, vib2_array, vib4_array, time_vib)
        return self._cached_data

    def valid_window_starts(self, stride=None):
        """Return deterministic, non-overlapping starts for val/test use."""
        self._load_data()
        stride = self.window_time if stride is None else float(stride)
        if stride <= 0:
            raise ValueError("Window stride must be positive")
        last_start = self.global_end - self.window_time
        if last_start < self.global_start:
            return []
        count = int(np.floor((last_start - self.global_start) / stride)) + 1
        return [self.global_start + i * stride for i in range(count)]

    def get_window_at(self, t0):
        internal_array, time_int, imu_array, time_imu, vib2_array, vib4_array, time_vib = self._load_data()
        t0 = float(t0)
        if t0 < self.global_start - 1e-9 or t0 + self.window_time > self.global_end + 1e-9:
            raise ValueError(f"Window [{t0}, {t0 + self.window_time}] is outside synchronized interval "
                             f"[{self.global_start}, {self.global_end}] in {self.trial_dir}")
        idx_int, idx_imu, idx_vib = (np.searchsorted(t, t0) if len(t) > 0 else 0 for t in
                                     (time_int, time_imu, time_vib))

        w_int = internal_array[idx_int: idx_int + self.len_int]
        w_imu = imu_array[idx_imu: idx_imu + self.len_imu]
        w_vib2 = vib2_array[idx_vib: idx_vib + self.len_vib]
        w_vib4 = vib4_array[idx_vib: idx_vib + self.len_vib]

        w_int = np.pad(w_int, ((0, max(0, self.len_int - len(w_int))), (0, 0), (0, 0)), mode='edge') if len(
            w_int) < self.len_int else w_int
        w_imu = np.pad(w_imu, ((0, max(0, self.len_imu - len(w_imu))), (0, 0), (0, 0)), mode='edge') if len(
            w_imu) < self.len_imu else w_imu
        w_vib2 = np.pad(w_vib2, ((0, max(0, self.len_vib - len(w_vib2))), (0, 0)), mode='edge') if len(
            w_vib2) < self.len_vib else w_vib2
        w_vib4 = np.pad(w_vib4, ((0, max(0, self.len_vib - len(w_vib4))), (0, 0)), mode='edge') if len(
            w_vib4) < self.len_vib else w_vib4

        return np.nan_to_num(w_int), np.nan_to_num(w_imu), np.nan_to_num(w_vib2), np.nan_to_num(w_vib4)

    def get_random_window(self):
        self._load_data()
        t0 = np.random.uniform(self.global_start, self.global_end - self.window_time)
        return self.get_window_at(t0)


class BorunteDatasetManager:
    # Preserve the manuscript/checkpoint label convention independently of
    # filesystem enumeration order (Linux and Windows return os.listdir in
    # different orders for this dataset).
    CANONICAL_CLASS_NAMES = ('Wear', 'Crack', 'Broken', 'Normal', 'Pitting')

    def __init__(self, root_dir, target_domains=None, eval_windows_per_trial=8,
                 physics_config=None, allowed_domains=None):
        self.root_dir = root_dir
        self.target_domains = set(target_domains or ['Domain_D_Trajectory'])
        # ``allowed_domains`` is a hard data-access guard for source-only model
        # development.  Domains outside this set are not registered and their
        # trial files are never opened.  This is stronger than merely omitting
        # Domain D from a training-domain list after the full dataset has been
        # scanned.
        self.allowed_domains = (None if allowed_domains is None
                                else set(allowed_domains))
        self.eval_windows_per_trial = int(eval_windows_per_trial)
        self.physics_config = physics_config
        self.registry = {}
        self.class_mapping = {}
        self.class_names_map = dict(enumerate(self.CANONICAL_CLASS_NAMES))
        self.current_class_id = len(self.CANONICAL_CLASS_NAMES)
        self._manifest_cache = {}
        self._scan_and_split_directory()

    def _scan_and_split_directory(self):
        print(f"开始扫描并严格划分数据集目录: {self.root_dir} ...")
        for domain in sorted(
                d for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
                and (self.allowed_domains is None or d in self.allowed_domains)):
            self.registry[domain] = {}
            domain_path = os.path.join(self.root_dir, domain)
            for cls_folder in sorted(
                    c for c in os.listdir(domain_path) if os.path.isdir(os.path.join(domain_path, c))):
                raw_class_idx, class_name = int(cls_folder.split('_')[0]), cls_folder.split('_')[-1]
                if class_name not in self.CANONICAL_CLASS_NAMES:
                    raise ValueError(
                        f"Unknown health-state folder {cls_folder}; expected one of "
                        f"{self.CANONICAL_CLASS_NAMES}")
                mapped_idx = self.CANONICAL_CLASS_NAMES.index(class_name)
                previous = self.class_mapping.setdefault(raw_class_idx, mapped_idx)
                if previous != mapped_idx:
                    raise ValueError(
                        f"Raw class prefix {raw_class_idx} maps to inconsistent names")
                cls_path = os.path.join(domain_path, cls_folder)
                trials = [RealBorunteTrial(os.path.join(cls_path, t)) for t in sorted(os.listdir(cls_path)) if
                          os.path.isdir(os.path.join(cls_path, t))]
                n = len(trials)
                if n == 0:
                    continue
                elif domain in self.target_domains:
                    # The zero-shot target domain must never be available to
                    # training or validation/model selection.
                    split_dict = {'train': [], 'val': [], 'test': trials}
                elif n == 1:
                    raise ValueError(f"At least three independent trials are required for source split: {cls_path}")
                elif n == 2:
                    raise ValueError(f"At least three independent trials are required for source split: {cls_path}")
                else:
                    # Freeze the already published/reviewer-audited trial IDs;
                    # do not let os.listdir order or a later RNG refactor alter
                    # the protocol. Three-trial Broken folders use 001/002/003
                    # for train/validation/test. Four- or five-trial folders
                    # reserve 003 for validation and 002 for source testing.
                    by_id = {os.path.basename(trial.trial_dir): trial for trial in trials}
                    val_id, test_id = (('trial_002', 'trial_003') if n == 3
                                       else ('trial_003', 'trial_002'))
                    missing_ids = [trial_id for trial_id in (val_id, test_id)
                                   if trial_id not in by_id]
                    if missing_ids:
                        raise ValueError(
                            f"Frozen split IDs missing in {cls_path}: {missing_ids}")
                    split_dict = {
                        'train': [by_id[trial_id] for trial_id in sorted(by_id)
                                  if trial_id not in {val_id, test_id}],
                        'val': [by_id[val_id]],
                        'test': [by_id[test_id]],
                    }
                self.registry[domain][mapped_idx] = split_dict

    def sample_trial(self, domain_list, class_idx=None, mode='train'):
        domain = np.random.choice(domain_list)
        available_classes = list(self.registry[domain].keys())
        if len(available_classes) == 0: raise ValueError(f"工况 {domain} 下无数据！")
        if class_idx is None or class_idx not in available_classes: class_idx = np.random.choice(available_classes)
        pool = self.registry[domain][class_idx].get(mode, [])
        if not pool:
            raise ValueError(f"No {mode} trials for domain={domain}, class={class_idx}; refusing split fallback")
        return np.random.choice(pool), domain, class_idx

    def build_window_manifest(self, domain_list, mode='test', stride=2.0):
        """Build a fixed trial-disjoint window manifest for evaluation.

        Validation and test windows use a 2 s stride by default, so windows do
        not overlap. The returned order is stable across runs.
        """
        cache_key = (tuple(domain_list), mode, float(stride), self.eval_windows_per_trial)
        if cache_key in self._manifest_cache:
            return self._manifest_cache[cache_key]

        manifest = []
        for domain in domain_list:
            if domain not in self.registry:
                raise KeyError(f"Unknown domain: {domain}")
            for class_idx in sorted(self.registry[domain]):
                trials = self.registry[domain][class_idx].get(mode, [])
                if not trials:
                    raise ValueError(f"No {mode} trials for domain={domain}, class={class_idx}")
                for trial in sorted(trials, key=lambda x: os.path.basename(x.trial_dir)):
                    starts = trial.valid_window_starts(stride=stride)
                    if mode in {'val', 'test'} and len(starts) > self.eval_windows_per_trial:
                        keep = np.linspace(0, len(starts) - 1, self.eval_windows_per_trial, dtype=int)
                        starts = [starts[i] for i in keep]
                    for window_idx, start_time in enumerate(starts):
                        manifest.append({
                            'domain': domain,
                            'class_idx': class_idx,
                            'class_name': self.class_names_map[class_idx],
                            'trial_id': os.path.basename(trial.trial_dir),
                            'trial_dir': trial.trial_dir,
                            'trial': trial,
                            'window_idx': window_idx,
                            'start_time': float(start_time),
                            'end_time': float(start_time + trial.window_time),
                            'split': mode,
                        })
        self._manifest_cache[cache_key] = manifest
        return manifest


class PhysicsDataProcessor:
    def __init__(self, physics_config):
        if physics_config is None:
            raise ValueError("A validated physics_config is required for physics-prior extraction")
        self.config = physics_config
        self.gear_ratio = float(physics_config['gear_ratio_n'])
        teeth = physics_config['rv_teeth']
        self.z1, self.z2, self.z3, self.z4 = (float(teeth[key]) for key in (
            'central_axis_z1', 'planetary_gear_z2', 'cycloid_gear_z3', 'pin_wheel_z4'))
        self.angle_scale = np.pi / 180.0 if physics_config['encoder_angle_unit'] == 'degree' else 1.0
        self.encoder_to_motor_scale = float(physics_config.get('encoder_to_motor_angle_scale', 1.0))
        self.torque_unit = physics_config.get('torque_unit')
        self.torque_count_per_rated = physics_config.get('torque_count_per_rated_torque')
        self.motor_rated_torque = physics_config.get('axis2_motor_rated_torque_Nm')
        self.torque_reference_frame = physics_config.get('torque_reference_frame')

    def rv_frequency_tracks(self, motor_speed_rad_s):
        """Return instantaneous RV40E characteristic-frequency tracks in Hz.

        Axis-1 is the joint-2 motor-side encoder.  The RV40E-121 structure is
        z1/z2/z3/z4 = 12/36/39/40.  The first-stage mesh is the physically
        relevant band for the seeded planetary-gear tooth faults; the second
        RV-stage mesh is retained as a structural comparison band.
        """
        motor_rotation_hz = np.abs(np.asarray(motor_speed_rad_s, dtype=np.float64)) / (2.0 * np.pi)
        denominator = self.z1 + self.z2 * self.z4
        output_rotation_hz = self.z1 / denominator * motor_rotation_hz
        planetary_rotation_hz = np.abs(
            self.z1 * self.z4 / ((self.z3 - self.z4) * denominator)) * motor_rotation_hz
        first_stage_mesh_hz = self.z1 * self.z2 * self.z4 / denominator * motor_rotation_hz
        second_stage_mesh_hz = self.z1 * self.z3 * self.z4 / denominator * motor_rotation_hz
        return {
            'motor_rotation': motor_rotation_hz,
            'output_rotation': output_rotation_hz,
            'planetary_rotation': planetary_rotation_hz,
            'first_stage_mesh': first_stage_mesh_hz,
            'second_stage_mesh': second_stage_mesh_hz,
            'first_mesh_minus_planet_rotation': np.abs(
                first_stage_mesh_hz - planetary_rotation_hz),
            'first_mesh_plus_planet_rotation': first_stage_mesh_hz + planetary_rotation_hz,
        }

    def rv_characteristic_orders(self):
        """Return RV40E characteristic orders per motor revolution.

        Orders are geometry-only quantities.  Expressing the prior in this
        coordinate removes the absolute speed of trajectories A--D from the
        diagnostic descriptor.
        """
        denominator = self.z1 + self.z2 * self.z4
        planetary = abs(self.z1 * self.z4 / ((self.z3 - self.z4) * denominator))
        first_mesh = self.z1 * self.z2 * self.z4 / denominator
        second_mesh = self.z1 * self.z3 * self.z4 / denominator
        return {
            'planetary_rotation': planetary,
            'first_stage_mesh': first_mesh,
            'second_stage_mesh': second_mesh,
            'first_mesh_minus_planet': abs(first_mesh - planetary),
            'first_mesh_plus_planet': first_mesh + planetary,
        }

    def encoder_kinematics(self, angle_values, fs_internal):
        """Derive signed motor velocity and acceleration from encoder angle.

        The supplied ``curSpeed`` channel is non-negative and therefore cannot
        supply the sign needed by the inertia and friction terms.
        """
        theta = (np.asarray(angle_values, dtype=np.float64) * self.angle_scale
                 * self.encoder_to_motor_scale)
        if len(theta) < 3:
            return theta, np.zeros_like(theta), np.zeros_like(theta)
        kernel_size = min(5, len(theta) if len(theta) % 2 == 1 else len(theta) - 1)
        kernel_size = max(kernel_size, 1)
        pad = kernel_size // 2
        theta_smooth = np.convolve(
            np.pad(theta, (pad, pad), mode='edge'),
            np.ones(kernel_size, dtype=np.float64) / kernel_size,
            mode='valid')
        dt = 1.0 / float(fs_internal)
        theta_dot = np.gradient(theta_smooth, dt)
        theta_ddot = np.gradient(theta_dot, dt)
        return theta, theta_dot, theta_ddot

    def torque_to_motor_side(self, torque_values):
        """Convert verified N*m or normalized counts to motor-side N*m."""
        torque = np.asarray(torque_values, dtype=np.float64)
        if self.torque_unit == 'normalized_count':
            if self.motor_rated_torque is None or not self.torque_count_per_rated:
                raise ValueError("Normalized curTorque conversion requires motor rated torque")
            torque = torque * float(self.motor_rated_torque) / float(self.torque_count_per_rated)
        elif self.torque_unit != 'N*m':
            raise ValueError("torque_unit is unresolved")
        if self.torque_reference_frame == 'motor_side':
            return torque
        if self.torque_reference_frame == 'output_side':
            return torque / self.gear_ratio
        raise ValueError(
            "torque_reference_frame is unresolved; verify whether curTorque-1 is motor- or output-side N*m")

    def extract_physics_priors(self, motor_speed_rad_s, vib_array, reducer_type, fs_vib):
        """Extract speed-invariant RV features by angular resampling.

        The vibration is resampled against cumulative *absolute* motor travel,
        so a reversal continues the mechanical passage count instead of
        cancelling the preceding phase.  Energy is then integrated over finite
        order bands at the first three harmonics of the planetary rotation,
        the two mesh orders and the first-mesh sidebands.  Finite bands are
        essential for a real, speed-varying two-second record: a seeded tooth
        fault produces modulation around an order, not an ideal spectral line.

        All returned features are dimensionless fractions or ratios.  The
        function therefore supplies a fault-oriented descriptor rather than a
        trajectory-dependent absolute vibration level.
        """
        if reducer_type != 'RV':
            raise NotImplementedError("Only the audited joint-2 RV prior is currently supported")
        vib_array = np.asarray(vib_array, dtype=np.float64)
        speed = np.asarray(motor_speed_rad_s, dtype=np.float64)
        length = len(vib_array)
        if length < 16 or len(speed) < 2:
            return np.zeros(len(PHYSICS_PRIOR_NAMES), dtype=np.float32)

        vib_axes = np.nan_to_num(vib_array[:, :3], copy=False)
        if vib_axes.ndim == 1:
            vib_axes = vib_axes[:, None]
        speed_at_vib = np.interp(
            np.linspace(0.0, 1.0, length),
            np.linspace(0.0, 1.0, len(speed)), speed)
        motor_revolutions = np.cumsum(np.abs(speed_at_vib)) / (
            2.0 * np.pi * float(fs_vib))
        motor_revolutions -= motor_revolutions[0]
        travel = float(motor_revolutions[-1])
        if not np.isfinite(travel) or travel < 0.75:
            return np.zeros(len(PHYSICS_PRIOR_NAMES), dtype=np.float32)

        # Flat portions arise at standstill.  Remove duplicate angular samples
        # before interpolation; otherwise time spent stopped is spuriously
        # interpreted as a mechanical order.
        keep = np.r_[True, np.diff(motor_revolutions) > 1e-8]
        angle_samples = motor_revolutions[keep]
        vibration_samples = vib_axes[keep]
        if len(angle_samples) < 16:
            return np.zeros(len(PHYSICS_PRIOR_NAMES), dtype=np.float32)

        samples_per_revolution = 128.0
        resampled_length = int(np.clip(np.ceil(travel * samples_per_revolution), 256, length))
        angle_grid = np.linspace(angle_samples[0], angle_samples[-1], resampled_length)
        angular_vibration = np.stack([
            np.interp(angle_grid, angle_samples, vibration_samples[:, axis])
            for axis in range(vibration_samples.shape[1])
        ], axis=1)
        angular_vibration -= np.mean(angular_vibration, axis=0, keepdims=True)

        taper = np.hanning(resampled_length)[:, None]
        spectrum = np.abs(np.fft.rfft(angular_vibration * taper, axis=0)) ** 2
        spectrum = np.sum(spectrum, axis=1)
        spectrum[0] = 0.0
        total_spectral_energy = float(np.sum(spectrum))
        if not np.isfinite(total_spectral_energy) or total_spectral_energy <= 1e-18:
            return np.zeros(len(PHYSICS_PRIOR_NAMES), dtype=np.float32)
        spectrum /= total_spectral_energy
        angular_step = float(angle_grid[-1] - angle_grid[0]) / max(resampled_length - 1, 1)
        order_axis = np.fft.rfftfreq(resampled_length, d=angular_step)
        order_resolution = 1.0 / max(travel, 1e-6)
        half_band = max(0.12, 1.5 * order_resolution)

        def band_energy(center_order):
            if center_order <= 0.0 or center_order >= order_axis[-1]:
                return 0.0
            mask = np.abs(order_axis - center_order) <= half_band
            return float(np.sum(spectrum[mask])) if np.any(mask) else 0.0

        characteristic_orders = self.rv_characteristic_orders()
        ordered_names = (
            'planetary_rotation', 'first_stage_mesh', 'second_stage_mesh',
            'first_mesh_minus_planet', 'first_mesh_plus_planet')
        energies = []
        harmonic_map = {}
        for name in ordered_names:
            values = [band_energy(characteristic_orders[name] * harmonic)
                      for harmonic in (1, 2, 3)]
            energies.extend(values)
            harmonic_map[name] = values

        eps = 1e-10
        first_harmonics = harmonic_map['first_stage_mesh']
        second_harmonics = harmonic_map['second_stage_mesh']
        energies.extend([
            (first_harmonics[1] + first_harmonics[2]) / (first_harmonics[0] + eps),
            (second_harmonics[1] + second_harmonics[2]) / (second_harmonics[0] + eps),
        ])

        rms = np.sqrt(np.mean(angular_vibration ** 2, axis=0) + eps)
        standardized = np.clip(angular_vibration / rms[None, :], -20.0, 20.0)
        kurtosis = np.mean(standardized ** 4, axis=0)
        crest = np.max(np.abs(standardized), axis=0)
        # A bounded aggregate avoids a single saturated accelerometer axis
        # dominating the descriptor.
        impulsiveness = np.median(np.log1p(np.maximum(kurtosis - 3.0, 0.0) + crest))
        energies.append(float(impulsiveness))
        return np.nan_to_num(
            np.asarray(energies, dtype=np.float32), nan=0.0,
            posinf=1.0e3, neginf=0.0).clip(0.0, 1.0e3)

    def angular_resample_vibration(self, motor_speed_rad_s, vib_array, fs_vib,
                                   samples_per_revolution=128,
                                   maximum_revolutions=4.0):
        """Return a fixed-length vibration waveform in motor-angle coordinates.

        The transform is deterministic and label free.  It is the student-side
        counterpart of the MMPS order descriptor: convolutional kernels see a
        fixed number of samples per motor revolution under every trajectory.
        Short windows are repeated to the fixed length so padding length cannot
        become a trajectory identifier. Windows with more than four revolutions
        use a centered four-revolution span so late impacts are not
        systematically discarded.
        """
        vib_array = np.asarray(vib_array, dtype=np.float64)
        speed = np.asarray(motor_speed_rad_s, dtype=np.float64)
        output_length = int(round(samples_per_revolution * maximum_revolutions))
        output = np.zeros((output_length, 3), dtype=np.float32)
        if len(vib_array) < 16 or len(speed) < 2:
            return output
        vibration = np.nan_to_num(vib_array[:, :3], copy=False)
        speed_at_vibration = np.interp(
            np.linspace(0.0, 1.0, len(vibration)),
            np.linspace(0.0, 1.0, len(speed)), speed)
        revolutions = np.cumsum(np.abs(speed_at_vibration)) / (
            2.0 * np.pi * float(fs_vib))
        revolutions -= revolutions[0]
        keep = np.r_[True, np.diff(revolutions) > 1e-8]
        revolutions = revolutions[keep]
        vibration = vibration[keep]
        if len(revolutions) < 16 or revolutions[-1] < 0.25:
            return output

        observed_revolutions = min(float(revolutions[-1]), float(maximum_revolutions))
        observed_length = min(
            int(np.floor(observed_revolutions * samples_per_revolution)), output_length)
        if observed_length < 16:
            return output
        # Preserve the angular step in every case; never stretch an incomplete
        # revolution to the fixed tensor length.
        if revolutions[-1] > maximum_revolutions:
            start = 0.5 * (float(revolutions[-1]) - maximum_revolutions)
            grid = start + np.arange(observed_length) / float(samples_per_revolution)
        else:
            grid = np.arange(observed_length) / float(samples_per_revolution)
        observed = np.zeros((observed_length, 3), dtype=np.float32)
        for axis in range(min(vibration.shape[1], 3)):
            observed[:, axis] = np.interp(
                grid, revolutions, vibration[:, axis]).astype(np.float32)
        repeats = int(np.ceil(output_length / observed_length))
        output[:] = np.tile(observed, (repeats, 1))[:output_length]
        return np.nan_to_num(output, nan=0.0, posinf=1.0e4, neginf=-1.0e4)


def process_real_data_to_tensor(trial_obj, processor, target_joint_idx, reducer_type, start_time=None):
    if torch is None or F is None:
        raise ImportError("PyTorch is required for tensor conversion/training functions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if start_time is None:
        internal, imu, vib2, vib4 = trial_obj.get_random_window()
    else:
        internal, imu, vib2, vib4 = trial_obj.get_window_at(start_time)
    imu_tensor = torch.tensor(imu, dtype=torch.float32).permute(1, 2, 0)
    imu_aligned = F.interpolate(imu_tensor, size=trial_obj.len_int, mode='linear', align_corners=False).permute(2, 0, 1)
    gnn_input = torch.cat([torch.tensor(internal, dtype=torch.float32), imu_aligned], dim=-1).to(device)
    target_state = internal[:, target_joint_idx, :]
    theta_np, theta_dot_np, theta_ddot_np = processor.encoder_kinematics(
        target_state[:, 0], trial_obj.fs_int)
    theta_seq = torch.tensor(theta_np, dtype=torch.float32, device=device)
    theta_dot_seq = torch.tensor(theta_dot_np, dtype=torch.float32, device=device)
    theta_ddot_seq = torch.tensor(theta_ddot_np, dtype=torch.float32, device=device)
    tau_true_seq = torch.tensor(
        processor.torque_to_motor_side(target_state[:, 2]), dtype=torch.float32, device=device)
    vib_energy_vector = processor.extract_physics_priors(theta_dot_np, vib2 if target_joint_idx == 1 else vib4,
                                                          reducer_type, trial_obj.fs_vib)
    selected_vibration = vib2 if target_joint_idx == 1 else vib4
    vibration_raw = torch.tensor(
        selected_vibration,
        dtype=torch.float32, device=device)
    vibration_order = torch.tensor(
        processor.angular_resample_vibration(
            theta_dot_np, selected_vibration, trial_obj.fs_vib),
        dtype=torch.float32, device=device)
    return (gnn_input, theta_seq, theta_dot_seq, theta_ddot_seq, tau_true_seq,
            torch.tensor(vib_energy_vector, dtype=torch.float32, device=device),
            vibration_raw, vibration_order)


def generate_cross_domain_batch(batch_size, dataset_manager, domain_list, target_joint_idx, reducer_type, mode='train',
                                global_domains=None, windows_per_trial=1):
    """Generate a class/domain-balanced source batch.

    When ``windows_per_trial`` is greater than one, consecutive samples are
    independent random windows from the same physical trial. This supports a
    label-preserving consistency loss without exposing any target trajectory.
    """
    if torch is None:
        raise ImportError("PyTorch is required for training batch generation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn_list, th_list, thd_list, thdd_list = [], [], [], []
    tau_list, vib_list, raw_vib_list, order_vib_list = [], [], [], []
    labels_list, domain_idx_list = [], []
    processor = PhysicsDataProcessor(dataset_manager.physics_config)
    available_classes = list(dataset_manager.registry[domain_list[0]].keys())
    combos = [(c, d) for c in available_classes for d in domain_list]
    windows_per_trial = int(windows_per_trial)
    if windows_per_trial < 1 or batch_size % windows_per_trial != 0:
        raise ValueError(
            "batch_size must be divisible by positive windows_per_trial")
    trial_groups = batch_size // windows_per_trial
    if trial_groups % len(combos) != 0:
        raise ValueError(
            "trial groups must be divisible by the number of source "
            "domain/class combinations to preserve balance")

    for group_idx in range(trial_groups):
        forced_class, forced_domain = combos[group_idx % len(combos)]
        trial_obj, domain_str, class_idx = dataset_manager.sample_trial(
            [forced_domain], class_idx=forced_class, mode=mode)
        for _ in range(windows_per_trial):
            labels_list.append(class_idx)
            if global_domains is not None:
                domain_idx_list.append(global_domains.index(domain_str))
            else:
                domain_idx_list.append(domain_list.index(domain_str))
            gnn, th_seq, thd_seq, thdd_seq, tau_seq, vib, raw_vib, order_vib = process_real_data_to_tensor(
                trial_obj, processor, target_joint_idx, reducer_type)
            gnn_list.append(gnn.unsqueeze(0));
            th_list.append(th_seq.unsqueeze(0));
            thd_list.append(thd_seq.unsqueeze(0))
            thdd_list.append(thdd_seq.unsqueeze(0))
            tau_list.append(tau_seq.unsqueeze(0));
            vib_list.append(vib.unsqueeze(0))
            raw_vib_list.append(raw_vib.unsqueeze(0))
            order_vib_list.append(order_vib.unsqueeze(0))

    phys_inputs = {
        'theta': torch.cat(th_list, dim=0), 'theta_dot': torch.cat(thd_list, dim=0),
        'theta_ddot': torch.cat(thdd_list, dim=0),
        'tau_true': torch.cat(tau_list, dim=0),
        'vibration_kinematics': torch.cat(vib_list, dim=0),
        'vibration_raw': torch.cat(raw_vib_list, dim=0),
        'vibration_order': torch.cat(order_vib_list, dim=0),
    }
    base_edge = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]], dtype=torch.long,
                             device=device)
    edge_index = torch.cat([base_edge + i * 6 for i in range(batch_size)], dim=1)
    batch = torch.arange(batch_size, device=device).repeat_interleave(6)
    return phys_inputs, torch.cat(gnn_list, dim=0), edge_index, batch, torch.tensor(labels_list, dtype=torch.long,
                                                                                    device=device), torch.tensor(
        domain_idx_list, dtype=torch.long, device=device)


def generate_random_eval_batch(batch_size, dataset_manager, domain_list, target_joint_idx, reducer_type, mode='test',
                               global_domains=None):
    """Return a fixed, non-overlapping evaluation set.

    The legacy function name is retained for compatibility. If ``batch_size``
    exceeds the number of available fixed windows, every available window is
    returned exactly once rather than sampling duplicates with replacement.
    """
    if torch is None:
        raise ImportError("PyTorch is required for evaluation batch generation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn_list, th_list, thd_list, thdd_list = [], [], [], []
    tau_list, vib_list, raw_vib_list, order_vib_list = [], [], [], []
    labels_list, domain_idx_list = [], []
    processor = PhysicsDataProcessor(dataset_manager.physics_config)
    manifest = dataset_manager.build_window_manifest(domain_list, mode=mode, stride=2.0)
    if not manifest:
        raise ValueError(f"No fixed evaluation windows for domains={domain_list}, mode={mode}")
    if batch_size is None or batch_size >= len(manifest):
        selected = manifest
    else:
        # Evenly cover the stable manifest rather than taking only its first
        # domains/classes. No stochastic selection is used for evaluation.
        indices = np.linspace(0, len(manifest) - 1, num=batch_size, dtype=int)
        selected = [manifest[i] for i in indices]

    for spec in selected:
        trial_obj = spec['trial']
        domain_str = spec['domain']
        class_idx = spec['class_idx']
        labels_list.append(class_idx)
        if global_domains is not None:
            domain_idx_list.append(global_domains.index(domain_str))
        else:
            domain_idx_list.append(domain_list.index(domain_str))

        gnn, th_seq, thd_seq, thdd_seq, tau_seq, vib, raw_vib, order_vib = process_real_data_to_tensor(
            trial_obj, processor, target_joint_idx, reducer_type, start_time=spec['start_time'])
        gnn_list.append(gnn.unsqueeze(0));
        th_list.append(th_seq.unsqueeze(0));
        thd_list.append(thd_seq.unsqueeze(0))
        thdd_list.append(thdd_seq.unsqueeze(0))
        tau_list.append(tau_seq.unsqueeze(0));
        vib_list.append(vib.unsqueeze(0))
        raw_vib_list.append(raw_vib.unsqueeze(0))
        order_vib_list.append(order_vib.unsqueeze(0))

    phys_inputs = {
        'theta': torch.cat(th_list, dim=0), 'theta_dot': torch.cat(thd_list, dim=0),
        'theta_ddot': torch.cat(thdd_list, dim=0),
        'tau_true': torch.cat(tau_list, dim=0),
        'vibration_kinematics': torch.cat(vib_list, dim=0),
        'vibration_raw': torch.cat(raw_vib_list, dim=0),
        'vibration_order': torch.cat(order_vib_list, dim=0),
    }
    actual_batch_size = len(selected)
    base_edge = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]], dtype=torch.long,
                             device=device)
    edge_index = torch.cat([base_edge + i * 6 for i in range(actual_batch_size)], dim=1)
    batch = torch.arange(actual_batch_size, device=device).repeat_interleave(6)
    return phys_inputs, torch.cat(gnn_list, dim=0), edge_index, batch, torch.tensor(labels_list, dtype=torch.long,
                                                                                    device=device), torch.tensor(
        domain_idx_list, dtype=torch.long, device=device)



####################
