import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from collections import defaultdict


# ── Activity code → 0-indexed class label (12 classes) ───────────────────────
# A04 (Stepping Up) and A12 (Leftfall) removed — zero files after cleaning
# SmartFall activities kept: 1-3, 5-11, 13-14
ALL_ACTIVITY_CODES = {
    "1": 0,  "2": 1,  "3": 2,
    "5": 3,  "6": 4,  "7": 5,
    "8": 6,  "9": 7,  "10": 8,
    "11": 9, "13": 10, "14": 11,
}
FALL_ACTIVITY_CODES = {"10", "11", "13", "14"}
NUM_CLASSES = 12


def read_csv_files(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    data = {}
    for file in files:
        file_path = os.path.join(folder, file)
        try:
            # header=None required — cleaned files have no header row
            data[file] = pd.read_csv(file_path, header=None)
        except pd.errors.EmptyDataError:
            print(f"Skipped empty or unreadable file: {file}")
        except Exception as e:
            print(f"Skipped file {file} due to error: {e}")
    return data


def handle_nan_and_scale(data, scaling_method="standard"):
    """NaN-fill only — no per-window scaling."""
    if np.all(np.isnan(data), axis=0).any():
        data[:, np.all(np.isnan(data), axis=0)] = 0
    col_mean = np.nanmean(data, axis=0)
    nan_mask = np.isnan(data)
    data[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    return data


def to_one_hot(label, num_classes):
    one_hot = np.zeros(num_classes)
    one_hot[label] = 1
    return one_hot


def adjust_keypoints(skeleton_window, key_joint_indexes, joint_order):
    adjusted_skeleton = []
    for joint_index in joint_order:
        if joint_index in key_joint_indexes:
            start_idx = key_joint_indexes.index(joint_index) * 3
            adjusted_skeleton.append(skeleton_window[:, start_idx:start_idx + 3])
    return np.hstack(adjusted_skeleton)


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        skeleton_data,
        sensor1_data,
        sensor2_data,
        common_files,
        window_size,
        overlap,
        label_encoder,
        scaling="minmax",
        sensor_mean=None,
        sensor_std=None,
    ):
        self.skeleton_data = skeleton_data
        self.sensor1_data  = sensor1_data
        self.sensor2_data  = sensor2_data
        self.common_files  = list(common_files)
        self.window_size   = window_size
        self.overlap       = overlap
        self.label_encoder = label_encoder
        self.scaling       = scaling

        # 16 joints from Azure Kinect: Pelvis, SpineNavel, Neck, ShoulderL,
        # ElbowL, WristL, ShoulderR, ElbowR, WristR, HipL, KneeL, AnkleL,
        # FootL, HipR, KneeR, AnkleR  → 16×3 = 48 features
        self.key_joint_indexes = [0, 1, 3, 5, 6, 7, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24]
        self.joint_order       = [0, 1, 3, 5, 6, 7, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24]

        # Global z-score for sensors: preserves fall spike magnitude
        if sensor_mean is not None and sensor_std is not None:
            self.sensor_mean = sensor_mean
            self.sensor_std  = sensor_std
        else:
            self.sensor_mean, self.sensor_std = self._compute_global_sensor_stats()
            print(f"  Sensor global mean: {self.sensor_mean.round(3)}")
            print(f"  Sensor global std:  {self.sensor_std.round(3)}")

        (
            self.skeleton_windows,
            self.sensor1_windows,
            self.sensor2_windows,
            self.labels,
        ) = self._create_windows()

    # ── Sensor normalization ──────────────────────────────────────────────────

    def _compute_global_sensor_stats(self):
        all_s1, all_s2 = [], []
        for file in self.common_files:
            # cleaned sensor files: 3 columns only (x, y, z) — use :3
            s1 = self.sensor1_data[file].iloc[:, :3].values.astype(np.float32)
            s2 = self.sensor2_data[file].iloc[:, :3].values.astype(np.float32)
            all_s1.append(np.nan_to_num(s1, nan=0.0))
            all_s2.append(np.nan_to_num(s2, nan=0.0))
        combined = np.concatenate(
            [np.concatenate(all_s1, axis=0), np.concatenate(all_s2, axis=0)], axis=0
        )
        mean = combined.mean(axis=0).astype(np.float32)
        std  = (combined.std(axis=0) + 1e-6).astype(np.float32)
        return mean, std

    def _zscore_sensor(self, data):
        return (data - self.sensor_mean) / self.sensor_std

    # ── Skeleton normalization ────────────────────────────────────────────────

    def _normalize_to_tensor(self, skeleton_window, ref_joint_1=3, ref_joint_2=6, target_length=1.0):
        ref_bone_lengths = np.linalg.norm(
            skeleton_window[:, ref_joint_1 * 3:ref_joint_1 * 3 + 3]
            - skeleton_window[:, ref_joint_2 * 3:ref_joint_2 * 3 + 3],
            axis=1,
        )
        ref_bone_lengths = np.clip(ref_bone_lengths, 1e-2, None)
        scale_factors    = target_length / ref_bone_lengths[:, np.newaxis]
        normalized       = np.clip(skeleton_window * scale_factors, -50.0, 50.0)
        return torch.tensor(normalized, dtype=torch.float32)

    # ── Window creation ───────────────────────────────────────────────────────

    def _create_windows(self):
        skeleton_windows, sensor1_windows, sensor2_windows, labels = [], [], [], []
        step = self.window_size - self.overlap

        for file in self.common_files:
            skeleton_df = self.skeleton_data[file]
            sensor1_df  = self.sensor1_data[file]
            sensor2_df  = self.sensor2_data[file]

            # ── 14-class label from activity code ────────────────────────────
            activity_code = file.split('A')[1][:2].lstrip('0')
            if activity_code not in ALL_ACTIVITY_CODES:
                continue                     # skip unknown activity codes
            label    = ALL_ACTIVITY_CODES[activity_code]   # 0-indexed, 0..13
            is_fall  = activity_code in FALL_ACTIVITY_CODES

            num_windows = (len(skeleton_df) - self.window_size) // step + 1

            for i in range(num_windows):
                # Cleaned data: ALL windows are valid for ALL activities.
                # Fall files are already trimmed to just the fall event,
                # so every window in a fall file is a real fall window.
                # The "last window only" restriction is removed to maximise
                # fall training examples from the limited cleaned data.

                start = i * step
                end   = start + self.window_size
                if end > len(skeleton_df):
                    continue

                skeleton_window = skeleton_df.iloc[start:end, :].values
                # cleaned sensor files have 3 columns only (x, y, z) — use :3
                sensor1_window  = sensor1_df.iloc[start:end, :3].values
                sensor2_window  = sensor2_df.iloc[start:end, :3].values

                if skeleton_window.shape[1] == 97:
                    skeleton_window = skeleton_window[:, 1:]

                joint_indices = np.array(self.key_joint_indexes)
                final_indices = np.concatenate(
                    [[j * 3, j * 3 + 1, j * 3 + 2] for j in joint_indices]
                )
                skeleton_window = skeleton_window[:, final_indices]

                if (
                    skeleton_window.shape[0] != self.window_size
                    or sensor1_window.shape[0] != self.window_size
                    or sensor2_window.shape[0] != self.window_size
                ):
                    continue

                skeleton_window = adjust_keypoints(
                    skeleton_window, self.key_joint_indexes, self.joint_order
                )
                skeleton_window = handle_nan_and_scale(skeleton_window, scaling_method=self.scaling)
                skeleton_window = np.clip(skeleton_window, -10.0, 10.0)

                # Global z-score for sensors (preserves fall-magnitude spike)
                sensor1_window = handle_nan_and_scale(sensor1_window, scaling_method=self.scaling)
                sensor2_window = handle_nan_and_scale(sensor2_window, scaling_method=self.scaling)
                sensor1_window = self._zscore_sensor(sensor1_window.astype(np.float32))
                sensor2_window = self._zscore_sensor(sensor2_window.astype(np.float32))

                skeleton_window = self._normalize_to_tensor(skeleton_window)

                # Skip windows dominated by clipped/corrupted values
                if (skeleton_window.abs() >= 49.0).float().mean().item() > 0.05:
                    continue

                skeleton_windows.append(skeleton_window)
                sensor1_windows.append(sensor1_window)
                sensor2_windows.append(sensor2_window)
                labels.append(int(label))

        # ── Class distribution report ─────────────────────────────────────────
        class_indices = defaultdict(list)
        for idx, lbl in enumerate(labels):
            class_indices[lbl].append(idx)

        # Reverse the ALL_ACTIVITY_CODES mapping for correct display
        LABEL_TO_ACT = {v: k for k, v in ALL_ACTIVITY_CODES.items()}

        print("  Class distribution before oversampling:")
        for lbl in sorted(class_indices.keys()):
            act_code = LABEL_TO_ACT.get(lbl, "?")
            kind = "Fall" if act_code in FALL_ACTIVITY_CODES else "ADL"
            print(f"    Activity {act_code:>2s} ({kind}): {len(class_indices[lbl])} windows")

        # ── Oversampling: cap at 3× minority to prevent fall duplication ─────
        # Without a cap, the majority ADL class (971 windows) would force
        # fall classes (62–81 windows) to be duplicated 10×, causing overfitting.
        # Cap: target = min(majority, max(minority × 3, 200))
        class_sizes = [len(v) for v in class_indices.values()]
        minority    = min(class_sizes)
        majority    = max(class_sizes)
        target      = min(majority, max(minority * 3, 200))

        sk_os, s1_os, s2_os, lb_os = [], [], [], []

        for lbl, indices in class_indices.items():
            if len(indices) >= target:
                # downsample large ADL classes randomly
                selected = random.sample(indices, target)
            else:
                # oversample small classes up to target
                extra    = random.choices(indices, k=target - len(indices))
                selected = indices + extra
            for idx in selected:
                sk_os.append(skeleton_windows[idx])
                s1_os.append(sensor1_windows[idx])
                s2_os.append(sensor2_windows[idx])
                lb_os.append(lbl)

        total_classes = len(class_indices)
        print(f"  After balancing: {len(sk_os)} windows total "
              f"({target} per class × {total_classes} classes, "
              f"minority={minority}, cap=3×={target})")
        return sk_os, s1_os, s2_os, lb_os

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.skeleton_windows[idx],
            torch.tensor(self.sensor1_windows[idx], dtype=torch.float32),
            torch.tensor(self.sensor2_windows[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
