"""Split task_simulation.mat into equal-duration trajectory chunks.

The script splits by IMU timeline duration, then slices all related arrays
consistently for each segment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io


IMU_KEYS = {"timeIMU", "zAcc", "zGyro", "xtrue"}
GNSS_KEYS = {"timeGNSS", "zGNSS"}
COMMON_KEYS = {"S_a", "S_g", "leverarm"}


def _to_1d(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).reshape(-1)


def _slice_with_indices(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Slice along the axis that matches indices length.

    For arrays with shape (d, N), this slices the second axis.
    For arrays with shape (N, d), this slices the first axis.
    For 1D arrays with shape (N,), this slices directly.
    """

    if arr.ndim == 1:
        return arr[indices]

    if arr.shape[-1] >= np.max(indices) + 1:
        return arr[..., indices]

    if arr.shape[0] >= np.max(indices) + 1:
        return arr[indices, ...]

    raise ValueError(f"Cannot slice array with shape {arr.shape} using provided indices")


def split_task_simulation_mat(input_path: Path, out_dir: Path, parts: int) -> None:
    data = scipy.io.loadmat(str(input_path))

    for required in ["timeIMU", "timeGNSS", "xtrue", "zAcc", "zGyro", "zGNSS"]:
        if required not in data:
            raise KeyError(f"Missing required key in mat file: {required}")

    time_imu = _to_1d(data["timeIMU"])
    time_gnss = _to_1d(data["timeGNSS"])

    if len(time_imu) < parts:
        raise ValueError("Number of IMU samples is smaller than number of parts")

    t0, t1 = float(time_imu[0]), float(time_imu[-1])
    boundaries = np.linspace(t0, t1, parts + 1)

    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(parts):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]

        if i < parts - 1:
            imu_mask = (time_imu >= seg_start) & (time_imu < seg_end)
            gnss_mask = (time_gnss >= seg_start) & (time_gnss < seg_end)
        else:
            imu_mask = (time_imu >= seg_start) & (time_imu <= seg_end)
            gnss_mask = (time_gnss >= seg_start) & (time_gnss <= seg_end)

        imu_idx = np.flatnonzero(imu_mask)
        gnss_idx = np.flatnonzero(gnss_mask)

        if imu_idx.size == 0:
            raise ValueError(f"Segment {i + 1} has no IMU samples; check time data")

        segment = {}

        for key in COMMON_KEYS:
            if key in data:
                segment[key] = data[key]

        for key in IMU_KEYS:
            if key in data:
                segment[key] = _slice_with_indices(np.asarray(data[key]), imu_idx)

        for key in GNSS_KEYS:
            if key in data:
                if gnss_idx.size == 0:
                    original = np.asarray(data[key])
                    if original.ndim == 1:
                        segment[key] = original[:0]
                    elif original.shape[-1] == len(time_gnss):
                        segment[key] = original[..., :0]
                    else:
                        segment[key] = original[:0, ...]
                else:
                    segment[key] = _slice_with_indices(np.asarray(data[key]), gnss_idx)

        segment["segment_index"] = np.array([[i + 1]], dtype=np.int32)
        segment["segment_time_start"] = np.array([[seg_start]], dtype=np.float64)
        segment["segment_time_end"] = np.array([[seg_end]], dtype=np.float64)

        output_path = out_dir / f"task_simulation_part_{i + 1:02d}.mat"
        scipy.io.savemat(str(output_path), segment)

        print(
            f"Saved {output_path.name}: "
            f"IMU={imu_idx.size} samples, GNSS={gnss_idx.size} samples"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split task_simulation.mat into equal-duration segments by time"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("task_simulation.mat"),
        help="Path to input task_simulation.mat",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("task_simulation_splits"),
        help="Directory to store output .mat files",
    )
    parser.add_argument(
        "--parts",
        type=int,
        default=10,
        help="Number of equal time segments",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.parts <= 0:
        raise ValueError("--parts must be a positive integer")

    split_task_simulation_mat(args.input, args.out_dir, args.parts)


if __name__ == "__main__":
    main()
