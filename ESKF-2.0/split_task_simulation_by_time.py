"""Split task_simulation.mat into trajectory chunks.

Supports:
1) equal-time split (legacy behavior)
2) random fixed-length IMU windows (for random trajectory extraction)
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


def split_task_simulation_random_windows(
    input_path: Path,
    out_dir: Path,
    parts: int,
    imu_points: int,
    seed: int,
    allow_overlap: bool,
) -> None:
    data = scipy.io.loadmat(str(input_path))

    for required in ["timeIMU", "timeGNSS", "xtrue", "zAcc", "zGyro", "zGNSS"]:
        if required not in data:
            raise KeyError(f"Missing required key in mat file: {required}")

    time_imu = _to_1d(data["timeIMU"])
    time_gnss = _to_1d(data["timeGNSS"])
    imu_total = len(time_imu)

    if imu_points <= 0:
        raise ValueError("--imu-points must be a positive integer")
    if imu_total < imu_points:
        raise ValueError(
            f"IMU sample count ({imu_total}) is smaller than requested window size ({imu_points})"
        )

    max_start = imu_total - imu_points
    rng = np.random.default_rng(seed)

    if not allow_overlap and parts * imu_points > imu_total:
        raise ValueError(
            "Requested non-overlap random windows exceed total IMU samples; "
            "set --allow-overlap to enable sampling with overlap"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    used_ranges = []
    starts = []
    if allow_overlap:
        starts = rng.integers(0, max_start + 1, size=parts)
    else:
        # Greedy random non-overlapping windows.
        candidates = list(rng.permutation(max_start + 1))
        for s in candidates:
            e = s + imu_points
            overlap = any(not (e <= us or s >= ue) for us, ue in used_ranges)
            if not overlap:
                starts.append(s)
                used_ranges.append((s, e))
            if len(starts) == parts:
                break
        if len(starts) < parts:
            raise ValueError("Unable to find enough non-overlapping random windows")

    for i, start in enumerate(starts, start=1):
        imu_start = int(start)
        imu_end = imu_start + imu_points
        imu_idx = np.arange(imu_start, imu_end, dtype=np.int64)

        seg_start_t = float(time_imu[imu_start])
        seg_end_t = float(time_imu[imu_end - 1])
        gnss_mask = (time_gnss >= seg_start_t) & (time_gnss <= seg_end_t)
        gnss_idx = np.flatnonzero(gnss_mask)

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

        segment["segment_index"] = np.array([[i]], dtype=np.int32)
        segment["segment_imu_start_index"] = np.array([[imu_start]], dtype=np.int32)
        segment["segment_imu_end_index"] = np.array([[imu_end - 1]], dtype=np.int32)
        segment["segment_time_start"] = np.array([[seg_start_t]], dtype=np.float64)
        segment["segment_time_end"] = np.array([[seg_end_t]], dtype=np.float64)

        output_path = out_dir / f"task_simulation_random_{i:02d}.mat"
        scipy.io.savemat(str(output_path), segment)
        print(
            f"Saved {output_path.name}: "
            f"IMU={imu_idx.size} samples, GNSS={gnss_idx.size} samples, "
            f"start={imu_start}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split task_simulation.mat into equal-time or random fixed-length segments"
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
        help="Number of output segments",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["time", "random"],
        default="time",
        help="Split mode: 'time' for equal-duration split, 'random' for random fixed-length windows",
    )
    parser.add_argument(
        "--imu-points",
        type=int,
        default=50000,
        help="IMU points per segment in random mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible random mode",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow overlap between random windows (recommended when parts*imu_points exceeds total IMU points)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.parts <= 0:
        raise ValueError("--parts must be a positive integer")
    if args.mode == "time":
        split_task_simulation_mat(args.input, args.out_dir, args.parts)
    else:
        split_task_simulation_random_windows(
            input_path=args.input,
            out_dir=args.out_dir,
            parts=args.parts,
            imu_points=args.imu_points,
            seed=args.seed,
            allow_overlap=args.allow_overlap,
        )


if __name__ == "__main__":
    main()
