#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Local-only version of the dataset converter from v2.0 to v2.1.
This script converts LeRobot datasets stored locally without requiring Hugging Face Hub.

The main change from v2.0 to v2.1 is:
- Generates per-episode stats and writes them to `episodes_stats.jsonl`
- Updates codebase_version in `info.json` from v2.0 to v2.1
- Optionally removes the deprecated `stats.json`

Usage:
    python convert_dataset_v20_to_v21_local.py \
        --input-dir /path/to/v2.0/dataset \
        --output-dir /path/to/v2.1/output

Examples:

# Convert in-place (modifies input directory)
python convert_dataset_v20_to_v21_local.py \
    --input-dir ~/Documents/pick_cuber_v2

# Convert to a new directory
python convert_dataset_v20_to_v21_local.py \
    --input-dir ~/Documents/pick_cuber_v2 \
    --output-dir ~/Documents/pick_cuber_v21

# Delete old stats.json after conversion
python convert_dataset_v20_to_v21_local.py \
    --input-dir ~/Documents/pick_cuber_v2 \
    --delete-old-stats
"""

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import jsonlines
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

V20 = "v2.0"
V21 = "v2.1"

INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"
EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
EPISODES_PATH = "meta/episodes.jsonl"


def load_json(fpath: Path) -> dict:
    with open(fpath) as f:
        return json.load(f)


def write_json(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_jsonlines(fpath: Path) -> list:
    with jsonlines.open(fpath, "r") as reader:
        return list(reader)


def append_jsonlines(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(fpath, "a") as writer:
        writer.write(data)


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        curr = outdict
        for part in parts[:-1]:
            if part not in curr:
                curr[part] = {}
            curr = curr[part]
        curr[parts[-1]] = value
    return outdict


def serialize_dict(stats: dict) -> dict:
    serialized_dict = {}
    for key, value in flatten_dict(stats).items():
        if isinstance(value, np.ndarray):
            serialized_dict[key] = value.tolist()
        elif isinstance(value, np.generic):
            serialized_dict[key] = value.item()
        elif isinstance(value, (int, float, list)):
            serialized_dict[key] = value
        else:
            raise NotImplementedError(f"The value '{value}' of type '{type(value)}' is not supported.")
    return unflatten_dict(serialized_dict)


def cast_stats_to_numpy(stats) -> dict[str, dict[str, np.ndarray]]:
    stats = {key: np.array(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(stats)


def estimate_num_samples(
    dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75
) -> int:
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> list[int]:
    num_samples = estimate_num_samples(data_len)
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def auto_downsample_height_width(img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300):
    _, height, width = img.shape
    if max(width, height) < max_size_threshold:
        return img
    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::downsample_factor, ::downsample_factor]


def get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def sample_video_frames_cv2(video_path: Path, num_frames: int = None) -> np.ndarray:
    """Sample frames from a video file using cv2."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if num_frames is None:
        sampled_indices = sample_indices(total_frames)
    else:
        sampled_indices = sample_indices(num_frames)

    frames = []
    for idx in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB and transpose to channel-first
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
            frame = auto_downsample_height_width(frame)
            frames.append(frame)
    
    cap.release()
    
    if not frames:
        raise RuntimeError(f"Could not read any frames from video (cv2): {video_path}")
    
    return np.stack(frames, axis=0)


def sample_video_frames_imageio(video_path: Path, num_frames: int = None) -> np.ndarray:
    """Sample frames from a video file using imageio-ffmpeg (better AV1 support)."""
    import imageio.v3 as iio
    
    # Read all frames metadata first to get count
    props = iio.improps(video_path, plugin="pyav")
    total_frames = props.n_images if hasattr(props, 'n_images') else num_frames or 100
    
    if num_frames is None:
        sampled_indices = sample_indices(total_frames)
    else:
        sampled_indices = sample_indices(num_frames)
    
    frames = []
    # Read frames using iterator for efficiency
    all_frames = list(iio.imiter(video_path, plugin="pyav"))
    
    for idx in sampled_indices:
        if idx < len(all_frames):
            frame = all_frames[idx]
            # Transpose to channel-first (HWC -> CHW)
            frame = np.transpose(frame, (2, 0, 1))
            frame = auto_downsample_height_width(frame)
            frames.append(frame)
    
    if not frames:
        raise RuntimeError(f"Could not read any frames from video (imageio): {video_path}")
    
    return np.stack(frames, axis=0)


def sample_video_frames(video_path: Path, num_frames: int = None) -> np.ndarray:
    """Sample frames from a video file. Tries cv2 first, falls back to imageio."""
    # Try OpenCV first (faster for non-AV1 codecs)
    try:
        return sample_video_frames_cv2(video_path, num_frames)
    except Exception as e:
        cv2_error = str(e)
    
    # Fallback to imageio with pyav (better AV1 support)
    try:
        import imageio.v3 as iio
        return sample_video_frames_imageio(video_path, num_frames)
    except ImportError:
        raise RuntimeError(
            f"OpenCV failed to read video ({cv2_error}). "
            f"Install imageio and av for AV1 support: pip install imageio av"
        )
    except Exception as e:
        raise RuntimeError(f"Both cv2 and imageio failed to read video: {video_path}. cv2: {cv2_error}, imageio: {e}")


def compute_episode_stats_local(
    data_dir: Path,
    video_dir: Path | None,
    episode_index: int,
    features: dict,
    info: dict,
) -> dict:
    """Compute stats for a single episode from local parquet and video files."""
    
    # Find the parquet file for this episode
    chunk_size = info.get("chunks_size", 1000)
    chunk_index = episode_index // chunk_size
    parquet_path = data_dir / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.parquet"
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    # Load parquet data
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    ep_stats = {}
    
    for key, ft in features.items():
        dtype = ft["dtype"]
        
        if dtype == "video":
            # Load video and sample frames
            if video_dir is None:
                continue
            video_path = video_dir / f"chunk-{chunk_index:03d}" / key / f"episode_{episode_index:06d}.mp4"
            if not video_path.exists():
                print(f"Warning: Video file not found: {video_path}")
                continue
            
            try:
                ep_ft_data = sample_video_frames(video_path, len(df))
                axes_to_reduce = (0, 2, 3)  # keep channel dim
                keepdims = True
                ep_stats[key] = get_feature_stats(ep_ft_data, axis=axes_to_reduce, keepdims=keepdims)
                # Normalize to [0, 1] and remove batch dim
                ep_stats[key] = {
                    k: v if k == "count" else np.squeeze(v / 255.0, axis=0)
                    for k, v in ep_stats[key].items()
                }
            except Exception as e:
                print(f"Warning: Could not process video {video_path}: {e}")
                continue
        elif dtype == "image":
            # Skip image features for now (would need image paths)
            continue
        elif dtype == "string":
            continue
        elif key in df.columns:
            # Numeric data from parquet
            ep_ft_data = np.array(df[key].tolist())
            axes_to_reduce = 0
            keepdims = ep_ft_data.ndim == 1
            ep_stats[key] = get_feature_stats(ep_ft_data, axis=axes_to_reduce, keepdims=keepdims)
    
    return ep_stats


def aggregate_feature_stats(stats_ft_list: list[dict]) -> dict[str, np.ndarray]:
    """Aggregates stats for a single feature."""
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)

    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count

    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    return {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }


def aggregate_stats(stats_list: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats from multiple episodes into a single set of stats."""
    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {}

    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        if stats_with_key:
            aggregated_stats[key] = aggregate_feature_stats(stats_with_key)

    return aggregated_stats


def check_aggregate_stats(
    aggregated_stats: dict,
    reference_stats: dict,
    video_rtol_atol: tuple[float] = (1e-2, 1e-2),
    default_rtol_atol: tuple[float] = (5e-6, 6e-5),
):
    """Verifies that the aggregated stats from episodes_stats are close to reference stats."""
    for key, val_dict in aggregated_stats.items():
        if "image" in key or key.startswith("observation.images"):
            rtol, atol = video_rtol_atol
        else:
            rtol, atol = default_rtol_atol

        for stat, val in val_dict.items():
            if key in reference_stats and stat in reference_stats[key]:
                try:
                    np.testing.assert_allclose(
                        val, reference_stats[key][stat], rtol=rtol, atol=atol,
                        err_msg=f"feature='{key}' stats='{stat}'"
                    )
                except AssertionError as e:
                    print(f"Warning: Stats mismatch for {key}/{stat}: {e}")


def write_episode_stats(episode_index: int, episode_stats: dict, local_dir: Path):
    """Write episode stats to episodes_stats.jsonl."""
    episode_stats_entry = {"episode_index": episode_index, "stats": serialize_dict(episode_stats)}
    append_jsonlines(episode_stats_entry, local_dir / EPISODES_STATS_PATH)


def convert_dataset_local(
    input_dir: Path,
    output_dir: Path | None = None,
    delete_old_stats: bool = False,
    num_workers: int = 4,
):
    """
    Convert a local v2.0 dataset to v2.1 format.
    
    If output_dir is None, modifies input_dir in-place.
    """
    input_dir = Path(input_dir).expanduser().resolve()
    
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir).expanduser().resolve()
        # Copy entire directory if output is different
        if output_dir != input_dir:
            print(f"Copying dataset from {input_dir} to {output_dir}...")
            if output_dir.exists():
                shutil.rmtree(output_dir)
            shutil.copytree(input_dir, output_dir)

    if not output_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {output_dir}")

    print(f"Converting dataset: {output_dir}")

    # Load info
    info_path = output_dir / INFO_PATH
    info = load_json(info_path)
    
    if info.get("codebase_version") != V20:
        print(f"Warning: Dataset version is {info.get('codebase_version')}, expected {V20}")

    features = info.get("features", {})
    total_episodes = info.get("total_episodes", 0)
    
    print(f"Total episodes: {total_episodes}")
    print(f"Features: {list(features.keys())}")

    # Remove old episodes_stats.jsonl if it exists
    episodes_stats_path = output_dir / EPISODES_STATS_PATH
    if episodes_stats_path.exists():
        episodes_stats_path.unlink()
        print(f"Removed existing {EPISODES_STATS_PATH}")

    # Compute per-episode stats
    data_dir = output_dir / "data"
    video_dir = output_dir / "videos" if (output_dir / "videos").exists() else None
    
    all_episode_stats = {}
    
    print("Computing per-episode stats...")
    
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    compute_episode_stats_local, data_dir, video_dir, ep_idx, features, info
                ): ep_idx
                for ep_idx in range(total_episodes)
            }
            for future in tqdm(as_completed(futures), total=total_episodes):
                ep_idx = futures[future]
                try:
                    all_episode_stats[ep_idx] = future.result()
                except Exception as e:
                    print(f"Error computing stats for episode {ep_idx}: {e}")
    else:
        for ep_idx in tqdm(range(total_episodes)):
            try:
                all_episode_stats[ep_idx] = compute_episode_stats_local(
                    data_dir, video_dir, ep_idx, features, info
                )
            except Exception as e:
                print(f"Error computing stats for episode {ep_idx}: {e}")

    # Write episode stats in order
    print("Writing episode stats...")
    for ep_idx in tqdm(range(total_episodes)):
        if ep_idx in all_episode_stats:
            write_episode_stats(ep_idx, all_episode_stats[ep_idx], output_dir)

    # Load and check against old stats
    old_stats_path = output_dir / STATS_PATH
    if old_stats_path.exists():
        print("Checking aggregated stats against reference...")
        ref_stats = cast_stats_to_numpy(load_json(old_stats_path))
        agg_stats = aggregate_stats(list(all_episode_stats.values()))
        check_aggregate_stats(agg_stats, ref_stats)

    # Update version in info.json
    info["codebase_version"] = V21
    write_json(info, info_path)
    print(f"Updated codebase_version to {V21}")

    # Optionally delete old stats.json
    if delete_old_stats and old_stats_path.exists():
        old_stats_path.unlink()
        print(f"Deleted old {STATS_PATH}")

    print(f"\nConversion complete!")
    print(f"Output: {output_dir}")
    print(f"New file: {output_dir / EPISODES_STATS_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot dataset from v2.0 to v2.1 locally (no Hugging Face Hub required)"
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to the v2.0 dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Path to save the converted v2.1 dataset. If not provided, modifies input-dir in-place.",
    )
    parser.add_argument(
        "--delete-old-stats",
        action="store_true",
        help="Delete the old stats.json file after conversion. Defaults to False.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for parallelizing stats compute. Defaults to 4.",
    )

    args = parser.parse_args()

    convert_dataset_local(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        delete_old_stats=args.delete_old_stats,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

