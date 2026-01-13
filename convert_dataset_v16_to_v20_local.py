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
Local-only version of the dataset converter from v1.6 to v2.0.
This script converts LeRobot datasets stored locally without requiring Hugging Face Hub.

Usage:
    python convert_dataset_v1_to_v2_local.py \
        --input-dir /path/to/v1.6/dataset \
        --output-dir /path/to/v2.0/output \
        --single-task "Pick up the cube"

Examples:

# Single task dataset
python convert_dataset_v1_to_v2_local.py \
    --input-dir ~/Documents/pick_cuber \
    --output-dir ~/Documents/pick_cuber_v2 \
    --single-task "Pick up the cube"

# With tasks from a JSON file
python convert_dataset_v1_to_v2_local.py \
    --input-dir ~/Documents/pick_cuber \
    --output-dir ~/Documents/pick_cuber_v2 \
    --tasks-path tasks.json
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import datasets
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from safetensors.torch import load_file

from lerobot.common.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_PARQUET_PATH,
    DEFAULT_VIDEO_PATH,
    EPISODES_PATH,
    INFO_PATH,
    STATS_PATH,
    TASKS_PATH,
    flatten_dict,
    unflatten_dict,
    write_json,
    write_jsonlines,
)
from lerobot.common.datasets.video_utils import (
    VideoFrame,  # noqa: F401
    get_image_pixel_channels,
    get_video_info,
)

V16 = "v1.6"
V20 = "v2.0"

V1_INFO_PATH = "meta_data/info.json"
V1_STATS_PATH = "meta_data/stats.safetensors"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def convert_stats_to_json(v1_dir: Path, v2_dir: Path) -> None:
    safetensor_path = v1_dir / V1_STATS_PATH
    stats = load_file(safetensor_path)
    serialized_stats = {key: value.tolist() for key, value in stats.items()}
    serialized_stats = unflatten_dict(serialized_stats)

    json_path = v2_dir / STATS_PATH
    json_path.parent.mkdir(exist_ok=True, parents=True)
    with open(json_path, "w") as f:
        json.dump(serialized_stats, f, indent=4)

    # Sanity check
    with open(json_path) as f:
        stats_json = json.load(f)

    stats_json = flatten_dict(stats_json)
    stats_json = {key: torch.tensor(value) for key, value in stats_json.items()}
    for key in stats:
        torch.testing.assert_close(stats_json[key], stats[key])


def get_features_from_hf_dataset(
    dataset: Dataset, robot_config: dict | None = None
) -> dict[str, list]:
    features = {}
    for key, ft in dataset.features.items():
        if isinstance(ft, datasets.Value):
            dtype = ft.dtype
            shape = (1,)
            names = None
        if isinstance(ft, datasets.Sequence):
            assert isinstance(ft.feature, datasets.Value)
            dtype = ft.feature.dtype
            shape = (ft.length,)
            motor_names = (
                robot_config["names"][key]
                if robot_config
                else [f"motor_{i}" for i in range(ft.length)]
            )
            assert len(motor_names) == shape[0]
            names = {"motors": motor_names}
        elif isinstance(ft, datasets.Image):
            dtype = "image"
            image = dataset[0][key]  # Assuming first row
            channels = get_image_pixel_channels(image)
            shape = (image.height, image.width, channels)
            names = ["height", "width", "channel"]
        elif ft._type == "VideoFrame":
            dtype = "video"
            shape = None  # Add shape later
            names = ["height", "width", "channel"]

        features[key] = {
            "dtype": dtype,
            "shape": shape,
            "names": names,
        }

    return features


def add_task_index_by_episodes(
    dataset: Dataset, tasks_by_episodes: dict
) -> tuple[Dataset, list[str]]:
    df = dataset.to_pandas()
    tasks = list(set(tasks_by_episodes.values()))
    tasks_to_task_index = {task: task_idx for task_idx, task in enumerate(tasks)}
    episodes_to_task_index = {
        ep_idx: tasks_to_task_index[task] for ep_idx, task in tasks_by_episodes.items()
    }
    df["task_index"] = df["episode_index"].map(episodes_to_task_index).astype(int)

    features = dataset.features
    features["task_index"] = datasets.Value(dtype="int64")
    dataset = Dataset.from_pandas(df, features=features, split="train")
    return dataset, tasks


def add_task_index_from_tasks_col(
    dataset: Dataset, tasks_col: str
) -> tuple[Dataset, dict[str, list[str]], list[str]]:
    df = dataset.to_pandas()

    # HACK: This is to clean some of the instructions in our version of Open X datasets
    prefix_to_clean = "tf.Tensor(b'"
    suffix_to_clean = "', shape=(), dtype=string)"
    df[tasks_col] = (
        df[tasks_col]
        .str.removeprefix(prefix_to_clean)
        .str.removesuffix(suffix_to_clean)
    )

    # Create task_index col
    tasks_by_episode = (
        df.groupby("episode_index")[tasks_col]
        .unique()
        .apply(lambda x: x.tolist())
        .to_dict()
    )
    tasks = df[tasks_col].unique().tolist()
    tasks_to_task_index = {task: idx for idx, task in enumerate(tasks)}
    df["task_index"] = df[tasks_col].map(tasks_to_task_index).astype(int)

    # Build the dataset back from df
    features = dataset.features
    features["task_index"] = datasets.Value(dtype="int64")
    dataset = Dataset.from_pandas(df, features=features, split="train")
    dataset = dataset.remove_columns(tasks_col)

    return dataset, tasks, tasks_by_episode


def split_parquet_by_episodes(
    dataset: Dataset,
    total_episodes: int,
    total_chunks: int,
    output_dir: Path,
) -> list:
    table = dataset.data.table
    episode_lengths = []
    for ep_chunk in range(total_chunks):
        ep_chunk_start = DEFAULT_CHUNK_SIZE * ep_chunk
        ep_chunk_end = min(DEFAULT_CHUNK_SIZE * (ep_chunk + 1), total_episodes)
        chunk_dir = "/".join(DEFAULT_PARQUET_PATH.split("/")[:-1]).format(
            episode_chunk=ep_chunk
        )
        (output_dir / chunk_dir).mkdir(parents=True, exist_ok=True)
        for ep_idx in range(ep_chunk_start, ep_chunk_end):
            ep_table = table.filter(pc.equal(table["episode_index"], ep_idx))
            episode_lengths.insert(ep_idx, len(ep_table))
            output_file = output_dir / DEFAULT_PARQUET_PATH.format(
                episode_chunk=ep_chunk, episode_index=ep_idx
            )
            pq.write_table(ep_table, output_file)

    return episode_lengths


def find_video_file(input_dir: Path, video_key: str, episode_index: int) -> Path | None:
    """
    Find video file in v1.6 format. Handles different directory structures:
    - videos/{video_key}_episode_{episode_index:06d}.mp4 (flat structure)
    - videos/episode_{episode_index:06d}/{video_key}_episode_{episode_index:06d}.mp4 (nested structure)
    """
    # Try nested structure first (as in pick_cuber)
    nested_path = (
        input_dir
        / "videos"
        / f"episode_{episode_index:06d}"
        / f"{video_key}_episode_{episode_index:06d}.mp4"
    )
    if nested_path.exists():
        return nested_path

    # Try flat structure
    flat_path = input_dir / "videos" / f"{video_key}_episode_{episode_index:06d}.mp4"
    if flat_path.exists():
        return flat_path

    # Try with different naming convention
    for videos_dir in input_dir.glob("videos*"):
        for pattern in [
            f"{video_key}_episode_{episode_index:06d}.mp4",
            f"*episode_{episode_index:06d}*/{video_key}*.mp4",
        ]:
            matches = list(videos_dir.glob(pattern))
            if matches:
                return matches[0]

    return None


def move_videos_local(
    input_dir: Path,
    output_dir: Path,
    video_keys: list[str],
    total_episodes: int,
    total_chunks: int,
) -> dict:
    """
    Copy/move video files from v1.6 structure to v2.0 structure locally.
    Returns video info dict for the first episode.
    """
    videos_info = {}

    for ep_chunk in range(total_chunks):
        ep_chunk_start = DEFAULT_CHUNK_SIZE * ep_chunk
        ep_chunk_end = min(DEFAULT_CHUNK_SIZE * (ep_chunk + 1), total_episodes)

        for vid_key in video_keys:
            chunk_dir = "/".join(DEFAULT_VIDEO_PATH.split("/")[:-1]).format(
                episode_chunk=ep_chunk, video_key=vid_key
            )
            (output_dir / chunk_dir).mkdir(parents=True, exist_ok=True)

            for ep_idx in range(ep_chunk_start, ep_chunk_end):
                src_video = find_video_file(input_dir, vid_key, ep_idx)
                if src_video is None:
                    raise FileNotFoundError(
                        f"Could not find video for {vid_key} episode {ep_idx}"
                    )

                target_path = output_dir / DEFAULT_VIDEO_PATH.format(
                    episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_idx
                )

                # Copy video file
                shutil.copy2(src_video, target_path)
                print(f"Copied: {src_video.name} -> {target_path}")

                # Get video info from first episode
                if ep_idx == 0 and vid_key not in videos_info:
                    videos_info[vid_key] = get_video_info(target_path)

    return videos_info


def convert_dataset_local(
    input_dir: Path,
    output_dir: Path,
    single_task: str | None = None,
    tasks_path: Path | None = None,
    tasks_col: str | None = None,
    robot_config: dict | None = None,
):
    """
    Convert a local v1.6 dataset to v2.0 format without using Hugging Face Hub.
    """
    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting dataset from: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Load v1.6 metadata
    metadata_v1 = load_json(input_dir / V1_INFO_PATH)
    print(f"V1.6 metadata: {metadata_v1}")

    # Load dataset from arrow/parquet files
    # Try different file patterns
    arrow_files = list((input_dir / "train").glob("*.arrow"))
    parquet_files = list((input_dir / "data").glob("**/*.parquet")) if (input_dir / "data").exists() else []
    
    if arrow_files:
        # Load from arrow file directly (avoid loading json files in same dir)
        dataset = datasets.Dataset.from_file(str(arrow_files[0]))
    elif parquet_files:
        # Load from parquet files
        dataset = datasets.load_dataset("parquet", data_dir=str(input_dir / "data"), split="train")
    else:
        raise FileNotFoundError(f"No arrow or parquet data files found in {input_dir}")
    print(f"Loaded dataset with {len(dataset)} frames")
    print(f"Features: {list(dataset.features.keys())}")

    features = get_features_from_hf_dataset(dataset, robot_config)
    video_keys = [key for key, ft in features.items() if ft["dtype"] == "video"]
    print(f"Video keys: {video_keys}")

    if single_task and "language_instruction" in dataset.column_names:
        print(
            "Warning: 'single_task' provided but 'language_instruction' column found. Using 'language_instruction'."
        )
        single_task = None
        tasks_col = "language_instruction"

    # Episodes & chunks
    episode_indices = sorted(dataset.unique("episode_index"))
    total_episodes = len(episode_indices)
    assert episode_indices == list(range(total_episodes))
    total_videos = total_episodes * len(video_keys)
    total_chunks = total_episodes // DEFAULT_CHUNK_SIZE
    if total_episodes % DEFAULT_CHUNK_SIZE != 0:
        total_chunks += 1

    print(f"Total episodes: {total_episodes}")
    print(f"Total chunks: {total_chunks}")

    # Tasks
    if single_task:
        tasks_by_episodes = {ep_idx: single_task for ep_idx in episode_indices}
        dataset, tasks = add_task_index_by_episodes(dataset, tasks_by_episodes)
        tasks_by_episodes = {
            ep_idx: [task] for ep_idx, task in tasks_by_episodes.items()
        }
    elif tasks_path:
        tasks_by_episodes = load_json(tasks_path)
        tasks_by_episodes = {
            int(ep_idx): task for ep_idx, task in tasks_by_episodes.items()
        }
        dataset, tasks = add_task_index_by_episodes(dataset, tasks_by_episodes)
        tasks_by_episodes = {
            ep_idx: [task] for ep_idx, task in tasks_by_episodes.items()
        }
    elif tasks_col:
        dataset, tasks, tasks_by_episodes = add_task_index_from_tasks_col(
            dataset, tasks_col
        )
    else:
        raise ValueError("Must provide one of: --single-task, --tasks-path, or --tasks-col")

    assert set(tasks) == {
        task for ep_tasks in tasks_by_episodes.values() for task in ep_tasks
    }
    tasks = [
        {"task_index": task_idx, "task": task} for task_idx, task in enumerate(tasks)
    ]
    write_jsonlines(tasks, output_dir / TASKS_PATH)
    features["task_index"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }

    # Videos
    if video_keys:
        assert metadata_v1.get("video", False)
        dataset = dataset.remove_columns(video_keys)

        print("Moving video files...")
        videos_info = move_videos_local(
            input_dir, output_dir, video_keys, total_episodes, total_chunks
        )

        for key in video_keys:
            features[key]["shape"] = (
                videos_info[key].pop("video.height"),
                videos_info[key].pop("video.width"),
                videos_info[key].pop("video.channels"),
            )
            features[key]["video_info"] = videos_info[key]
            assert math.isclose(
                videos_info[key]["video.fps"], metadata_v1["fps"], rel_tol=1e-3
            )
            if "encoding" in metadata_v1:
                assert (
                    videos_info[key]["video.pix_fmt"]
                    == metadata_v1["encoding"]["pix_fmt"]
                )
    else:
        assert metadata_v1.get("video", 0) == 0
        videos_info = None

    # Split data into 1 parquet file by episode
    print("Splitting parquet by episodes...")
    episode_lengths = split_parquet_by_episodes(
        dataset, total_episodes, total_chunks, output_dir
    )

    if robot_config is not None:
        robot_type = robot_config["robot_type"]
    else:
        robot_type = "unknown"

    # Episodes
    episodes = [
        {
            "episode_index": ep_idx,
            "tasks": tasks_by_episodes[ep_idx],
            "length": episode_lengths[ep_idx],
        }
        for ep_idx in episode_indices
    ]
    write_jsonlines(episodes, output_dir / EPISODES_PATH)

    # Assemble metadata v2.0
    metadata_v2_0 = {
        "codebase_version": V20,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": len(dataset),
        "total_tasks": len(tasks),
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": metadata_v1["fps"],
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": DEFAULT_PARQUET_PATH,
        "video_path": DEFAULT_VIDEO_PATH if video_keys else None,
        "features": features,
    }
    write_json(metadata_v2_0, output_dir / INFO_PATH)

    # Convert stats
    print("Converting stats...")
    convert_stats_to_json(input_dir, output_dir)

    print(f"\nConversion complete!")
    print(f"Output saved to: {output_dir}")
    print(f"V2.0 metadata: {output_dir / INFO_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot dataset from v1.6 to v2.0 locally (no Hugging Face Hub required)"
    )
    task_args = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to the v1.6 dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to save the converted v2.0 dataset",
    )
    task_args.add_argument(
        "--single-task",
        type=str,
        help="A short but accurate description of the single task performed in the dataset.",
    )
    task_args.add_argument(
        "--tasks-col",
        type=str,
        help="The name of the column containing language instructions",
    )
    task_args.add_argument(
        "--tasks-path",
        type=Path,
        help="The path to a .json file containing one language instruction for each episode_index",
    )

    args = parser.parse_args()

    convert_dataset_local(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        single_task=args.single_task,
        tasks_path=args.tasks_path,
        tasks_col=args.tasks_col,
    )


if __name__ == "__main__":
    main()

