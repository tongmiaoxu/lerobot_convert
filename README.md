# LeRobot Dataset Conversion: v1.6 → v3.0 (Local)

Convert a LeRobot dataset from **v1.6** to **v3.0** format locally, without Hugging Face Hub.

## Overview

```
v1.6 → v2.0 → v2.1 → v3.0
```

| Script | What it does |
|--------|--------------|
| `convert_dataset_v16_to_v20_local.py` | Reorganizes data, splits parquet by episode |
| `convert_dataset_v20_to_v21_local.py` | Adds per-episode statistics |
| `convert_dataset_v21_to_v30_local.py` | Concatenates files, adds quantile stats for pi0.5 |

---

## Installation

```bash
# 1. Clone this repository (converter scripts)
git clone <your-github-repo-url>
cd lerobot_convert

# 2. Create conda environment
conda create -n lerobot_convert python=3.10 -y
conda activate lerobot_convert

# 3. Install dependencies
pip install numpy pandas pyarrow torch torchvision safetensors jsonlines tqdm pillow datasets opencv-python imageio av imageio-ffmpeg

# 4. Install lerobot (required for v1.6 → v2.0 conversion only)
# Clone lerobot in a separate directory (can be anywhere)
cd ~  # or wherever you want
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout c574eb49845d48f5aad532d823ef56aec1c0d0f2
pip install -e .
cd ~/lerobot_convert  # back to converter repo
```

> **Note:** 
> - The specific lerobot commit is required because newer versions have breaking API changes
> - Steps 2-3 (v2.0→v2.1→v3.0) don't require lerobot - only the first script needs it
> - You can clone lerobot anywhere - once installed with `pip install -e .`, Python will find it

---

## Quick Start (3 Commands)

```bash
# Make sure you're in the converter repo directory
cd ~/lerobot_convert  # or wherever you cloned it
conda activate lerobot_convert

# Step 1: v1.6 → v2.0
python convert_dataset_v16_to_v20_local.py \
    --input-dir ~/Documents/pick_cuber \
    --output-dir ~/Documents/pick_cuber_v2 \
    --single-task "Pick up the cube"

# Step 2: v2.0 → v2.1
python convert_dataset_v20_to_v21_local.py \
    --input-dir ~/Documents/pick_cuber_v2 \
    --output-dir ~/Documents/pick_cuber_v21

# Step 3: v2.1 → v3.0
python convert_dataset_v21_to_v30_local.py \
    --input-dir ~/Documents/pick_cuber_v21 \
    --output-dir ~/Documents/pick_cuber_v30
```

**That's it!** Your v3.0 dataset is ready for pi0.5 training.

---

## Directory Structures

### v1.6
```
pick_cuber/
├── train/data-00000-of-00001.arrow
├── meta_data/
│   ├── info.json
│   └── stats.safetensors
├── episodes/episode_0.pth, ...
└── videos/episode_000000/
    └── observation.images.cam_high_episode_000000.mp4
```

### v2.0
```
pick_cuber_v2/
├── data/chunk-000/
│   ├── episode_000000.parquet
│   ├── episode_000001.parquet
│   └── ...
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── episodes.jsonl
│   └── tasks.jsonl
└── videos/chunk-000/observation.images.cam_high/
    ├── episode_000000.mp4
    ├── episode_000001.mp4
    └── ...
```

### v2.1
```
pick_cuber_v21/
├── data/chunk-000/
│   ├── episode_000000.parquet
│   └── ...
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl    # NEW: per-episode stats
│   └── tasks.jsonl
└── videos/chunk-000/observation.images.cam_high/
    ├── episode_000000.mp4
    └── ...
```

### v3.0
```
pick_cuber_v30/
├── data/chunk-000/
│   └── file-000.parquet        # Concatenated episodes
├── meta/
│   ├── info.json
│   ├── stats.json              # Includes q01/q99 for pi0.5
│   ├── tasks.parquet           # Task string as index
│   └── episodes/chunk-000/
│       └── file-000.parquet
└── videos/observation.images.cam_high/chunk-000/
    └── file-000.mp4            # Concatenated videos
```

---

## Troubleshooting

### AV1 Video Warnings
```
Your platform doesn't support hardware accelerated AV1 decoding
```
**This is normal.** The script automatically uses software decoding.

### Arrow File Errors
```
Not an Arrow file
```
**Fixed in the script.** It loads `.arrow` files directly.

---

## Files

| File | Description |
|------|-------------|
| `convert_dataset_v16_to_v20_local.py` | v1.6 → v2.0 |
| `convert_dataset_v20_to_v21_local.py` | v2.0 → v2.1 |
| `convert_dataset_v21_to_v30_local.py` | v2.1 → v3.0 |
