# Dataset Preparation

## Training Dataset

This project trains on a mixture of public video/image editing datasets:

- Video editing: [Ditto-1M](https://huggingface.co/datasets/QingyanBai/Ditto-1M/tree/main/videos/source), [OpenVE-3M](https://huggingface.co/datasets/Lewandofski/OpenVE-3M), [ReCo](https://huggingface.co/datasets/HiDream-ai/ReCo-Data/tree/main)
- Image editing: [GPT-Image-Edit-1.5M](https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M), [NHR-Edit](https://huggingface.co/datasets/iitolstykh/NHR-Edit)

Released Kiwi-Edit training metadata:

- Image metadata: [image_edit_metadata](https://huggingface.co/datasets/linyq/kiwi_edit_training_data/tree/main/image_edit_metadata)
- Video metadata: [video_edit_metadata](https://huggingface.co/datasets/linyq/kiwi_edit_training_data/tree/main/video_edit_metadata)
- Reference-video parquet: [refvie_477k](https://huggingface.co/datasets/linyq/kiwi_edit_training_data/tree/main/refvie_477k)

For runnable examples, see files under [`demo_data/`](demo_data/).

## Evaluation Benchmark

Here are the download links for the evaluation benchmark datasets:

- **OpenVE-Bench**: https://huggingface.co/datasets/Lewandofski/OpenVE-Bench
- **RefVIE-Bench**: https://huggingface.co/datasets/linyq/RefVIE-Bench

After downloading, the expected directory structure under `./benchmark/` is:

```
benchmark/
├── OpenVE-Bench/
│   ├── videos/
│   ├── benchmark_videos.csv
│   └── README.md
└── RefVIE-Bench/
    ├── ref_images/
    ├── source_videos/
    └── refvie_bench.yaml
```


## CSV Format by Training Stage

All metadata files are CSV. Paths should be relative to `--dataset_base_path` (or absolute paths).

| Stage | Required columns |
| - | - |
| Stage 1 (Image) | `src_video`, `tgt_video`, `prompt` |
| Stage 2 (Image + Video) | `src_video`, `tgt_video`, `prompt` |
| Stage 3 (Ref-Video) | `src_video`, `tgt_video`, `ref_image`, `prompt` |

Example rows:

```csv
src_video,tgt_video,prompt
video/source/xxx.mp4,video/target/xxx.mp4,Remove the monkey.
```

```csv
src_video,tgt_video,ref_image,prompt
video/source/yyy.mp4,video/target/yyy.mp4,ref_images/yyy.jpg,Make it in anime style.
```

## How Metadata Maps to Training Scripts

- `--img_dataset_metadata_path`: stage-1 image-style supervision (and reused in mixed stages)
- `--vid_dataset_metadata_path`: stage-2 video supervision
- `--vid_ref_dataset_metadata_path`: stage-3 reference-video supervision

## Build Reference CSV from Parquet

Use this script to export `ref_image_bytes` and generate a stage-3 CSV:

```python
import glob
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

INPUT_PATTERN = "path_to_parquet/chunk_*.parquet"
IMAGE_DIR = Path("path_to_ref_image")
CSV_OUTPUT = "refvie_dataset.csv"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

chunk_files = sorted(glob.glob(INPUT_PATTERN))
print(f"Found {len(chunk_files)} chunks.")

all_rows = []
for file_path in chunk_files:
    print(f"Processing: {os.path.basename(file_path)}")
    df = pd.read_parquet(
        file_path,
        columns=["iid", "prompt", "src_video", "tgt_video", "ref_image_bytes"],
    )

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Saving ref images"):
        iid = str(row["iid"])
        image_path = IMAGE_DIR / f"{iid}.jpg"

        if row["ref_image_bytes"] is not None and not image_path.exists():
            with open(image_path, "wb") as f:
                f.write(row["ref_image_bytes"])

        all_rows.append(
            {
                "src_video": row["src_video"],
                "tgt_video": row["tgt_video"],
                "ref_image": str(image_path),
                "prompt": row["prompt"],
            }
        )

final_df = pd.DataFrame(all_rows)
final_df.to_csv(CSV_OUTPUT, index=False)
print(f"Saved {len(final_df)} rows to {CSV_OUTPUT}")
```
