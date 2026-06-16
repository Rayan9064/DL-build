from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from datasets import Features, Value, load_dataset


DATASET_ID = "builddotai/Egocentric-10K"

EGOCENTRIC_FEATURES = Features(
    {
        "mp4": Value("binary"),
        "json": {
            "factory_id": Value("string"),
            "worker_id": Value("string"),
            "video_index": Value("int64"),
            "duration_sec": Value("float64"),
            "width": Value("int64"),
            "height": Value("int64"),
            "fps": Value("float64"),
            "size_bytes": Value("int64"),
            "codec": Value("string"),
        },
        "__key__": Value("string"),
        "__url__": Value("string"),
    }
)


def _safe_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)
    return cleaned.strip("._") or "clip"


def _read_video_bytes(mp4_value: Any) -> bytes:
    if isinstance(mp4_value, bytes):
        return mp4_value

    if isinstance(mp4_value, bytearray):
        return bytes(mp4_value)

    if isinstance(mp4_value, dict):
        if isinstance(mp4_value.get("bytes"), bytes):
            return mp4_value["bytes"]
        if mp4_value.get("path"):
            with open(mp4_value["path"], "rb") as handle:
                return handle.read()

    if hasattr(mp4_value, "read"):
        data = mp4_value.read()
        return data if isinstance(data, bytes) else bytes(data)

    raise TypeError(f"Unsupported mp4 payload type: {type(mp4_value)!r}")


def _copy_or_write_video(mp4_value: Any, destination: Path) -> None:
    if isinstance(mp4_value, dict) and mp4_value.get("path"):
        shutil.copyfile(mp4_value["path"], destination)
        return

    destination.write_bytes(_read_video_bytes(mp4_value))


def _label_studio_video_url(project_root: Path, video_path: Path) -> str:
    relative = video_path.resolve().relative_to(project_root.resolve()).as_posix()
    return f"/data/local-files/?d={relative}"


def export_samples(
    output_dir: Path,
    count: int,
    split: str,
    seed: int,
    shuffle_buffer: int,
    overwrite: bool,
) -> Path:
    project_root = Path.cwd()
    output_dir = output_dir.resolve()

    if output_dir.exists() and overwrite:
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATASET_ID} split={split!r} in streaming mode...")
    dataset = load_dataset(
        DATASET_ID,
        split=split,
        streaming=True,
        features=EGOCENTRIC_FEATURES,
    )

    if shuffle_buffer > 0:
        print(f"Shuffling with buffer_size={shuffle_buffer}; this may read many clips before writing the first file.")
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    else:
        print("Shuffle disabled; exporting the first streamed clips immediately.")
        random.seed(seed)

    tasks: list[dict[str, Any]] = []

    for index, sample in enumerate(dataset.take(count), start=1):
        metadata = dict(sample.get("json") or {})
        key = _safe_filename(str(sample.get("__key__") or f"sample_{index:04d}"))
        video_path = output_dir / f"{index:04d}_{key}.mp4"
        metadata_path = output_dir / f"{index:04d}_{key}.json"

        print(f"Writing clip {index}/{count}: {video_path.name}")
        _copy_or_write_video(sample["mp4"], video_path)

        metadata.update(
            {
                "__key__": sample.get("__key__"),
                "__url__": sample.get("__url__"),
                "local_video": str(video_path),
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        tasks.append(
            {
                "data": {"video": _label_studio_video_url(project_root, video_path)},
                "meta": metadata,
            }
        )

        print(f"Exported {video_path}")

    tasks_path = output_dir / "label_studio_tasks.json"
    tasks_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    print(f"Wrote {len(tasks)} Label Studio tasks to {tasks_path}")
    return tasks_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream and export a small Egocentric-10K sample.")
    parser.add_argument("--output-dir", type=Path, default=Path("egocentric_samples"))
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1")

    export_samples(
        output_dir=args.output_dir,
        count=args.count,
        split=args.split,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
