from __future__ import annotations

import argparse
import json
from pathlib import Path

import decord
import imageio_ffmpeg


def transcode_video(input_path: Path, output_path: Path, crf: int, preset: str) -> None:
    reader = decord.VideoReader(str(input_path), ctx=decord.cpu(0))
    if len(reader) == 0:
        raise ValueError(f"No frames found in {input_path}")

    height, width, _ = reader[0].shape
    fps = float(reader.get_avg_fps() or 30.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio_ffmpeg.write_frames(
        str(output_path),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        output_params=[
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-preset",
            preset,
            "-crf",
            str(crf),
        ],
    )
    writer.send(None)

    total = len(reader)
    checkpoint = max(total // 20, 1)
    for index in range(total):
        frame = reader[index].asnumpy()
        writer.send(frame)
        if index % checkpoint == 0 or index + 1 == total:
            print(f"{input_path.name}: {index + 1}/{total} frames")

    writer.close()


def rewrite_task_paths(tasks_path: Path, output_tasks_path: Path, source_segment: str, target_segment: str) -> None:
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    for task in tasks:
        video_value = task["data"]["video"]
        task["data"]["video"] = video_value.replace(source_segment, target_segment)
    output_tasks_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcode Label Studio videos to browser-friendly H.264 MP4 files.")
    parser.add_argument(
        "--input-dir",
        default="egocentric_samples/worker001_full_playable",
        help="Directory containing source MP4 files.",
    )
    parser.add_argument(
        "--output-dir",
        default="egocentric_samples/worker001_browser_playable",
        help="Directory for transcoded H.264 MP4 files.",
    )
    parser.add_argument(
        "--tasks-json",
        default="egocentric_samples/label_studio_worker001_full_tasks.json",
        help="Existing task JSON to rewrite.",
    )
    parser.add_argument(
        "--output-tasks-json",
        default="egocentric_samples/label_studio_worker001_browser_tasks.json",
        help="Output task JSON that points at transcoded files.",
    )
    parser.add_argument("--crf", type=int, default=23, help="H.264 quality setting. Lower is higher quality.")
    parser.add_argument("--preset", default="veryfast", help="ffmpeg x264 preset.")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of videos to process.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    tasks_path = Path(args.tasks_json)
    output_tasks_path = Path(args.output_tasks_json)

    input_videos = sorted(input_dir.glob("*.mp4"))
    if args.limit > 0:
        input_videos = input_videos[: args.limit]
    if not input_videos:
        raise SystemExit(f"No MP4 files found in {input_dir}")

    for input_video in input_videos:
        output_video = output_dir / input_video.name
        if output_video.exists():
            print(f"Skipping existing {output_video.name}")
            continue
        print(f"Transcoding {input_video.name} -> {output_video.name}")
        transcode_video(input_video, output_video, crf=args.crf, preset=args.preset)

    rewrite_task_paths(
        tasks_path,
        output_tasks_path,
        "worker001_full_playable",
        "worker001_browser_playable",
    )
    print(f"Wrote {output_tasks_path}")


if __name__ == "__main__":
    main()
