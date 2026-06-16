from __future__ import annotations

import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import decord
import numpy as np
import requests
import torch
from label_studio_ml.model import LabelStudioMLBase
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18


LOGGER = logging.getLogger(__name__)

TARGET_LABELS = ("Assemble", "Inspect", "Idle")

# Timeline inference runs over short overlapping windows, then converts the
# window decisions into non-overlapping frame spans for Label Studio.
WINDOW_DURATION_SEC = 8.0
WINDOW_STRIDE_SEC = 8.0
WINDOW_SAMPLE_COUNT = 16
TOP_K = 5
MIN_REGION_SEC = 8.0
BATCH_SIZE = 8

# These thresholds are intentionally conservative. They do not replace the
# classifier, but they keep the fallback path from calling everything "Idle".
MOTION_IDLE_THRESHOLD = 0.018
MOTION_ACTIVE_THRESHOLD = 0.040

# Edit this dictionary first when you replace the starter labels with your
# factory-specific action taxonomy.
sample_label_mapping = {
    "assemble": "Assemble",
    "assembling computer": "Assemble",
    "bending metal": "Assemble",
    "blowing glass": "Assemble",
    "building": "Assemble",
    "carpentry": "Assemble",
    "changing oil": "Assemble",
    "changing wheel": "Assemble",
    "clean and jerk": "Assemble",
    "fixing": "Assemble",
    "hammer": "Assemble",
    "making": "Assemble",
    "operating": "Assemble",
    "playing paintball": "Assemble",
    "sharpening pencil": "Assemble",
    "spray painting": "Assemble",
    "unboxing": "Assemble",
    "welding": "Assemble",
    "woodworking": "Assemble",
    "checking": "Inspect",
    "inspecting": "Inspect",
    "looking": "Inspect",
    "reading": "Inspect",
    "review": "Inspect",
    "watching": "Inspect",
    "waiting": "Idle",
    "standing": "Idle",
    "sitting": "Idle",
    "sleeping": "Idle",
}

ASSEMBLE_HINTS = (
    "assemble",
    "bending",
    "blowing glass",
    "building",
    "changing",
    "cleaning",
    "fixing",
    "hammer",
    "making",
    "operating",
    "opening",
    "packing",
    "painting",
    "polishing",
    "repair",
    "sanding",
    "sawing",
    "sharpening",
    "spray",
    "tightening",
    "turning",
    "unboxing",
    "using",
    "welding",
    "working",
    "wrapping",
)

INSPECT_HINTS = (
    "checking",
    "examining",
    "inspecting",
    "looking",
    "measuring",
    "observ",
    "reading",
    "review",
    "scan",
    "searching",
    "sorting",
    "studying",
    "watching",
)

IDLE_HINTS = (
    "sleeping",
    "sitting",
    "standing",
    "waiting",
    "resting",
    "sniffing",
    "staring",
)


@dataclass(slots=True)
class WindowPrediction:
    start_frame: int
    end_frame: int
    label: str
    score: float
    kinetics_label: str
    motion_score: float


class EgocentricActionBackend(LabelStudioMLBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = R2Plus1D_18_Weights.KINETICS400_V1
        self.categories = self.weights.meta["categories"]
        self.transforms = self.weights.transforms()
        self.model = r2plus1d_18(weights=self.weights).to(self.device)
        self.model.eval()

    def predict(
        self,
        tasks: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []

        for task in tasks:
            try:
                video_path = self._resolve_video_path(task)
                windows, frame_count = self._predict_windows(video_path)
                segments = self._windows_to_segments(windows, frame_count, self._task_fps(task))
                predictions.append(self._format_prediction(task, segments))
            except Exception as exc:
                LOGGER.exception("Prediction failed for task %s", task.get("id"))
                predictions.append(self._fallback_prediction(task, f"error: {exc}"))

        return predictions

    def fit(self, event: str, data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        LOGGER.info("Received training event %s. Fine-tuning is not implemented in this starter.", event)
        return {"status": "ok", "message": "Training hook received; fine-tuning is not implemented yet."}

    def _predict_windows(self, video_path: Path) -> tuple[list[WindowPrediction], int]:
        video_reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        frame_count = len(video_reader)
        if frame_count == 0:
            raise ValueError(f"No frames found in {video_path}")

        fps = video_reader.get_avg_fps() or 30.0
        window_size = max(int(round(WINDOW_DURATION_SEC * fps)), WINDOW_SAMPLE_COUNT)
        stride_size = max(int(round(WINDOW_STRIDE_SEC * fps)), 1)
        starts = self._window_starts(frame_count, window_size, stride_size)

        windows: list[WindowPrediction] = []
        for batch_start in range(0, len(starts), BATCH_SIZE):
            batch_starts = starts[batch_start : batch_start + BATCH_SIZE]
            batch_specs: list[tuple[int, int, int, int]] = []
            batch_frames: list[np.ndarray] = []
            motion_scores: list[float] = []

            for start in batch_starts:
                sample_start = min(start, max(frame_count - window_size, 0))
                sample_end = min(sample_start + window_size, frame_count)
                frame_indices = np.linspace(sample_start, sample_end - 1, num=WINDOW_SAMPLE_COUNT, dtype=np.int64)
                frames = video_reader.get_batch(frame_indices).asnumpy()
                batch_specs.append((start, sample_start, sample_end, len(batch_specs)))
                batch_frames.append(frames)
                motion_scores.append(self._motion_score(frames))

            ranked_predictions = self._classify_window_batch(batch_frames)

            for local_index, (start, _sample_start, _sample_end, prediction_index) in enumerate(batch_specs):
                global_index = batch_start + local_index
                interval_start = start
                if global_index + 1 < len(starts):
                    interval_end = max(starts[global_index + 1] - 1, interval_start)
                else:
                    interval_end = frame_count - 1

                ranked = ranked_predictions[prediction_index]
                label, score = self._aggregate_window_label(ranked, motion_scores[prediction_index])
                kinetics_label = ranked[0][0]

                windows.append(
                    WindowPrediction(
                        start_frame=interval_start,
                        end_frame=interval_end,
                        label=label,
                        score=score,
                        kinetics_label=kinetics_label,
                        motion_score=motion_scores[prediction_index],
                    )
                )

        return self._smooth_windows(windows, fps), frame_count

    def _window_starts(self, frame_count: int, window_size: int, stride_size: int) -> list[int]:
        if frame_count <= window_size:
            return [0]

        starts = list(range(0, max(frame_count - window_size + 1, 1), stride_size))
        tail_start = max(frame_count - window_size, 0)
        if not starts or starts[-1] != tail_start:
            starts.append(tail_start)

        # Keep non-overlapping timeline assignment monotonic.
        return sorted(set(max(0, start) for start in starts))

    def _classify_window_batch(self, batch_frames: list[np.ndarray]) -> list[list[tuple[str, float]]]:
        batch_clips = [
            self.transforms(torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0) for frames in batch_frames
        ]
        clips_tensor = torch.stack(batch_clips, dim=0).to(self.device)

        with torch.inference_mode():
            probabilities = torch.softmax(self.model(clips_tensor), dim=1)

        ranked_predictions: list[list[tuple[str, float]]] = []
        for row in probabilities:
            topk = torch.topk(row, TOP_K)
            ranked_predictions.append(
                [(self.categories[int(index)], float(score)) for score, index in zip(topk.values, topk.indices)]
            )
        return ranked_predictions

    def _aggregate_window_label(self, ranked: list[tuple[str, float]], motion_score: float) -> tuple[str, float]:
        scores = {label: 0.0 for label in TARGET_LABELS}

        for kinetics_label, probability in ranked:
            mapped = self._map_kinetics_label(kinetics_label)
            if mapped:
                scores[mapped] += probability

        # Motion helps when Kinetics labels are irrelevant or too noisy.
        if motion_score >= MOTION_ACTIVE_THRESHOLD:
            scores["Assemble"] += min(0.40, motion_score * 4.0)
        elif motion_score <= MOTION_IDLE_THRESHOLD:
            scores["Idle"] += 0.25
        else:
            scores["Inspect"] += 0.12

        top_label = max(scores, key=scores.get)
        top_score = scores[top_label]

        # If the classifier has no useful semantic match, prefer a motion-based
        # fallback that biases active factory work toward Assemble.
        if top_score < 0.20:
            if motion_score >= MOTION_ACTIVE_THRESHOLD:
                return "Assemble", max(0.20, motion_score)
            if motion_score <= MOTION_IDLE_THRESHOLD:
                return "Idle", max(0.15, 1.0 - motion_score)
            return "Inspect", max(0.15, motion_score)

        return top_label, top_score

    def _map_kinetics_label(self, kinetics_label: str) -> str | None:
        normalized = kinetics_label.lower().replace("_", " ")

        for phrase, target_label in sample_label_mapping.items():
            if phrase in normalized:
                return target_label

        if any(hint in normalized for hint in ASSEMBLE_HINTS):
            return "Assemble"
        if any(hint in normalized for hint in INSPECT_HINTS):
            return "Inspect"
        if any(hint in normalized for hint in IDLE_HINTS):
            return "Idle"
        return None

    def _motion_score(self, frames: np.ndarray) -> float:
        if len(frames) < 2:
            return 0.0

        # Downsample before frame differencing to keep this cheap.
        reduced = frames[:, ::8, ::8, :].astype(np.float32) / 255.0
        deltas = np.abs(np.diff(reduced, axis=0))
        return float(deltas.mean())

    def _smooth_windows(self, windows: list[WindowPrediction], fps: float) -> list[WindowPrediction]:
        if len(windows) < 3:
            return windows

        smoothed = list(windows)
        short_frames = max(int(round(MIN_REGION_SEC * fps)), 1)

        for index in range(1, len(smoothed) - 1):
            previous_window = smoothed[index - 1]
            current_window = smoothed[index]
            next_window = smoothed[index + 1]
            current_duration = current_window.end_frame - current_window.start_frame + 1

            if (
                current_duration <= short_frames
                and previous_window.label == next_window.label
                and current_window.label != previous_window.label
                and current_window.score <= max(previous_window.score, next_window.score)
            ):
                smoothed[index] = WindowPrediction(
                    start_frame=current_window.start_frame,
                    end_frame=current_window.end_frame,
                    label=previous_window.label,
                    score=max(previous_window.score, next_window.score),
                    kinetics_label=current_window.kinetics_label,
                    motion_score=current_window.motion_score,
                )

        return smoothed

    def _windows_to_segments(
        self,
        windows: list[WindowPrediction],
        frame_count: int,
        fps: float,
    ) -> list[WindowPrediction]:
        if not windows:
            return [
                WindowPrediction(
                    start_frame=0,
                    end_frame=max(frame_count - 1, 0),
                    label="Idle",
                    score=0.0,
                    kinetics_label="empty",
                    motion_score=0.0,
                )
            ]

        merged: list[WindowPrediction] = []
        for window in windows:
            if merged and merged[-1].label == window.label:
                previous = merged[-1]
                merged[-1] = WindowPrediction(
                    start_frame=previous.start_frame,
                    end_frame=window.end_frame,
                    label=window.label,
                    score=max(previous.score, window.score),
                    kinetics_label=previous.kinetics_label if previous.score >= window.score else window.kinetics_label,
                    motion_score=max(previous.motion_score, window.motion_score),
                )
            else:
                merged.append(window)

        min_region_frames = max(int(round(MIN_REGION_SEC * fps)), 1)
        if len(merged) >= 3:
            adjusted: list[WindowPrediction] = [merged[0]]
            for index in range(1, len(merged) - 1):
                previous = adjusted[-1]
                current = merged[index]
                next_window = merged[index + 1]
                duration = current.end_frame - current.start_frame + 1

                if (
                    duration < min_region_frames
                    and previous.label == next_window.label
                    and current.label != previous.label
                ):
                    adjusted[-1] = WindowPrediction(
                        start_frame=previous.start_frame,
                        end_frame=current.end_frame,
                        label=previous.label,
                        score=max(previous.score, current.score),
                        kinetics_label=previous.kinetics_label,
                        motion_score=max(previous.motion_score, current.motion_score),
                    )
                else:
                    adjusted.append(current)
            adjusted.append(merged[-1])
            merged = []
            for window in adjusted:
                if merged and merged[-1].label == window.label:
                    previous = merged[-1]
                    merged[-1] = WindowPrediction(
                        start_frame=previous.start_frame,
                        end_frame=window.end_frame,
                        label=window.label,
                        score=max(previous.score, window.score),
                        kinetics_label=previous.kinetics_label if previous.score >= window.score else window.kinetics_label,
                        motion_score=max(previous.motion_score, window.motion_score),
                    )
                else:
                    merged.append(window)

        merged[0] = WindowPrediction(
            start_frame=0,
            end_frame=merged[0].end_frame,
            label=merged[0].label,
            score=merged[0].score,
            kinetics_label=merged[0].kinetics_label,
            motion_score=merged[0].motion_score,
        )
        merged[-1] = WindowPrediction(
            start_frame=merged[-1].start_frame,
            end_frame=max(frame_count - 1, merged[-1].start_frame),
            label=merged[-1].label,
            score=merged[-1].score,
            kinetics_label=merged[-1].kinetics_label,
            motion_score=merged[-1].motion_score,
        )
        return merged

    def _format_prediction(self, task: dict[str, Any], segments: list[WindowPrediction]) -> dict[str, Any]:
        task_id = task.get("id", "task")
        results: list[dict[str, Any]] = []

        for segment in segments:
            results.append(
                {
                    "id": f"{task_id}-{uuid.uuid4().hex[:8]}",
                    "from_name": "actions",
                    "to_name": "video",
                    "type": "timelinelabels",
                    "origin": "prediction",
                    "readonly": True,
                    "value": {
                        "ranges": [{"start": segment.start_frame, "end": segment.end_frame}],
                        "timelinelabels": [segment.label],
                    },
                    "score": segment.score,
                    "meta": {
                        "kinetics_label": segment.kinetics_label,
                        "motion_score": round(segment.motion_score, 6),
                    },
                }
            )

        average_score = float(sum(segment.score for segment in segments) / max(len(segments), 1))
        return {
            "model_version": "r2plus1d_18_kinetics400_segmented_v2",
            "score": average_score,
            "result": results,
        }

    def _fallback_prediction(self, task: dict[str, Any], kinetics_label: str) -> dict[str, Any]:
        frame_count = self._task_frame_count(task)
        fallback_segment = WindowPrediction(
            start_frame=0,
            end_frame=max(frame_count - 1, 0),
            label="Idle",
            score=0.0,
            kinetics_label=kinetics_label,
            motion_score=0.0,
        )
        return self._format_prediction(task, [fallback_segment])

    def _task_duration(self, task: dict[str, Any]) -> float:
        meta = task.get("meta") or {}
        duration = meta.get("duration_sec")
        if duration is None:
            duration = meta.get("duration")
        try:
            return max(float(duration), 0.001)
        except (TypeError, ValueError):
            return 1.0

    def _task_fps(self, task: dict[str, Any]) -> float:
        meta = task.get("meta") or {}
        fps = meta.get("fps") or 30.0
        try:
            return max(float(fps), 1.0)
        except (TypeError, ValueError):
            return 30.0

    def _task_frame_count(self, task: dict[str, Any]) -> int:
        duration = self._task_duration(task)
        fps = self._task_fps(task)
        return max(int(round(duration * fps)), 1)

    def _resolve_video_path(self, task: dict[str, Any]) -> Path:
        video_value = (task.get("data") or {}).get("video")
        if not video_value:
            raise ValueError("Task does not contain data.video")

        parsed = urlparse(video_value)
        if parsed.scheme in ("http", "https"):
            if parsed.path == "/data/local-files/" or parsed.path.endswith("/data/local-files/"):
                query_path = parse_qs(parsed.query).get("d", [None])[0]
                if query_path:
                    return self._local_files_path(query_path)
            return self._download_remote_video(video_value)

        if video_value.startswith("/data/local-files/"):
            query_path = parse_qs(parsed.query).get("d", [None])[0]
            if not query_path:
                raise ValueError(f"Could not parse local-files video URL: {video_value}")
            return self._local_files_path(query_path)

        path = Path(unquote(video_value)).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def _local_files_path(self, relative_path: str) -> Path:
        root = Path(os.getenv("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT", Path.cwd())).resolve()
        candidate = (root / unquote(relative_path).lstrip("/")).resolve()
        if not str(candidate).startswith(str(root)):
            raise ValueError(f"Refusing to read outside local files root: {candidate}")
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate

    def _download_remote_video(self, url: str) -> Path:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        try:
            handle.write(response.content)
            return Path(handle.name)
        finally:
            handle.close()


NewModel = EgocentricActionBackend
