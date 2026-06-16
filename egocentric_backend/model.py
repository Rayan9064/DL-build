from __future__ import annotations

import logging
import os
import tempfile
import uuid
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

# Edit this dictionary first when you replace the starter labels with your
# factory-specific action taxonomy.
sample_label_mapping = {
    "assemble": "Assemble",
    "building": "Assemble",
    "carpentry": "Assemble",
    "clean and jerk": "Assemble",
    "fixing": "Assemble",
    "hammer": "Assemble",
    "making": "Assemble",
    "welding": "Assemble",
    "woodworking": "Assemble",
    "checking": "Inspect",
    "looking": "Inspect",
    "reading": "Inspect",
    "review": "Inspect",
    "watching": "Inspect",
    "waiting": "Idle",
    "standing": "Idle",
    "sitting": "Idle",
    "sleeping": "Idle",
}


class EgocentricActionBackend(LabelStudioMLBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = R2Plus1D_18_Weights.KINETICS400_V1
        self.categories = self.weights.meta["categories"]
        self.transforms = self.weights.transforms()
        self.model = r2plus1d_18(weights=self.weights).to(self.device)
        self.model.eval()

    def predict(self, tasks: list[dict[str, Any]], context: dict[str, Any] | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []

        for task in tasks:
            try:
                video_path = self._resolve_video_path(task)
                label, score, kinetics_label = self._predict_clip(video_path)
                duration = self._video_duration(video_path)
                predictions.append(self._format_prediction(task, label, score, duration, kinetics_label))
            except Exception as exc:
                LOGGER.exception("Prediction failed for task %s", task.get("id"))
                predictions.append(
                    self._format_prediction(
                        task,
                        "Idle",
                        0.0,
                        self._task_duration(task),
                        f"error: {exc}",
                    )
                )

        return predictions

    def fit(self, event: str, data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        LOGGER.info("Received training event %s. Fine-tuning is not implemented in this starter.", event)
        return {"status": "ok", "message": "Training hook received; fine-tuning is not implemented yet."}

    def _predict_clip(self, video_path: Path) -> tuple[str, float, str]:
        video_reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        frame_count = len(video_reader)
        if frame_count == 0:
            raise ValueError(f"No frames found in {video_path}")

        frame_indices = np.linspace(0, frame_count - 1, num=16, dtype=np.int64)
        frames = video_reader.get_batch(frame_indices).asnumpy()
        clip = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        clip = self.transforms(clip).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(clip)
            probabilities = torch.softmax(logits, dim=1)
            score_tensor, class_tensor = probabilities.max(dim=1)

        kinetics_label = self.categories[int(class_tensor.item())]
        target_label = self._map_kinetics_label(kinetics_label)
        return target_label, float(score_tensor.item()), kinetics_label

    def _map_kinetics_label(self, kinetics_label: str) -> str:
        normalized = kinetics_label.lower().replace("_", " ")
        for phrase, target_label in sample_label_mapping.items():
            if phrase in normalized:
                return target_label

        if any(word in normalized for word in ("holding", "moving", "using", "picking", "throwing", "lifting")):
            return "Assemble"
        if any(word in normalized for word in ("watching", "reading", "looking", "checking")):
            return "Inspect"
        return "Idle"

    def _format_prediction(
        self,
        task: dict[str, Any],
        label: str,
        score: float,
        duration: float,
        kinetics_label: str,
    ) -> dict[str, Any]:
        task_id = task.get("id", "task")
        region_id = f"{task_id}-{uuid.uuid4().hex[:8]}"
        return {
            "model_version": "r2plus1d_18_kinetics400_starter",
            "score": score,
            "result": [
                {
                    "id": region_id,
                    "from_name": "actions",
                    "to_name": "video",
                    "type": "timelinelabels",
                    "origin": "prediction",
                    "readonly": True,
                    "value": {
                        "ranges": [
                            {
                                "start": 0,
                                "end": max(duration, 0.001),
                            }
                        ],
                        "timelinelabels": [label],
                    },
                    "score": score,
                    "meta": {"kinetics_label": kinetics_label},
                }
            ],
        }

    def _task_duration(self, task: dict[str, Any]) -> float:
        meta = task.get("meta") or {}
        duration = meta.get("duration_sec")
        if duration is None:
            duration = meta.get("duration")
        try:
            return max(float(duration), 0.001)
        except (TypeError, ValueError):
            return 1.0

    def _video_duration(self, video_path: Path) -> float:
        video_reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        fps = video_reader.get_avg_fps() or 30.0
        return float(len(video_reader) / fps)

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
