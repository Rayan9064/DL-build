from __future__ import annotations

import argparse
import os
from typing import Any

import requests


def list_tasks(base_url: str, token: str, project_id: int, page_size: int = 100) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Token {token}"}
    page = 1
    tasks: list[dict[str, Any]] = []

    while True:
        response = requests.get(
            f"{base_url}/api/tasks",
            headers=headers,
            params={"project": project_id, "page": page, "page_size": page_size},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        page_tasks = payload.get("tasks", [])
        tasks.extend(page_tasks)

        total = payload.get("total", len(tasks))
        if len(tasks) >= total or not page_tasks:
            break
        page += 1

    return tasks


def delete_annotations(base_url: str, token: str, task: dict[str, Any]) -> None:
    headers = {"Authorization": f"Token {token}"}
    response = requests.get(
        f"{base_url}/api/tasks/{task['id']}/annotations/",
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    annotations = payload if isinstance(payload, list) else payload.get("annotations") or payload.get("results") or payload.get("value") or []

    for annotation in annotations:
        annotation_id = annotation["id"]
        response = requests.delete(
            f"{base_url}/api/annotations/{annotation_id}",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()


def _task_fps(task: dict[str, Any]) -> float:
    meta = task.get("meta") or {}
    fps = meta.get("fps") or 30.0
    try:
        return max(float(fps), 1.0)
    except (TypeError, ValueError):
        return 30.0


def _task_frame_count(task: dict[str, Any]) -> int:
    meta = task.get("meta") or {}
    fps = _task_fps(task)
    duration = meta.get("duration_sec")
    if duration is None:
        duration = meta.get("duration")

    try:
        return max(int(round(float(duration) * fps)), 1)
    except (TypeError, ValueError):
        return 1


def _normalize_timeline_value(value: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    ranges = value.get("ranges") or []
    if not ranges:
        return value

    fps = _task_fps(task)
    max_frame = _task_frame_count(task) - 1
    normalized_ranges = []

    for index, range_item in enumerate(ranges):
        start = range_item.get("start")
        end = range_item.get("end")

        # Label Studio 1.16 TimelineLabels expects integer frame indices.
        # Older code paths in this repo stored seconds, often as floats.
        if isinstance(start, float):
            start = int(round(start * fps))
        elif isinstance(start, int):
            start = int(start)
        elif start is not None:
            start = int(round(float(start)))

        if isinstance(end, float):
            end = int(round(end * fps))
        elif isinstance(end, int):
            end = int(end)
        elif end is not None:
            end = int(round(float(end)))

        if start is None:
            start = 0 if index == 0 else None
        if end is None:
            end = max_frame

        start = max(int(start), 0)
        end = max(int(end), start)
        normalized_ranges.append({"start": min(start, max_frame), "end": min(end, max_frame)})

    normalized_value = dict(value)
    normalized_value["ranges"] = normalized_ranges
    return normalized_value


def to_annotation_results(prediction: dict[str, Any], task: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in prediction.get("result", []):
        value = item["value"]
        if item.get("type") == "timelinelabels":
            value = _normalize_timeline_value(value, task)
        results.append(
            {
                "id": item.get("id"),
                "from_name": item["from_name"],
                "to_name": item["to_name"],
                "type": item["type"],
                "origin": "manual",
                "value": value,
            }
        )
    return results


def create_annotation(base_url: str, token: str, task_id: int, results: list[dict[str, Any]]) -> dict[str, Any]:
    headers = {"Authorization": f"Token {token}"}
    payload = {"result": results}
    response = requests.post(
        f"{base_url}/api/tasks/{task_id}/annotations/",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def predict(backend_url: str, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    response = requests.post(
        f"{backend_url}/predict",
        json={"tasks": tasks},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["results"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply ML backend outputs as real Label Studio annotations.")
    parser.add_argument("--label-studio-url", default="http://localhost:8080")
    parser.add_argument("--backend-url", default="http://localhost:9090")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--token", default=os.getenv("LABEL_STUDIO_API_TOKEN"))
    parser.add_argument("--task-ids", nargs="*", type=int, help="Optional specific task ids to annotate")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing annotations before creating new ones")
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("Missing Label Studio API token. Pass --token or set LABEL_STUDIO_API_TOKEN.")

    tasks = list_tasks(args.label_studio_url, args.token, args.project_id)
    if args.task_ids:
        wanted = set(args.task_ids)
        tasks = [task for task in tasks if task["id"] in wanted]

    if not tasks:
        raise SystemExit("No matching tasks found.")

    backend_tasks = [{"id": task["id"], "data": task["data"], "meta": task.get("meta") or {}} for task in tasks]
    predictions = predict(args.backend_url, backend_tasks)

    for task, prediction in zip(tasks, predictions):
        if args.overwrite and task.get("annotations"):
            delete_annotations(args.label_studio_url, args.token, task)

        results = to_annotation_results(prediction, task)
        annotation = create_annotation(args.label_studio_url, args.token, task["id"], results)
        label_names = []
        for result in results:
            label_names.extend(result["value"].get("timelinelabels", []))
        print(f"Task {task['id']}: created annotation {annotation['id']} with labels {label_names}")


if __name__ == "__main__":
    main()
