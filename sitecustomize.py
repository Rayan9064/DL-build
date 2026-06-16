from __future__ import annotations

import logging
from importlib import import_module
from typing import Any


LOGGER = logging.getLogger(__name__)


def _import_any(*module_names: str):
    for module_name in module_names:
        try:
            return import_module(module_name)
        except Exception:
            continue
    return None


def _normalize_task_payload(data: dict[str, Any]) -> dict[str, Any]:
    # Older Label Studio frontends expect file_upload to be an integer ID or null.
    # When tasks are imported from a local JSON file, the backend can emit the
    # uploaded filename string instead, which breaks MST hydration.
    if isinstance(data.get("file_upload"), str):
        data["file_upload"] = None

    for annotation in data.get("annotations") or []:
        completed_by = annotation.get("completed_by")
        if isinstance(completed_by, dict):
            annotation["completed_by"] = completed_by.get("id")

    return data


def _patch_label_studio_task_api() -> None:
    tasks_api = _import_any("tasks.api", "label_studio.tasks.api")
    if tasks_api is None:
        return
    try:
        TaskAPI = tasks_api.TaskAPI
        evaluate_predictions = tasks_api.evaluate_predictions
        from rest_framework.response import Response
    except Exception:
        return

    original_get = getattr(TaskAPI, "get", None)
    if original_get is None or getattr(original_get, "_egocentric_patched", False):
        return

    def patched_get(self, request, pk):
        context = self.get_retrieve_serializer_context(request)
        context["project"] = project = self.task.project

        if (project.evaluate_predictions_automatically or project.show_collab_predictions) and not self.task.predictions.exists():
            evaluate_predictions([self.task])
            self.task.refresh_from_db()

        serializer = self.get_serializer_class()(self.task, many=False, context=context)
        return Response(_normalize_task_payload(serializer.data))

    patched_get._egocentric_patched = True  # type: ignore[attr-defined]
    TaskAPI.get = patched_get
    LOGGER.info("Applied Label Studio task API compatibility patch")


def _patch_label_studio_data_manager_serializer() -> None:
    serializers_module = _import_any("data_manager.serializers", "label_studio.data_manager.serializers")
    if serializers_module is None:
        return

    serializer_cls = getattr(serializers_module, "DataManagerTaskSerializer", None)
    if serializer_cls is None or getattr(serializer_cls.get_file_upload, "_egocentric_patched", False):
        return

    @staticmethod
    def patched_get_file_upload(task):
        return None

    patched_get_file_upload._egocentric_patched = True  # type: ignore[attr-defined]
    serializer_cls.get_file_upload = patched_get_file_upload
    LOGGER.info("Patched DataManagerTaskSerializer.get_file_upload for frontend compatibility")


_patch_label_studio_task_api()
_patch_label_studio_data_manager_serializer()
