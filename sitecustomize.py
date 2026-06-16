from __future__ import annotations

import logging
import mimetypes
import os
import posixpath
from importlib import import_module
from pathlib import Path
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

    if "annotations" in data:
        data["annotations"] = [_normalize_annotation_payload(annotation) for annotation in data.get("annotations") or []]

    return data


def _normalize_annotation_payload(annotation: dict[str, Any]) -> dict[str, Any]:
    completed_by = annotation.get("completed_by")
    if isinstance(completed_by, dict):
        completed_by = completed_by.get("id")

    # Keep the annotation payload close to Label Studio 1.16 task fixtures.
    safe_annotation = {
        "id": annotation.get("id"),
        "result": annotation.get("result") or [],
        "created_username": annotation.get("created_username", ""),
        "created_ago": annotation.get("created_ago", ""),
        "completed_by": completed_by,
        "was_cancelled": bool(annotation.get("was_cancelled", False)),
        "ground_truth": bool(annotation.get("ground_truth", False)),
        "created_at": annotation.get("created_at"),
        "updated_at": annotation.get("updated_at"),
        "lead_time": annotation.get("lead_time"),
    }
    return safe_annotation


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


def _patch_label_studio_annotations_api() -> None:
    tasks_api = _import_any("tasks.api", "label_studio.tasks.api")
    if tasks_api is None:
        return
    try:
        AnnotationsListAPI = tasks_api.AnnotationsListAPI
        AnnotationAPI = tasks_api.AnnotationAPI
    except Exception:
        return

    original_list_get = getattr(AnnotationsListAPI, "get", None)
    if original_list_get is not None and not getattr(original_list_get, "_egocentric_patched", False):

        def patched_list_get(self, request, *args, **kwargs):
            response = original_list_get(self, request, *args, **kwargs)
            if isinstance(response.data, list):
                response.data = [_normalize_annotation_payload(item) for item in response.data]
            return response

        patched_list_get._egocentric_patched = True  # type: ignore[attr-defined]
        AnnotationsListAPI.get = patched_list_get

    original_annotation_get = getattr(AnnotationAPI, "get", None)
    if original_annotation_get is not None and not getattr(original_annotation_get, "_egocentric_patched", False):

        def patched_annotation_get(self, request, *args, **kwargs):
            response = original_annotation_get(self, request, *args, **kwargs)
            if isinstance(response.data, dict):
                response.data = _normalize_annotation_payload(response.data)
            return response

        patched_annotation_get._egocentric_patched = True  # type: ignore[attr-defined]
        AnnotationAPI.get = patched_annotation_get

    LOGGER.info("Patched Label Studio annotation APIs for frontend compatibility")


def _patch_label_studio_localfiles_view() -> None:
    views_module = _import_any("core.views", "label_studio.core.views")
    if views_module is None:
        return

    original_view = getattr(views_module, "localfiles_data", None)
    if original_view is None or getattr(original_view, "_egocentric_patched", False):
        return

    try:
        from django.conf import settings
        from django.db.models import CharField, F, Value
        from django.http import HttpResponseForbidden, HttpResponseNotFound
        from django.utils._os import safe_join
        from io_storages.localfiles.models import LocalFilesImportStorage
        from ranged_fileresponse import RangedFileResponse
        from rest_framework.decorators import api_view, permission_classes
        from rest_framework.permissions import IsAuthenticated
    except Exception:
        return

    @api_view(["GET"])
    @permission_classes([IsAuthenticated])
    def patched_localfiles_data(request):
        path = request.GET.get("d")
        if settings.LOCAL_FILES_SERVING_ENABLED is False:
            return HttpResponseForbidden(
                "Serving local files can be dangerous, so it's disabled by default. "
                "You can enable it with LOCAL_FILES_SERVING_ENABLED environment variable, "
                "please check docs: https://labelstud.io/guide/storage.html#Local-storage"
            )

        if not path or not request.user.is_authenticated:
            return HttpResponseForbidden()

        normalized_path = posixpath.normpath(path).lstrip("/")
        candidate_roots = [settings.LOCAL_FILES_DOCUMENT_ROOT, os.getcwd()]
        full_path = None
        for root in candidate_roots:
            try:
                candidate = Path(safe_join(root, normalized_path))
            except Exception:
                continue
            if candidate.exists():
                full_path = candidate
                break

        if full_path is None:
            return HttpResponseNotFound()

        user_has_permissions = False
        localfiles_storage = LocalFilesImportStorage.objects.annotate(
            _full_path=Value(os.path.dirname(full_path), output_field=CharField())
        ).filter(_full_path__startswith=F("path"))
        if localfiles_storage.exists():
            user_has_permissions = any(storage.project.has_permission(request.user) for storage in localfiles_storage)

        if not user_has_permissions:
            # Local dev compatibility: allow authenticated users to load files
            # from the configured document root or current working directory
            # without requiring a separate Local Files Storage connection.
            full_path_str = str(full_path.resolve())
            allowed_roots = {
                str(Path(settings.LOCAL_FILES_DOCUMENT_ROOT).resolve()),
                str(Path(os.getcwd()).resolve()),
            }
            user_has_permissions = any(full_path_str.startswith(root) for root in allowed_roots)

        if not user_has_permissions or not full_path.exists():
            return HttpResponseNotFound()

        content_type, encoding = mimetypes.guess_type(str(full_path))
        content_type = content_type or "application/octet-stream"
        return RangedFileResponse(request, open(full_path, mode="rb"), content_type)

    patched_localfiles_data._egocentric_patched = True  # type: ignore[attr-defined]
    views_module.localfiles_data = patched_localfiles_data
    LOGGER.info("Patched Label Studio localfiles_data for local dev file serving")


_patch_label_studio_task_api()
_patch_label_studio_data_manager_serializer()
_patch_label_studio_annotations_api()
_patch_label_studio_localfiles_view()
