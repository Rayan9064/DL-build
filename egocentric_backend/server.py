from __future__ import annotations

import argparse
import logging
from threading import Lock
from typing import Any

from flask import Flask, jsonify, request

from .model import EgocentricActionBackend


LOGGER = logging.getLogger(__name__)
MODEL_VERSION = "r2plus1d_18_kinetics400_segmented_v2"

app = Flask(__name__)
_model: EgocentricActionBackend | None = None
_model_lock = Lock()


def get_model(label_config: str | None = None, force_reload: bool = False) -> EgocentricActionBackend:
    global _model
    with _model_lock:
        if _model is None or force_reload:
            _model = EgocentricActionBackend(label_config=label_config)
        return _model


@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "UP", "model_version": MODEL_VERSION, "v2": False})


@app.route("/setup", methods=["POST"])
def setup():
    payload: dict[str, Any] = request.get_json(silent=True) or {}
    label_config = payload.get("schema") or payload.get("label_config")
    force_reload = bool(payload.get("force_reload", False))
    get_model(label_config=label_config, force_reload=force_reload)
    return jsonify({"model_version": MODEL_VERSION})


@app.route("/predict", methods=["POST"])
def predict():
    payload: dict[str, Any] = request.get_json(silent=True) or {}
    tasks = payload.get("tasks") or []
    if not isinstance(tasks, list):
        return jsonify({"error": "Request field 'tasks' must be a list"}), 400

    label_config = payload.get("label_config")
    predictions = get_model(label_config=label_config).predict(tasks, context=payload.get("context"))
    return jsonify({"results": predictions, "model_version": MODEL_VERSION})


def main() -> None:
    parser = argparse.ArgumentParser(description="Egocentric Label Studio ML backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--preload", action="store_true", help="Load the video model before starting the HTTP server")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.preload:
        get_model()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
