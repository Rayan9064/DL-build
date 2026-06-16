import argparse
import json
import logging
import logging.config
import os

from label_studio_ml.api import init_app
from model import EgocentricActionBackend


logging.config.dictConfig(
    {
        "version": 1,
        "formatters": {
            "standard": {
                "format": "[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "stream": "ext://sys.stdout",
                "formatter": "standard",
            }
        },
        "root": {"level": "ERROR", "handlers": ["console"], "propagate": True},
    }
)


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def get_kwargs_from_config(config_path=DEFAULT_CONFIG_PATH):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Backend config.json must contain a JSON object")
    return config


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def parse_kwargs(raw_kwargs):
    parsed = {}
    for key, value in raw_kwargs:
        if value.isdigit():
            parsed[key] = int(value)
        elif value.lower() == "true":
            parsed[key] = True
        elif value.lower() == "false":
            parsed[key] = False
        elif is_float(value):
            parsed[key] = float(value)
        else:
            parsed[key] = value
    return parsed


def create_app(model_dir=None, **kwargs):
    return init_app(
        model_class=EgocentricActionBackend,
        model_dir=os.environ.get("MODEL_DIR", model_dir or os.path.dirname(__file__)),
        redis_queue=os.environ.get("RQ_QUEUE_NAME", "default"),
        redis_host=os.environ.get("REDIS_HOST", "localhost"),
        redis_port=os.environ.get("REDIS_PORT", 6379),
        **kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Egocentric Label Studio ML backend")
    parser.add_argument("-p", "--port", dest="port", type=int, default=9090)
    parser.add_argument("--host", dest="host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--kwargs",
        "--with",
        dest="kwargs",
        metavar="KEY=VAL",
        nargs="+",
        type=lambda kv: kv.split("=", 1),
    )
    parser.add_argument("-d", "--debug", dest="debug", action="store_true")
    parser.add_argument("--log-level", dest="log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--model-dir", dest="model_dir", default=os.path.dirname(__file__))
    parser.add_argument("--check", dest="check", action="store_true")
    args = parser.parse_args()

    if args.log_level:
        logging.root.setLevel(args.log_level)

    model_kwargs = get_kwargs_from_config()
    if args.kwargs:
        model_kwargs.update(parse_kwargs(args.kwargs))

    if args.check:
        print(f'Check "{EgocentricActionBackend.__name__}" instance creation...')
        EgocentricActionBackend(**model_kwargs)

    app = create_app(model_dir=args.model_dir, **model_kwargs)
    app.run(host=args.host, port=args.port, debug=args.debug)
else:
    app = create_app()
