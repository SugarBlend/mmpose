import os
import logging.config
import argparse

from flask import request, jsonify
from flasgger import Swagger

from label_studio_ml.api import init_app
from model import PoseEstimationModel
from pathlib import Path


log_level = os.getenv("LOG_LEVEL", "INFO")

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": log_level,
            "stream": "ext://sys.stdout",
            "formatter": "standard",
        }
    },
    "root": {
        "level": log_level,
        "handlers": ["console"],
    },
})


SWAGGER_CONFIG = {
    "headers": [],
    "specs": [
        {
            "endpoint": "swagger_json",
            "route": "/swagger.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "swagger_ui": False,
}

with Path(__file__).parent.joinpath("swagger-ui.html").open("r", encoding="utf-8") as file:
    _SWAGGER_UI_HTML = file.read()

# Module-level singleton — set once in create_app(), used by /reload endpoint.
_model_instance: PoseEstimationModel | None = None


def create_app(**init_kwargs):
    global _model_instance

    # Wrap PoseEstimationModel.setup() so the first instance to finish
    # loading registers itself in our module-level variable.
    _original_setup = PoseEstimationModel.setup

    def _setup_and_register(self):
        _original_setup(self)
        global _model_instance
        _model_instance = self
        PoseEstimationModel.setup = _original_setup  # unwrap after first call

    PoseEstimationModel.setup = _setup_and_register

    app = init_app(model_class=PoseEstimationModel, **init_kwargs)

    Swagger(app, template_file=Path(__file__).parent.joinpath("swagger-template.json").as_posix(), config=SWAGGER_CONFIG)

    @app.route("/api/docs")
    @app.route("/api/docs/")
    def swagger_ui():
        return _SWAGGER_UI_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

    @app.route("/health-ready", methods=["GET"])
    def health_ready():
        status = getattr(PoseEstimationModel, "_status", "loading")
        if status == "ready":
            return jsonify({"status": "ready"}), 200

        if status == "error":
            return jsonify({"status": "error"}), 500

        return jsonify({"status": "loading"}), 503

    @app.route("/reload", methods=["POST"])
    def reload_models():
        body = request.get_json(silent=True) or {}

        model_instance = _model_instance
        if model_instance is None:
            return jsonify({"error": "Model not initialised yet"}), 500

        # Pass only non-empty values so reload_models() keeps current paths for omitted fields
        kwargs = {k: v for k, v in {
            "pose_config": body.get("pose_config"),
            "pose_checkpoint": body.get("pose_checkpoint"),
            "detector_checkpoint": body.get("detector_checkpoint"),
        }.items() if v}

        try:
            result = model_instance.reload_models(**kwargs)
            return jsonify({"status": "reloading", "result": result}), 202
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=9090)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)
