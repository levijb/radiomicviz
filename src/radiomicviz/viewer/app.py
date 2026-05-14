"""
Flask server for the RadiomicViz browser viewer.

Routes:
  GET /              → viewer.html
  GET /data/<file>   → serve a NIfTI from the registered files dict
  GET /api/volumes   → JSON manifest of available volumes
"""
from __future__ import annotations

from pathlib import Path

from flask import Flask, abort, jsonify, render_template, send_file


def create_app(files: dict[str, str], manifest: dict) -> Flask:
    """
    Build the Flask app.

    Parameters
    ----------
    files : dict[str, str]
        Mapping of URL filename → absolute path on disk.
    manifest : dict
        {
          "image": "image.nii.gz",
          "mask": "mask.nii.gz" | null,
          "overlays": ["feat.nii.gz", ...],
          "feature_4d": "maps.nii.gz" | null,
          "feature_4d_features": [...],
          "feature_4d_n_frames": int,
        }
    """
    app = Flask(__name__, template_folder="templates")
    app.config["FILES"] = files
    app.config["MANIFEST"] = manifest

    @app.route("/")
    def index():
        return render_template("viewer.html", manifest=manifest)

    @app.route("/data/<path:filename>")
    def serve_data(filename):
        path = app.config["FILES"].get(filename)
        if not path or not Path(path).exists():
            abort(404)
        return send_file(str(path), mimetype="application/octet-stream")

    @app.route("/api/volumes")
    def volumes():
        return jsonify(app.config["MANIFEST"])

    return app
