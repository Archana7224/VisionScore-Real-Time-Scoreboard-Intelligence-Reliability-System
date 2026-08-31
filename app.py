"""
VisionScore - Interactive dashboard server.

A thin Flask layer over the existing VisionScore computer-vision pipeline.
It lets a user upload a video or image (or run the bundled demo), executes
the REAL OpenCV + EasyOCR pipeline on the input, and streams live progress
plus the detected/extracted scoreboard results to the dashboard UI.
"""

import os
import threading
import traceback
import uuid
from pathlib import Path

from flask import (Flask, jsonify, render_template, request,
                   send_from_directory, abort)
from werkzeug.utils import secure_filename

import vision_engine

ROOT = Path(__file__).resolve().parent
JOBS_DIR = ROOT / "static" / "jobs"
UPLOAD_DIR = ROOT / "uploads"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300 MB

# In-memory job registry: job_id -> status dict
JOBS = {}
LOCK = threading.Lock()


def _set(job_id, **kw):
    with LOCK:
        JOBS.setdefault(job_id, {})
        JOBS[job_id].update(kw)


def _run_job(job_id, input_path, kind, demo):
    job_dir = JOBS_DIR / job_id

    def progress(update):
        _set(job_id, **update)

    try:
        _set(job_id, state="running", percent=1, stage="init",
             message="Starting VisionScore pipeline...")
        result = vision_engine.run_pipeline(
            input_path=input_path, kind=kind, job_dir=job_dir,
            progress=progress, demo=demo,
        )
        _set(job_id, state="complete", percent=100, result=result,
             message="Analysis complete.")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        _set(job_id, state="error", percent=100,
             message=f"Pipeline error: {exc}")


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/process", methods=["POST"])
def process():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    filename = secure_filename(file.filename)
    ext = Path(filename).suffix.lower()

    if ext in VIDEO_EXT:
        kind = "video"
    elif ext in IMAGE_EXT:
        kind = "image"
    else:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    job_id = uuid.uuid4().hex[:12]
    save_path = UPLOAD_DIR / f"{job_id}{ext}"
    file.save(str(save_path))

    _set(job_id, state="queued", percent=0, kind=kind,
         filename=filename, message="Queued.")

    thread = threading.Thread(
        target=_run_job, args=(job_id, str(save_path), kind, False),
        daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "kind": kind, "filename": filename})


@app.route("/api/demo", methods=["POST"])
def demo():
    job_id = "demo_" + uuid.uuid4().hex[:8]
    _set(job_id, state="queued", percent=0, kind="video",
         filename="bowling_scoreboard.mp4 (bundled)", message="Queued.")

    thread = threading.Thread(
        target=_run_job, args=(job_id, None, "video", True),
        daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "kind": "video",
                    "filename": "bowling_scoreboard.mp4 (bundled)"})


@app.route("/api/status/<job_id>")
def status(job_id):
    with LOCK:
        data = JOBS.get(job_id)
        if data is None:
            return jsonify({"error": "Unknown job."}), 404
        return jsonify(dict(data))


@app.route("/static/jobs/<job_id>/<path:subpath>")
def job_asset(job_id, subpath):
    directory = JOBS_DIR / job_id
    if not directory.exists():
        abort(404)
    return send_from_directory(str(directory), subpath)


@app.route("/api/demo-input")
def demo_input():
    """Ordered list of the bundled video's real frames (input preview)."""
    frame_dir = ROOT / "screenshots" / "training_frames"
    names = sorted(p.name for p in frame_dir.glob("train_*.jpg"))
    return jsonify({"frames": [f"/demo-frame/{n}" for n in names]})


@app.route("/demo-frame/<name>")
def demo_frame(name):
    directory = ROOT / "screenshots" / "training_frames"
    return send_from_directory(str(directory), secure_filename(name))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
