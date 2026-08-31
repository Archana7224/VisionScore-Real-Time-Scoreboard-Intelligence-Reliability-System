"""
VisionScore - Live pipeline engine.

This module wraps the EXISTING VisionScore computer-vision pipeline
(OpenCV localization + EasyOCR extraction + temporal validation +
change detection + reliability analysis) so it can be executed on demand
against a user-uploaded video or image.

Nothing about the underlying tech stack changes: it reuses the real
`detect_scoreboard` localizer from `src/localization_v7.py` and the same
EasyOCR digit-recognition approach used by `src/ocr_scoreboard.py`.
"""

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# Reuse the real localization detector without modifying it.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from localization_v7 import detect_scoreboard, DIRECT_CONFIDENCE, MAX_HOLD_FRAMES  # noqa: E402


# ---------------------------------------------------------------------------
# ROI geometry (from src/ocr_scoreboard.py, defined on a 1920x1080 frame).
# Expressed as ratios so it generalizes to any resolution.
# ---------------------------------------------------------------------------

BASE_W, BASE_H = 1920.0, 1080.0

ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 70, 500, 1760, 830
SCORE_X1, SCORE_Y1, SCORE_X2, SCORE_Y2 = 190, 50, 770, 145

PLAYERS = ["player_1", "player_2", "player_3", "player_4"]

PERSISTENCE_FRAMES = 2

# Target number of sampled frames for a video.
TARGET_FRAMES = 26

_READER = None


def get_reader():
    """Lazily construct the EasyOCR reader (same config as ocr_scoreboard.py)."""
    global _READER
    if _READER is None:
        import easyocr
        _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _READER


# ---------------------------------------------------------------------------
# Frame acquisition
# ---------------------------------------------------------------------------

def _sample_video(path, target=TARGET_FRAMES):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    frames = []
    if total <= 0:
        # Fallback: read sequentially.
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append((idx, frame))
            idx += 1
        cap.release()
        step = max(1, len(frames) // target)
        return [f for i, f in enumerate(frames) if i % step == 0][:target], fps

    step = max(1, total // target)
    picked = []
    idx = 0
    while idx < total and len(picked) < target:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            picked.append((idx, frame))
        idx += step

    cap.release()
    return picked, fps


def _load_frames(input_path, kind):
    """Return list of (label, frame_bgr) and fps."""
    input_path = Path(input_path)
    if kind == "image":
        img = cv2.imread(str(input_path))
        if img is None:
            raise RuntimeError("Could not read the uploaded image.")
        return [("frame_001", img)], 0.0

    picked, fps = _sample_video(input_path)
    frames = []
    for i, (_, frame) in enumerate(picked, start=1):
        frames.append((f"frame_{i:03d}", frame))
    return frames, fps


def _load_demo_frames():
    """Use the real extracted training frames from the bundled video."""
    frame_dir = ROOT / "screenshots" / "training_frames"
    paths = sorted(frame_dir.glob("train_*.jpg"))
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append((p.stem, img))
    return frames, 0.0


# ---------------------------------------------------------------------------
# OCR (same technique as src/ocr_scoreboard.py, callable per frame)
# ---------------------------------------------------------------------------

def _scale_roi(fw, fh):
    sx, sy = fw / BASE_W, fh / BASE_H
    rx1, ry1 = int(ROI_X1 * sx), int(ROI_Y1 * sy)
    rx2, ry2 = int(ROI_X2 * sx), int(ROI_Y2 * sy)
    return rx1, ry1, rx2, ry2, sx, sy


def _ocr_scores(frame):
    """Extract the four player scores from one frame. Returns list[int|None]."""
    fh, fw = frame.shape[:2]
    rx1, ry1, rx2, ry2, sx, sy = _scale_roi(fw, fh)

    rx1, rx2 = max(0, rx1), min(fw, rx2)
    ry1, ry2 = max(0, ry1), min(fh, ry2)
    if rx2 - rx1 < 10 or ry2 - ry1 < 10:
        return [None, None, None, None]

    board = frame[ry1:ry2, rx1:rx2]

    s_x1 = int(SCORE_X1 * sx)
    s_y1 = int(SCORE_Y1 * sy)
    s_x2 = int(SCORE_X2 * sx)
    s_y2 = int(SCORE_Y2 * sy)
    s_x2 = min(s_x2, board.shape[1])
    s_y2 = min(s_y2, board.shape[0])
    if s_x2 - s_x1 < 5 or s_y2 - s_y1 < 5:
        return [None, None, None, None]

    score_row = board[s_y1:s_y2, s_x1:s_x2]

    reader = get_reader()

    def read(img):
        results = reader.readtext(img, detail=1, paragraph=False,
                                  allowlist="0123456789")
        cands = []
        for box, text, conf in results:
            digits = "".join(ch for ch in text if ch.isdigit())
            if not digits:
                continue
            xs = [int(pt[0]) for pt in box]
            cands.append((min(xs), int(digits), float(conf)))
        cands.sort(key=lambda c: c[0])
        return cands

    cands = read(score_row)
    if len(cands) < 4:
        gray = cv2.cvtColor(score_row, cv2.COLOR_BGR2GRAY)
        otsu = cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cands = read(otsu)

    if len(cands) > 4:
        cands = sorted(cands, key=lambda c: c[2], reverse=True)[:4]
        cands.sort(key=lambda c: c[0])

    if len(cands) != 4:
        return [None, None, None, None]

    return [cands[0][1], cands[1][1], cands[2][1], cands[3][1]]


def _validate_temporally(current, previous):
    """Reject the common 'multi-digit truncated to first digit' OCR failure."""
    if current is None:
        return previous
    if previous is None:
        return current
    out = list(current)
    for i in range(4):
        p, c = previous[i], current[i]
        if p is None or c is None:
            continue
        if c == p:
            continue
        if str(p).startswith(str(c)):
            out[i] = p
    return out


# ---------------------------------------------------------------------------
# Change detection + reliability
# ---------------------------------------------------------------------------

def _detect_changes(values, frames):
    """
    Persistence-based change detection (same algorithm as
    src/change_detector.py): a change is only registered after the new
    value persists for PERSISTENCE_FRAMES consecutive frames, which
    filters single-frame OCR blips.

    Returns list of raw change dicts.
    """
    events = []
    baseline = None
    candidate = None
    candidate_count = 0

    for value, frame in zip(values, frames):
        if value is None:
            continue
        if baseline is None:
            baseline = value
            continue
        if value == baseline:
            candidate = None
            candidate_count = 0
            continue
        if candidate != value:
            candidate = value
            candidate_count = 1
        else:
            candidate_count += 1
        if candidate_count >= PERSISTENCE_FRAMES:
            events.append({
                "old_score": baseline,
                "new_score": value,
                "event_frame": frame,
            })
            baseline = value
            candidate = None
            candidate_count = 0
    return events


def _analyze_events(series):
    """
    Returns (events, per_player_final, confirmed, rejected).

    Uses persistence-based change detection, then applies the reliability
    rule from final_report.py: a change is REJECTED when the score later
    returns to its previous value, otherwise CONFIRMED as a stable score.
    """
    events = []

    for player in PLAYERS:
        values = [row[player] for row in series]
        frames = [row["frame"] for row in series]

        raw = _detect_changes(values, frames)

        for i, ev in enumerate(raw):
            reverted = any(
                later["new_score"] == ev["old_score"]
                for later in raw[i + 1:]
            )
            if reverted:
                status, reliability, reason = "REJECTED", 10.0, \
                    "score_returned_to_previous_value"
            else:
                status, reliability, reason = "CONFIRMED", 90.0, \
                    "stable_new_score"

            events.append({
                "player": player,
                "old_score": ev["old_score"],
                "new_score": ev["new_score"],
                "delta": ev["new_score"] - ev["old_score"],
                "event_frame": ev["event_frame"],
                "reliability": reliability,
                "status": status,
                "reason": reason,
            })

    per_player_final = {}
    for player in PLAYERS:
        vals = [row[player] for row in series if row[player] is not None]
        per_player_final[player] = vals[-1] if vals else None

    confirmed = [e for e in events if e["status"] == "CONFIRMED"]
    rejected = [e for e in events if e["status"] == "REJECTED"]
    return events, per_player_final, confirmed, rejected


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run_pipeline(input_path, kind, job_dir, progress=None, demo=False):
    """
    Execute the full VisionScore pipeline.

    input_path : path to uploaded file (ignored when demo=True)
    kind       : "video" | "image"
    job_dir    : directory to write annotated frames + crops
    progress   : optional callable(dict) for live status updates
    """
    def emit(**kw):
        if progress:
            progress(kw)

    t0 = time.time()
    job_dir = Path(job_dir)
    frames_dir = job_dir / "frames"
    crops_dir = job_dir / "crops"
    frames_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    emit(stage="load", message="Reading input and sampling frames...",
         percent=3)

    if demo:
        frames, fps = _load_demo_frames()
    else:
        frames, fps = _load_frames(input_path, kind)

    if not frames:
        raise RuntimeError("No frames could be read from the input.")

    total = len(frames)
    emit(stage="load",
         message=f"Loaded {total} frame(s) for analysis.",
         percent=6, total_frames=total)

    ocr_available = True
    try:
        get_reader()
    except Exception:
        ocr_available = False

    previous_box = None
    hold_frames = 0
    direct_count = 0
    hold_count = 0
    lost_count = 0

    previous_scores = None
    series = []
    frame_records = []

    for i, (label, frame) in enumerate(frames, start=1):
        fh, fw = frame.shape[:2]

        result = detect_scoreboard(frame, previous_box)
        box = None
        confidence = 0.0
        status = "LOST"

        if result is not None:
            x, y, w, h, confidence = result
            if confidence >= DIRECT_CONFIDENCE:
                box = (x, y, w, h)
                status = "DIRECT"
                previous_box = box
                hold_frames = 0
                direct_count += 1

        if box is None and previous_box is not None and hold_frames < MAX_HOLD_FRAMES:
            box = previous_box
            hold_frames += 1
            status = "TRACKED"
            hold_count += 1
        elif box is None:
            status = "LOST"
            lost_count += 1

        annotated = frame.copy()
        crop_name = ""
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cv2.putText(annotated, f"{status} conf={confidence:.2f}",
                        (x, max(y - 12, 28)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
            crop = frame[max(0, y):y + h, max(0, x):x + w]
            if crop.size > 0:
                crop_name = f"{label}_crop.jpg"
                cv2.imwrite(str(crops_dir / crop_name),
                            _resize_max(crop, 640))

        frame_name = f"{label}.jpg"
        cv2.imwrite(str(frames_dir / frame_name), _resize_max(annotated, 900))

        # OCR extraction
        raw_scores = None
        if ocr_available:
            try:
                raw_scores = _ocr_scores(frame)
                if all(s is None for s in raw_scores):
                    raw_scores = None
            except Exception:
                raw_scores = None

        validated = _validate_temporally(raw_scores, previous_scores)
        if validated is None:
            validated = previous_scores if previous_scores else [None] * 4
        if raw_scores is not None:
            previous_scores = validated

        series.append({
            "frame": label,
            "player_1": validated[0] if validated else None,
            "player_2": validated[1] if validated else None,
            "player_3": validated[2] if validated else None,
            "player_4": validated[3] if validated else None,
        })

        rec = {
            "label": label,
            "frame_url": f"frames/{frame_name}",
            "crop_url": f"crops/{crop_name}" if crop_name else "",
            "status": status,
            "confidence": round(float(confidence), 3),
            "box": list(box) if box else None,
            "scores": validated if validated else [None] * 4,
        }
        frame_records.append(rec)

        percent = 6 + int((i / total) * 84)
        emit(stage="process",
             message=f"Frame {i}/{total}  -  localization: {status}"
                     + (f"  scores: {validated}" if validated and any(
                         v is not None for v in validated) else ""),
             percent=percent,
             current_frame=i,
             total_frames=total,
             preview=rec["frame_url"],
             scores=rec["scores"])

    emit(stage="analyze",
         message="Running change detection + reliability analysis...",
         percent=92)

    events, per_player_final, confirmed, rejected = _analyze_events(series)

    coverage = (direct_count + hold_count) / total * 100 if total else 0.0
    total_increase = sum(e["delta"] for e in confirmed)

    metrics = {
        "frames_processed": total,
        "localization_coverage": round(coverage, 2),
        "direct_detections": direct_count,
        "tracking_holds": hold_count,
        "lost_frames": lost_count,
        "ocr_available": ocr_available,
        "events_detected": len(events),
        "events_confirmed": len(confirmed),
        "events_rejected": len(rejected),
        "average_confirmed_reliability": round(
            sum(e["reliability"] for e in confirmed) / len(confirmed), 1
        ) if confirmed else 0.0,
        "total_confirmed_increase": total_increase,
        "elapsed_seconds": round(time.time() - t0, 2),
        "fps": round(fps, 2) if fps else 0.0,
    }

    result = {
        "metrics": metrics,
        "frames": frame_records,
        "series": series,
        "events": events,
        "confirmed_events": confirmed,
        "rejected_events": rejected,
        "final_scores": per_player_final,
        "players": PLAYERS,
    }

    emit(stage="done", message="Analysis complete.", percent=100)
    return result


def _resize_max(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)
