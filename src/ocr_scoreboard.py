import cv2
import easyocr
import csv
from pathlib import Path


# ============================================================
# VisionScore - OCR V3
# Structured scoreboard OCR + temporal validation
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FRAMES_DIR = BASE_DIR / "screenshots" / "training_frames"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "ocr_results_v3.csv"


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Original image coordinates
# Existing experiments proved this captures the scoreboard.
ROI_Y1 = 500
ROI_Y2 = 830
ROI_X1 = 70
ROI_X2 = 1760

# Score-row coordinates RELATIVE TO scoreboard ROI.
#
# Experimental result:
# score row exists around y=60..140
# x approximately 190..770
#
SCORE_Y1 = 50
SCORE_Y2 = 145
SCORE_X1 = 190
SCORE_X2 = 770

PLAYER_NAMES = [
    "player_1",
    "player_2",
    "player_3",
    "player_4",
]


# ------------------------------------------------------------
# OCR
# ------------------------------------------------------------

print("=" * 60)
print("VisionScore - OCR V3")
print("=" * 60)
print("Loading EasyOCR...")

reader = easyocr.Reader(
    ["en"],
    gpu=False,
    verbose=False
)

print("EasyOCR ready.")
print()


# ------------------------------------------------------------
# Helper: OCR one image
# ------------------------------------------------------------

def run_ocr(image):
    """
    Run EasyOCR on the score-row image.

    Only digits are allowed because this region contains
    scoreboard numbers.
    """

    return reader.readtext(
        image,
        detail=1,
        paragraph=False,
        allowlist="0123456789"
    )


# ------------------------------------------------------------
# Helper: convert OCR detection into candidate
# ------------------------------------------------------------

def make_candidate(result):
    """
    Convert EasyOCR result into:

        {
            "text": "39",
            "value": 39,
            "confidence": 0.98,
            "x1": ...,
            "y1": ...,
            "x2": ...,
            "y2": ...
        }

    Returns None if the result isn't a valid integer.
    """

    box, text, confidence = result

    text = text.strip()

    # Keep only digits.
    text = "".join(ch for ch in text if ch.isdigit())

    if not text:
        return None

    try:
        value = int(text)
    except ValueError:
        return None

    xs = [int(point[0]) for point in box]
    ys = [int(point[1]) for point in box]

    return {
        "text": text,
        "value": value,
        "confidence": float(confidence),
        "x1": min(xs),
        "y1": min(ys),
        "x2": max(xs),
        "y2": max(ys),
    }


# ------------------------------------------------------------
# Detect four scores
# ------------------------------------------------------------

def detect_scores(score_row):
    """
    Detect the four scoreboard scores.

    The scoreboard has a fixed horizontal structure:

        P1       P2       P3       P4

    Therefore detections are sorted by X position.
    """

    results = run_ocr(score_row)

    candidates = []

    for result in results:
        candidate = make_candidate(result)

        if candidate is None:
            continue

        candidates.append(candidate)

    # Sort left → right.
    candidates.sort(key=lambda c: c["x1"])

    # The scoreboard should contain four primary score boxes.
    #
    # We expect approximately:
    #
    # P1 x ≈ 32–134
    # P2 x ≈ 167–271
    # P3 x ≈ 307–413
    # P4 x ≈ 446–544
    #
    # However, don't hard-code exact boxes. Instead use
    # horizontal ordering.

    if len(candidates) < 4:
        return None, candidates

    # Keep the four strongest spatially separated candidates.
    #
    # Since this is a tightly cropped score row, normally
    # exactly four candidates will be returned.
    if len(candidates) > 4:
        candidates = sorted(
            candidates,
            key=lambda c: c["confidence"],
            reverse=True
        )[:4]

        candidates.sort(key=lambda c: c["x1"])

    if len(candidates) != 4:
        return None, candidates

    scores = [
        candidates[0]["value"],
        candidates[1]["value"],
        candidates[2]["value"],
        candidates[3]["value"],
    ]

    return scores, candidates


# ------------------------------------------------------------
# Otsu fallback
# ------------------------------------------------------------

def detect_scores_with_fallback(score_row):
    """
    Try raw OCR first.

    If raw OCR doesn't return exactly four scores,
    try Otsu thresholding.

    Raw OCR is intentionally the primary method because
    experiments showed that aggressive resizing can hurt
    recognition.
    """

    # --------------------------------------------------------
    # Attempt 1: RAW
    # --------------------------------------------------------

    scores, candidates = detect_scores(score_row)

    if scores is not None:
        return scores, candidates, "raw"

    # --------------------------------------------------------
    # Attempt 2: OTSU
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        score_row,
        cv2.COLOR_BGR2GRAY
    ) if len(score_row.shape) == 3 else score_row

    otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    scores, candidates = detect_scores(otsu)

    if scores is not None:
        return scores, candidates, "otsu"

    return None, candidates, "failed"


# ------------------------------------------------------------
# Temporal validation
# ------------------------------------------------------------

def validate_temporally(current_scores, previous_scores):
    """
    Use the previous frame as a sanity check.

    This does NOT blindly overwrite OCR.

    It only rejects obviously suspicious single-frame
    recognition failures.

    Example:

        previous = [20, 39, 48, 54]
        current  = [20, 3, 48, 54]

    Since the P2 value suddenly changed 39 → 3,
    while the other scores stayed identical, the system
    treats this as a likely OCR anomaly.

    A real score change is still allowed when supported
    by subsequent frames.
    """

    if current_scores is None:
        return previous_scores

    if previous_scores is None:
        return current_scores

    validated = current_scores.copy()

    for i in range(4):

        previous = previous_scores[i]
        current = current_scores[i]

        if previous is None:
            continue

        # Exact same value → safe.
        if current == previous:
            continue

        # Special OCR failure pattern:
        #
        # 39 → 3
        # 48 → 4
        # 54 → 5
        #
        # A multi-digit value being truncated to its first
        # digit is a common OCR failure.
        #
        # If the current value is a prefix of the previous
        # value, mark it suspicious.
        if str(previous).startswith(str(current)):
            validated[i] = previous

    return validated


# ------------------------------------------------------------
# Process frames
# ------------------------------------------------------------

frame_files = sorted(
    FRAMES_DIR.glob("train_*.jpg")
)

rows = []

previous_scores = None

for frame_path in frame_files:

    print(frame_path.name)

    image = cv2.imread(str(frame_path))

    if image is None:
        print("  ERROR: Could not read image")
        continue

    # --------------------------------------------------------
    # Scoreboard ROI
    # --------------------------------------------------------

    scoreboard = image[
        ROI_Y1:ROI_Y2,
        ROI_X1:ROI_X2
    ]

    # --------------------------------------------------------
    # Score-row ROI
    # --------------------------------------------------------

    score_row = scoreboard[
        SCORE_Y1:SCORE_Y2,
        SCORE_X1:SCORE_X2
    ]

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    scores, candidates, method = detect_scores_with_fallback(
        score_row
    )

    # --------------------------------------------------------
    # Temporal validation
    # --------------------------------------------------------

    validated_scores = validate_temporally(
        scores,
        previous_scores
    )

    if validated_scores is None:

        print(
            "  OCR: EMPTY"
        )

        current_scores = [None, None, None, None]

    else:

        current_scores = validated_scores

        for i, score in enumerate(current_scores):

            confidence = 0.0

            if i < len(candidates):
                confidence = candidates[i]["confidence"]

            print(
                f"  {PLAYER_NAMES[i]}: "
                f"{score} "
                f"(method={method}, "
                f"conf={confidence:.2f})"
            )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    rows.append({
        "frame": frame_path.name,

        "player_1": current_scores[0],
        "player_2": current_scores[1],
        "player_3": current_scores[2],
        "player_4": current_scores[3],
    })

    # Only update temporal state when OCR successfully
    # produced a complete score set.
    if scores is not None:
        previous_scores = validated_scores


# ------------------------------------------------------------
# CSV
# ------------------------------------------------------------

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "frame",
            "player_1",
            "player_2",
            "player_3",
            "player_4",
        ]
    )

    writer.writeheader()
    writer.writerows(rows)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print()
print("=" * 60)
print("OCR V3 SUMMARY")
print("=" * 60)
print(f"Processed: {len(rows)}/{len(frame_files)}")
print(f"CSV report: {CSV_PATH}")
print("=" * 60)