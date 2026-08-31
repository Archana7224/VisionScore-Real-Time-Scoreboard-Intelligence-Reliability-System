import cv2
from pathlib import Path


INPUT_DIR = Path("screenshots/video_analysis")
OUTPUT_DIR = Path("screenshots/detection_v2")


def detect_scoreboard(frame):
    """
    Detect a likely scoreboard region using
    multiple visual clues.

    This is a baseline detector.
    It does not use OCR or deep learning.
    """

    if frame is None:
        return None

    height, width = frame.shape[:2]

    # ---------------------------------------------------------
    # 1. Convert to grayscale
    # ---------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # ---------------------------------------------------------
    # 2. Slight blur to reduce noise
    # ---------------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # ---------------------------------------------------------
    # 3. Detect edges
    # ---------------------------------------------------------

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    # ---------------------------------------------------------
    # 4. Close small gaps in table borders
    # ---------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7)
    )

    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ---------------------------------------------------------
    # 5. Find contours
    # ---------------------------------------------------------

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    frame_area = width * height

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        area = w * h

        if h == 0:
            continue

        aspect_ratio = w / h

        rectangularity = (
            cv2.contourArea(contour) / area
        )

        area_ratio = area / frame_area

        # -----------------------------------------------------
        # Candidate filtering
        # -----------------------------------------------------

        # Too small
        if area_ratio < 0.10:
            continue

        # Almost entire frame
        if area_ratio > 0.90:
            continue

        # Scoreboard should generally be wider than tall
        if aspect_ratio < 1.5:
            continue

        # Avoid extremely thin regions
        if h < height * 0.15:
            continue

        # Avoid extremely narrow regions
        if w < width * 0.25:
            continue

        # A scoreboard should have reasonable rectangularity
        if rectangularity < 0.25:
            continue

        # -----------------------------------------------------
        # Candidate scoring
        # -----------------------------------------------------

        score = 0

        # Prefer large regions
        score += area_ratio * 40

        # Prefer wide regions
        score += min(aspect_ratio / 4.0, 1.0) * 25

        # Prefer rectangular regions
        score += rectangularity * 25

        # Prefer regions that are not touching
        # too many frame boundaries
        if x > width * 0.02:
            score += 5

        if y > height * 0.02:
            score += 5

        candidates.append(
            (
                score,
                x,
                y,
                w,
                h
            )
        )

    if not candidates:
        return None

    # Highest scoring candidate
    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    _, x, y, w, h = candidates[0]

    return x, y, w, h


def process_image(image_path):

    frame = cv2.imread(
        str(image_path)
    )

    if frame is None:
        print(
            f"ERROR: Could not read "
            f"{image_path}"
        )
        return

    rectangle = detect_scoreboard(frame)

    if rectangle is None:

        print(
            f"NO SCOREBOARD DETECTED: "
            f"{image_path.name}"
        )

        return

    x, y, w, h = rectangle

    annotated = frame.copy()

    # Draw detection rectangle
    cv2.rectangle(
        annotated,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        3
    )

    # Add label
    cv2.putText(
        annotated,
        "Scoreboard Candidate",
        (x, max(30, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR /
        image_path.name
    )

    cv2.imwrite(
        str(output_path),
        annotated
    )

    print(
        f"{image_path.name}: "
        f"x={x}, "
        f"y={y}, "
        f"width={w}, "
        f"height={h}"
    )

    print(
        f"Saved: {output_path}"
    )


def main():

    print("=" * 60)
    print(
        "VisionScore - Scoreboard Detection V2"
    )
    print("=" * 60)

    if not INPUT_DIR.exists():

        raise FileNotFoundError(
            f"Input directory not found: "
            f"{INPUT_DIR}"
        )

    images = sorted(
        INPUT_DIR.glob("*.jpg")
    )

    if not images:

        raise FileNotFoundError(
            f"No JPG images found in "
            f"{INPUT_DIR}"
        )

    print(
        f"Found {len(images)} images."
    )

    print()

    for image_path in images:

        process_image(
            image_path
        )

        print()

    print("=" * 60)
    print(
        "Detection V2 complete."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()