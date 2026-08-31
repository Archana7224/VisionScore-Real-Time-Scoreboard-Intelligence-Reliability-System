import cv2
from pathlib import Path


INPUT_DIR = Path("screenshots/training_frames")
OUTPUT_DIR = Path("screenshots/auto_detection_v2")


# ---------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------

def score_candidate(x, y, w, h, frame_width, frame_height):
    """
    Score a possible scoreboard rectangle.

    The scoring uses:
    - expected scoreboard size
    - expected position
    - aspect ratio
    - rectangularity
    - rejection of full-frame candidates
    """

    frame_area = frame_width * frame_height
    box_area = w * h

    if box_area <= 0:
        return -1

    area_ratio = box_area / frame_area

    aspect_ratio = w / max(h, 1)

    # -----------------------------------------------------
    # Hard rejection rules
    # -----------------------------------------------------

    # Reject tiny regions.
    if w < frame_width * 0.55:
        return -1

    if h < frame_height * 0.50:
        return -1

    # Reject extremely wide / narrow candidates.
    if aspect_ratio < 1.5 or aspect_ratio > 3.0:
        return -1

    # Reject almost entire-frame candidates.
    if (
        x <= 5
        and y <= 5
        and w >= frame_width * 0.97
        and h >= frame_height * 0.97
    ):
        return -1

    # -----------------------------------------------------
    # Expected normalized ranges learned from annotations
    # -----------------------------------------------------

    nx = x / frame_width
    ny = y / frame_height
    nw = w / frame_width
    nh = h / frame_height

    # Expected center.
    expected_x = 0.07
    expected_y = 0.07
    expected_w = 0.87
    expected_h = 0.80

    # Distance from expected geometry.
    position_distance = (
        abs(nx - expected_x) * 2.0
        + abs(ny - expected_y) * 2.0
    )

    size_distance = (
        abs(nw - expected_w) * 2.0
        + abs(nh - expected_h) * 2.0
    )

    # -----------------------------------------------------
    # Aspect ratio preference
    # -----------------------------------------------------

    expected_aspect = expected_w / expected_h

    aspect_distance = abs(
        aspect_ratio - expected_aspect
    )

    aspect_score = max(
        0.0,
        2.0 - aspect_distance
    )

    # -----------------------------------------------------
    # Position score
    # -----------------------------------------------------

    position_score = max(
        0.0,
        3.0 - position_distance * 10
    )

    # -----------------------------------------------------
    # Size score
    # -----------------------------------------------------

    size_score = max(
        0.0,
        3.0 - size_distance * 8
    )

    # -----------------------------------------------------
    # Area score
    # -----------------------------------------------------

    area_score = min(
        area_ratio * 3.0,
        3.0
    )

    # -----------------------------------------------------
    # Final score
    # -----------------------------------------------------

    score = (
        position_score
        + size_score
        + aspect_score
        + area_score
    )

    return score


# ---------------------------------------------------------
# Main detector
# ---------------------------------------------------------

def detect_scoreboard(frame):

    if frame is None or frame.size == 0:
        return None

    frame_height, frame_width = frame.shape[:2]

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Reduce noise.
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # -----------------------------------------------------
    # Try multiple edge thresholds
    # -----------------------------------------------------

    edge_results = []

    threshold_pairs = [
        (30, 100),
        (50, 150),
        (70, 180),
    ]

    for low, high in threshold_pairs:

        edges = cv2.Canny(
            blurred,
            low,
            high
        )

        # Smaller morphology than V1.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (9, 9)
        )

        connected = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel
        )

        # Slight dilation connects broken borders.
        connected = cv2.dilate(
            connected,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (5, 5)
            ),
            iterations=1
        )

        edge_results.append(connected)

    # -----------------------------------------------------
    # Generate candidates
    # -----------------------------------------------------

    candidates = []

    for connected in edge_results:

        contours, _ = cv2.findContours(
            connected,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            x, y, w, h = cv2.boundingRect(
                contour
            )

            if w <= 0 or h <= 0:
                continue

            contour_area = cv2.contourArea(
                contour
            )

            rectangle_area = w * h

            if rectangle_area <= 0:
                continue

            # -------------------------------------------------
            # Rectangularity
            # -------------------------------------------------

            rectangularity = (
                contour_area /
                rectangle_area
            )

            if rectangularity < 0.20:
                continue

            # -------------------------------------------------
            # Candidate score
            # -------------------------------------------------

            score = score_candidate(
                x,
                y,
                w,
                h,
                frame_width,
                frame_height
            )

            if score < 0:
                continue

            # Reward reasonably rectangular contours.
            score += rectangularity * 2.0

            candidates.append(
                (
                    score,
                    x,
                    y,
                    w,
                    h
                )
            )

    # -----------------------------------------------------
    # No candidate
    # -----------------------------------------------------

    if not candidates:
        return None

    # Highest score first.
    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    best = candidates[0]

    _, x, y, w, h = best

    return x, y, w, h


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("VisionScore - Automated Localization V2")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_paths = sorted(
        INPUT_DIR.glob("*.jpg")
    )

    print(
        f"Found {len(image_paths)} training images."
    )

    print()

    detected = 0

    for image_path in image_paths:

        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:

            print(
                f"ERROR: Could not read "
                f"{image_path.name}"
            )

            continue

        result = detect_scoreboard(
            frame
        )

        output = frame.copy()

        if result is None:

            print(
                f"{image_path.name}: "
                f"NO SCOREBOARD DETECTED"
            )

        else:

            x, y, w, h = result

            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                4
            )

            cv2.putText(
                output,
                "AUTO DETECTION V2",
                (x, max(y - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            print(
                f"{image_path.name}: "
                f"x={x}, y={y}, "
                f"width={w}, height={h}"
            )

            detected += 1

        output_path = (
            OUTPUT_DIR /
            image_path.name
        )

        cv2.imwrite(
            str(output_path),
            output
        )

    print()

    print("=" * 60)

    print(
        f"Detection complete: "
        f"{detected}/{len(image_paths)} images detected"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()