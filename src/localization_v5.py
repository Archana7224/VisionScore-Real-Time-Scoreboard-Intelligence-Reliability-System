import cv2
import csv
from pathlib import Path


INPUT_DIR = Path("screenshots/training_frames")
OUTPUT_DIR = Path("screenshots/auto_detection_v5")
REPORT_DIR = Path("outputs")

CSV_PATH = REPORT_DIR / "localization_v5.csv"


# =========================================================
# Configuration
# =========================================================

MAX_HOLD_FRAMES = 2

MIN_WIDTH_RATIO = 0.55
MIN_HEIGHT_RATIO = 0.50

MAX_WIDTH_RATIO = 0.98
MAX_HEIGHT_RATIO = 0.98

MIN_CONFIDENCE = 0.55

EXPECTED_X = 0.07
EXPECTED_Y = 0.07
EXPECTED_W = 0.87
EXPECTED_H = 0.80


# =========================================================
# IoU
# =========================================================

def calculate_iou(box_a, box_b):

    if box_a is None or box_b is None:
        return 0.0

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2 = ax + aw
    ay2 = ay + ah

    bx2 = bx + bw
    by2 = by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = aw * ah
    area_b = bw * bh

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# =========================================================
# Candidate confidence
# =========================================================

def candidate_confidence(
    x,
    y,
    w,
    h,
    frame_width,
    frame_height
):

    nx = x / frame_width
    ny = y / frame_height
    nw = w / frame_width
    nh = h / frame_height

    aspect_ratio = w / max(h, 1)

    # -----------------------------------------------------
    # Hard rejection
    # -----------------------------------------------------

    if nw < MIN_WIDTH_RATIO:
        return 0.0

    if nh < MIN_HEIGHT_RATIO:
        return 0.0

    if nw > MAX_WIDTH_RATIO and nh > MAX_HEIGHT_RATIO:
        return 0.0

    if aspect_ratio < 1.5 or aspect_ratio > 3.0:
        return 0.0

    # -----------------------------------------------------
    # Geometry similarity
    # -----------------------------------------------------

    position_distance = (
        abs(nx - EXPECTED_X)
        + abs(ny - EXPECTED_Y)
    )

    size_distance = (
        abs(nw - EXPECTED_W)
        + abs(nh - EXPECTED_H)
    )

    position_score = max(
        0.0,
        1.0 - position_distance * 5.0
    )

    size_score = max(
        0.0,
        1.0 - size_distance * 3.0
    )

    expected_aspect = EXPECTED_W / EXPECTED_H

    aspect_distance = abs(
        aspect_ratio - expected_aspect
    )

    aspect_score = max(
        0.0,
        1.0 - aspect_distance * 0.5
    )

    # -----------------------------------------------------
    # Area
    # -----------------------------------------------------

    area_ratio = nw * nh

    area_score = min(
        area_ratio / 0.70,
        1.0
    )

    # -----------------------------------------------------
    # Final confidence
    # -----------------------------------------------------

    confidence = (
        position_score * 0.30
        + size_score * 0.30
        + aspect_score * 0.20
        + area_score * 0.20
    )

    return max(0.0, min(1.0, confidence))


# =========================================================
# Detect candidates
# =========================================================

def detect_scoreboard(frame):

    if frame is None or frame.size == 0:
        return None

    frame_height, frame_width = frame.shape[:2]

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    threshold_pairs = [
        (30, 100),
        (50, 150),
        (70, 180)
    ]

    candidates = []

    for low, high in threshold_pairs:

        edges = cv2.Canny(
            blurred,
            low,
            high
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (9, 9)
        )

        connected = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel
        )

        connected = cv2.dilate(
            connected,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (5, 5)
            ),
            iterations=1
        )

        contours, _ = cv2.findContours(
            connected,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            x, y, w, h = cv2.boundingRect(contour)

            if w <= 0 or h <= 0:
                continue

            rectangle_area = w * h

            contour_area = cv2.contourArea(contour)

            if rectangle_area <= 0:
                continue

            rectangularity = (
                contour_area / rectangle_area
            )

            if rectangularity < 0.20:
                continue

            confidence = candidate_confidence(
                x,
                y,
                w,
                h,
                frame_width,
                frame_height
            )

            if confidence <= 0:
                continue

            # Reward rectangular contours.
            confidence += (
                min(rectangularity, 1.0)
                * 0.10
            )

            confidence = min(
                confidence,
                1.0
            )

            candidates.append(
                (
                    confidence,
                    x,
                    y,
                    w,
                    h
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    confidence, x, y, w, h = candidates[0]

    return (
        x,
        y,
        w,
        h,
        confidence
    )


# =========================================================
# Temporal validation
# =========================================================

def validate_temporal(
    current_box,
    previous_box
):

    if current_box is None:
        return False

    if previous_box is None:
        return True

    current = current_box[:4]

    overlap = calculate_iou(
        current,
        previous_box
    )

    return overlap >= 0.50


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 60)
    print("VisionScore - Confidence-Aware Localization V5")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
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

    previous_box = None
    hold_count = 0

    direct_count = 0
    tracked_count = 0
    hold_count_total = 0
    rejected_count = 0
    lost_count = 0

    rows = []

    for image_path in image_paths:

        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:

            print(
                f"{image_path.name}: "
                f"READ ERROR"
            )

            continue

        result = detect_scoreboard(frame)

        output = frame.copy()

        status = "LOST"
        confidence = 0.0
        box = None

        # -------------------------------------------------
        # Direct detection
        # -------------------------------------------------

        if result is not None:

            x, y, w, h, confidence = result

            candidate_box = (
                x,
                y,
                w,
                h
            )

            temporal_ok = validate_temporal(
                result,
                previous_box
            )

            if confidence >= MIN_CONFIDENCE:

                if (
                    previous_box is None
                    or temporal_ok
                ):

                    box = candidate_box

                    status = "DIRECT"

                    previous_box = box

                    hold_count = 0

                    direct_count += 1

                else:

                    # Candidate exists but jumps
                    # unexpectedly.
                    if (
                        previous_box is not None
                        and hold_count < MAX_HOLD_FRAMES
                    ):

                        box = previous_box

                        hold_count += 1

                        status = (
                            f"TRACKED-HOLD ({hold_count})"
                        )

                        hold_count_total += 1

                    else:

                        status = "REJECTED"

                        rejected_count += 1

            else:

                result = None

        # -------------------------------------------------
        # No reliable detection
        # -------------------------------------------------

        if box is None:

            if (
                result is None
                and previous_box is not None
                and hold_count < MAX_HOLD_FRAMES
            ):

                box = previous_box

                hold_count += 1

                status = (
                    f"TRACKED-HOLD ({hold_count})"
                )

                hold_count_total += 1

            elif status != "REJECTED":

                status = "LOST"

                lost_count += 1

        # -------------------------------------------------
        # Draw result
        # -------------------------------------------------

        if box is not None:

            x, y, w, h = box

            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                4
            )

            label = (
                f"V5 {status} "
                f"conf={confidence:.2f}"
            )

            cv2.putText(
                output,
                label,
                (x, max(y - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            print(
                f"{image_path.name}: "
                f"x={x}, y={y}, "
                f"width={w}, height={h} "
                f"[{status} conf={confidence:.2f}]"
            )

        else:

            print(
                f"{image_path.name}: "
                f"NO SCOREBOARD [{status}]"
            )

        # -------------------------------------------------
        # Save image
        # -------------------------------------------------

        output_path = (
            OUTPUT_DIR /
            image_path.name
        )

        cv2.imwrite(
            str(output_path),
            output
        )

        # -------------------------------------------------
        # CSV row
        # -------------------------------------------------

        rows.append(
            {
                "frame": image_path.name,
                "status": status,
                "confidence": round(
                    confidence,
                    4
                ),
                "x": box[0] if box else "",
                "y": box[1] if box else "",
                "width": box[2] if box else "",
                "height": box[3] if box else ""
            }
        )

    # =====================================================
    # CSV
    # =====================================================

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "frame",
            "status",
            "confidence",
            "x",
            "y",
            "width",
            "height"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)

    # =====================================================
    # Summary
    # =====================================================

    successful = (
        direct_count
        + tracked_count
        + hold_count_total
    )

    total = len(image_paths)

    coverage = (
        successful / total * 100
        if total > 0
        else 0
    )

    print()

    print("=" * 60)
    print("LOCALIZATION V5 SUMMARY")
    print("=" * 60)

    print(
        f"Direct detections: {direct_count}"
    )

    print(
        f"Tracking holds:    {hold_count_total}"
    )

    print(
        f"Rejected:          {rejected_count}"
    )

    print(
        f"Lost frames:       {lost_count}"
    )

    print(
        f"Successful localization: "
        f"{successful}/{total}"
    )

    print(
        f"Coverage: {coverage:.2f}%"
    )

    print()

    print(
        f"CSV report: {CSV_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()