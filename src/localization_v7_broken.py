import cv2
import csv
from pathlib import Path


INPUT_DIR = Path("screenshots/training_frames")
# V7.1 recovery settings
RECOVERY_ENABLED = True
RECOVERY_RADIUS = 120
RECOVERY_STEP = 20

OUTPUT_DIR = Path("screenshots/auto_detection_v6")
REPORT_DIR = Path("outputs")

CSV_PATH = REPORT_DIR / "localization_v6.csv"


# =========================================================
# Configuration
# =========================================================

MAX_HOLD_FRAMES = 2

DIRECT_CONFIDENCE = 0.45

MIN_WIDTH_RATIO = 0.55
MIN_HEIGHT_RATIO = 0.50

MAX_WIDTH_RATIO = 0.99
MAX_HEIGHT_RATIO = 0.99


# =========================================================
# Expected geometry
#
# These are SOFT preferences.
# They must NOT be used to reject legitimate movement.
# =========================================================

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
    # Hard geometry rejection
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
    # Soft geometry scores
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
        1.0 - position_distance * 4.0
    )

    size_score = max(
        0.0,
        1.0 - size_distance * 2.5
    )

    expected_aspect = EXPECTED_W / EXPECTED_H

    aspect_distance = abs(
        aspect_ratio - expected_aspect
    )

    aspect_score = max(
        0.0,
        1.0 - aspect_distance * 0.4
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
    # Final score
    # -----------------------------------------------------

    confidence = (
        position_score * 0.25
        + size_score * 0.25
        + aspect_score * 0.20
        + area_score * 0.30
    )

    return max(
        0.0,
        min(1.0, confidence)
    )


# =========================================================
# Detect scoreboard
# =========================================================

def detect_scoreboard(
    frame,
    previous_box=None
):

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

            x, y, w, h = cv2.boundingRect(
                contour
            )

            if w <= 0 or h <= 0:
                continue

            rectangle_area = w * h

            contour_area = cv2.contourArea(
                contour
            )

            if rectangle_area <= 0:
                continue

            rectangularity = (
                contour_area /
                rectangle_area
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

            # -------------------------------------------------
            # Temporal similarity is a BONUS.
            #
            # It is NOT a rejection condition.
            # -------------------------------------------------

            temporal_bonus = 0.0

            if previous_box is not None:

                overlap = calculate_iou(
                    (x, y, w, h),
                    previous_box
                )

                temporal_bonus = (
                    overlap * 0.15
                )

            confidence += (
                min(rectangularity, 1.0)
                * 0.10
            )

            confidence += temporal_bonus

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
# Main
# =========================================================

def main():

    print("=" * 60)
    print("VisionScore - Temporal Localization V6")
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
    hold_frames = 0

    direct_count = 0
    hold_count = 0
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

        result = detect_scoreboard(
            frame,
            previous_box
        )

        output = frame.copy()

        box = None
        confidence = 0.0
        status = "LOST"

        # =================================================
        # DIRECT DETECTION
        # =================================================

        if result is not None:

            x, y, w, h, confidence = result

            if confidence >= DIRECT_CONFIDENCE:

                box = (
                    x,
                    y,
                    w,
                    h
                )

                status = "DIRECT"

                previous_box = box

                hold_frames = 0

                direct_count += 1

        # =================================================
        # V7.1 RECOVERY SEARCH
        # =================================================

        if box is None and previous_box is not None and RECOVERY_ENABLED:

            recovered = recover_scoreboard(
                frame,
                previous_box
            )

            if recovered is not None:

                box = recovered
                previous_box = box
                hold_frames = 0
                status = "RECOVERED"
                confidence = 0.65

        # =================================================
        # TEMPORAL HOLD
        # =================================================

        if box is None:

            if (
                previous_box is not None
                and hold_frames < MAX_HOLD_FRAMES
            ):

                box = previous_box

                hold_frames += 1

                status = (
                    f"TRACKED-HOLD ({hold_frames})"
                )

                hold_count += 1

            else:

                status = "LOST"

                lost_count += 1

        # =================================================
        # Draw
        # =================================================

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
                f"V6 {status} "
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

        # =================================================
        # Save annotated frame
        # =================================================

        output_path = (
            OUTPUT_DIR /
            image_path.name
        )

        cv2.imwrite(
            str(output_path),
            output
        )

        # =================================================
        # CSV
        # =================================================

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
    # Save CSV
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

    total = len(image_paths)

    successful = (
        direct_count
        + hold_count
    )

    coverage = (
        successful / total * 100
        if total
        else 0
    )

    print()

    print("=" * 60)
    print("TEMPORAL LOCALIZATION V6 SUMMARY")
    print("=" * 60)

    print(
        f"Direct detections: {direct_count}"
    )

    print(
        f"Tracking holds:    {hold_count}"
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

# =========================================================
# ISOLATED LOST-FRAME REPAIR
# =========================================================

def repair_isolated_lost_frames(csv_path):
    """
    Repair a single LOST frame when it is surrounded by
    valid localized frames.

    The repair uses linear interpolation between the
    previous and next known bounding boxes.

    This is intentionally conservative:
    - only repairs one-frame gaps
    - requires valid boxes on both sides
    - never modifies original training images
    """

    import csv

    rows = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    repaired = 0

    for i in range(1, len(rows) - 1):

        current = rows[i]

        if current["status"] != "LOST":
            continue

        previous = rows[i - 1]
        following = rows[i + 1]

        # Require valid boxes immediately before and after.
        if not (
            previous["x"]
            and previous["y"]
            and previous["width"]
            and previous["height"]
        ):
            continue

        if not (
            following["x"]
            and following["y"]
            and following["width"]
            and following["height"]
        ):
            continue

        # Convert coordinates.
        px = int(previous["x"])
        py = int(previous["y"])
        pw = int(previous["width"])
        ph = int(previous["height"])

        nx = int(following["x"])
        ny = int(following["y"])
        nw = int(following["width"])
        nh = int(following["height"])

        # Linear interpolation for the middle frame.
        x = round((px + nx) / 2)
        y = round((py + ny) / 2)
        w = round((pw + nw) / 2)
        h = round((ph + nh) / 2)

        current["status"] = "TEMPORAL-REPAIR"
        current["confidence"] = "0.50"
        current["x"] = str(x)
        current["y"] = str(y)
        current["width"] = str(w)
        current["height"] = str(h)

        repaired += 1

    # Rewrite CSV.
    if rows:

        fieldnames = [
            "frame",
            "status",
            "confidence",
            "x",
            "y",
            "width",
            "height"
        ]

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(rows)

    print()
    print("=" * 60)
    print("ISOLATED TEMPORAL REPAIR")
    print("=" * 60)
    print(f"Frames repaired: {repaired}")
    print("=" * 60)

if __name__ == "__main__":
    main()