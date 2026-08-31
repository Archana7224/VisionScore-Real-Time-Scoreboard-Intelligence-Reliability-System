import cv2
from pathlib import Path

from auto_localize import detect_scoreboard


INPUT_DIR = Path("screenshots/training_frames")
OUTPUT_DIR = Path("screenshots/auto_detection_v4")


# =========================================================
# Configuration
# =========================================================

MAX_HOLD_FRAMES = 2

# Maximum allowed movement between consecutive frames.
MAX_CENTER_SHIFT = 0.12

# Maximum allowed relative size change.
MAX_SIZE_CHANGE = 0.20

# Minimum confidence required to accept a direct detection.
MIN_CONFIDENCE = 0.45

# Strong confidence threshold.
HIGH_CONFIDENCE = 0.70


# =========================================================
# Utility functions
# =========================================================

def box_to_corners(box):
    x, y, w, h = box

    return (
        x,
        y,
        x + w,
        y + h
    )


def calculate_iou(box_a, box_b):

    if box_a is None or box_b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = box_to_corners(box_a)
    bx1, by1, bx2, by2 = box_to_corners(box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = (
        (ax2 - ax1) *
        (ay2 - ay1)
    )

    area_b = (
        (bx2 - bx1) *
        (by2 - by1)
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def box_center(box):

    x, y, w, h = box

    return (
        x + w / 2,
        y + h / 2
    )


def center_shift(box_a, box_b, frame_width, frame_height):

    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)

    dx = abs(ax - bx) / frame_width
    dy = abs(ay - by) / frame_height

    return max(dx, dy)


def size_change(box_a, box_b):

    _, _, aw, ah = box_a
    _, _, bw, bh = box_b

    width_change = abs(bw - aw) / max(aw, 1)
    height_change = abs(bh - ah) / max(ah, 1)

    return max(
        width_change,
        height_change
    )


# =========================================================
# Direct detection confidence
# =========================================================

def calculate_confidence(
    current_box,
    previous_box,
    frame_width,
    frame_height
):

    if current_box is None:
        return 0.0

    # -----------------------------------------------------
    # If there is no previous box, use geometry only.
    # -----------------------------------------------------

    if previous_box is None:

        x, y, w, h = current_box

        area_ratio = (
            (w * h) /
            (frame_width * frame_height)
        )

        # Expected scoreboard occupies a large portion
        # of the video.
        area_score = min(
            area_ratio / 0.70,
            1.0
        )

        return area_score

    # -----------------------------------------------------
    # Temporal consistency
    # -----------------------------------------------------

    iou = calculate_iou(
        current_box,
        previous_box
    )

    shift = center_shift(
        current_box,
        previous_box,
        frame_width,
        frame_height
    )

    change = size_change(
        current_box,
        previous_box
    )

    # Convert movement into a score.
    movement_score = max(
        0.0,
        1.0 -
        shift / MAX_CENTER_SHIFT
    )

    size_score = max(
        0.0,
        1.0 -
        change / MAX_SIZE_CHANGE
    )

    # Weighted confidence.
    confidence = (
        iou * 0.50
        + movement_score * 0.30
        + size_score * 0.20
    )

    return confidence


# =========================================================
# Validate detection
# =========================================================

def validate_detection(
    current_box,
    previous_box,
    frame_width,
    frame_height
):

    if current_box is None:
        return False, 0.0

    confidence = calculate_confidence(
        current_box,
        previous_box,
        frame_width,
        frame_height
    )

    # First detection has no temporal reference.
    if previous_box is None:
        return True, confidence

    shift = center_shift(
        current_box,
        previous_box,
        frame_width,
        frame_height
    )

    change = size_change(
        current_box,
        previous_box
    )

    iou = calculate_iou(
        current_box,
        previous_box
    )

    # -----------------------------------------------------
    # Reject sudden impossible movement.
    # -----------------------------------------------------

    if shift > MAX_CENTER_SHIFT:
        return False, confidence

    # -----------------------------------------------------
    # Reject sudden size change.
    # -----------------------------------------------------

    if change > MAX_SIZE_CHANGE:
        return False, confidence

    # -----------------------------------------------------
    # Very low temporal overlap is suspicious.
    # -----------------------------------------------------

    if (
        iou < 0.20
        and confidence < MIN_CONFIDENCE
    ):
        return False, confidence

    return True, confidence


# =========================================================
# Draw detection
# =========================================================

def draw_box(
    image,
    box,
    label,
    thickness=4
):

    if box is None:
        return

    x, y, w, h = box

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        thickness
    )

    cv2.putText(
        image,
        label,
        (x, max(y - 12, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 60)
    print("VisionScore - Temporal Tracking V4")
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

    previous_box = None
    hold_count = 0

    direct_count = 0
    tracked_count = 0
    hold_count_total = 0
    lost_count = 0

    for image_path in image_paths:

        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:

            print(
                f"{image_path.name}: "
                "ERROR READING IMAGE"
            )

            continue

        frame_height, frame_width = (
            frame.shape[:2]
        )

        output = frame.copy()

        # -------------------------------------------------
        # Step 1: Run direct detector.
        # -------------------------------------------------

        detected_box = detect_scoreboard(
            frame
        )

        # -------------------------------------------------
        # Step 2: Validate direct detection.
        # -------------------------------------------------

        accepted = False
        confidence = 0.0

        if detected_box is not None:

            accepted, confidence = validate_detection(
                detected_box,
                previous_box,
                frame_width,
                frame_height
            )

        # -------------------------------------------------
        # Case A: Good direct detection.
        # -------------------------------------------------

        if accepted:

            previous_box = detected_box
            hold_count = 0

            direct_count += 1

            label = (
                f"DIRECT "
                f"conf={confidence:.2f}"
            )

            draw_box(
                output,
                detected_box,
                label
            )

            print(
                f"{image_path.name}: "
                f"x={detected_box[0]}, "
                f"y={detected_box[1]}, "
                f"width={detected_box[2]}, "
                f"height={detected_box[3]} "
                f"[DIRECT conf={confidence:.2f}]"
            )

        # -------------------------------------------------
        # Case B: Detection exists but is suspicious.
        # Try temporal hold.
        # -------------------------------------------------

        elif previous_box is not None:

            if hold_count < MAX_HOLD_FRAMES:

                hold_count += 1
                hold_count_total += 1

                draw_box(
                    output,
                    previous_box,
                    f"TRACKED-HOLD ({hold_count})"
                )

                print(
                    f"{image_path.name}: "
                    f"x={previous_box[0]}, "
                    f"y={previous_box[1]}, "
                    f"width={previous_box[2]}, "
                    f"height={previous_box[3]} "
                    f"[TRACKED-HOLD ({hold_count})]"
                )

            else:

                previous_box = None
                hold_count = 0

                lost_count += 1

                print(
                    f"{image_path.name}: "
                    f"NO SCOREBOARD "
                    f"[LOST]"
                )

        # -------------------------------------------------
        # Case C: No detection and no previous box.
        # -------------------------------------------------

        else:

            lost_count += 1

            print(
                f"{image_path.name}: "
                f"NO SCOREBOARD "
                f"[LOST]"
            )

        # -------------------------------------------------
        # Save visualization.
        # -------------------------------------------------

        output_path = (
            OUTPUT_DIR /
            image_path.name
        )

        cv2.imwrite(
            str(output_path),
            output
        )

    # =====================================================
    # Summary
    # =====================================================

    total = len(image_paths)

    successful = (
        direct_count
        + hold_count_total
    )

    coverage = (
        successful / total * 100
        if total > 0
        else 0
    )

    print()

    print("=" * 60)
    print("TEMPORAL TRACKING V4 SUMMARY")
    print("=" * 60)

    print(
        f"Direct detections: {direct_count}"
    )

    print(
        f"Tracking holds:    {hold_count_total}"
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

    print("=" * 60)


if __name__ == "__main__":
    main()