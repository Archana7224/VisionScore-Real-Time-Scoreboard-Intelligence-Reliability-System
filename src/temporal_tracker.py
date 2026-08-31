import cv2
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

INPUT_DIR = Path("screenshots/training_frames")
OUTPUT_DIR = Path("screenshots/temporal_detection_v3")


# ---------------------------------------------------------
# Tracking configuration
# ---------------------------------------------------------

# Maximum number of consecutive frames for which we are
# willing to keep using the previous box.
MAX_MISSED_FRAMES = 2

# Maximum normalized movement allowed between frames.
# Example: 0.08 means 8% of frame dimension.
MAX_X_MOVEMENT = 0.08
MAX_Y_MOVEMENT = 0.08

# Maximum relative size change allowed.
MAX_SIZE_CHANGE = 0.25


# ---------------------------------------------------------
# V2 detector
# ---------------------------------------------------------

def score_candidate(
    x,
    y,
    w,
    h,
    frame_width,
    frame_height
):
    """
    Score a possible scoreboard rectangle.

    This is the same V2 candidate scoring logic.
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

    if w < frame_width * 0.55:
        return -1

    if h < frame_height * 0.50:
        return -1

    if aspect_ratio < 1.5 or aspect_ratio > 3.0:
        return -1

    if (
        x <= 5
        and y <= 5
        and w >= frame_width * 0.97
        and h >= frame_height * 0.97
    ):
        return -1

    # -----------------------------------------------------
    # Normalized geometry
    # -----------------------------------------------------

    nx = x / frame_width
    ny = y / frame_height
    nw = w / frame_width
    nh = h / frame_height

    expected_x = 0.07
    expected_y = 0.07
    expected_w = 0.87
    expected_h = 0.80

    position_distance = (
        abs(nx - expected_x) * 2.0
        + abs(ny - expected_y) * 2.0
    )

    size_distance = (
        abs(nw - expected_w) * 2.0
        + abs(nh - expected_h) * 2.0
    )

    expected_aspect = expected_w / expected_h

    aspect_distance = abs(
        aspect_ratio - expected_aspect
    )

    aspect_score = max(
        0.0,
        2.0 - aspect_distance
    )

    position_score = max(
        0.0,
        3.0 - position_distance * 10
    )

    size_score = max(
        0.0,
        3.0 - size_distance * 8
    )

    area_score = min(
        area_ratio * 3.0,
        3.0
    )

    score = (
        position_score
        + size_score
        + aspect_score
        + area_score
    )

    return score


# ---------------------------------------------------------
# V2 detection
# ---------------------------------------------------------

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

        edge_results.append(connected)

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

            rectangularity = (
                contour_area /
                rectangle_area
            )

            if rectangularity < 0.20:
                continue

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

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    _, x, y, w, h = candidates[0]

    return x, y, w, h


# ---------------------------------------------------------
# Tracking validation
# ---------------------------------------------------------

def is_reasonable_movement(
    previous_box,
    current_box,
    frame_width,
    frame_height
):
    """
    Check whether the new detection is reasonably close
    to the previous scoreboard position and size.
    """

    if previous_box is None:
        return True

    px, py, pw, ph = previous_box
    cx, cy, cw, ch = current_box

    # -----------------------------------------------------
    # Position movement
    # -----------------------------------------------------

    max_x = frame_width * MAX_X_MOVEMENT
    max_y = frame_height * MAX_Y_MOVEMENT

    if abs(cx - px) > max_x:
        return False

    if abs(cy - py) > max_y:
        return False

    # -----------------------------------------------------
    # Size movement
    # -----------------------------------------------------

    width_change = abs(cw - pw) / max(pw, 1)
    height_change = abs(ch - ph) / max(ph, 1)

    if width_change > MAX_SIZE_CHANGE:
        return False

    if height_change > MAX_SIZE_CHANGE:
        return False

    return True


# ---------------------------------------------------------
# Find detection close to previous box
# ---------------------------------------------------------

def detect_near_previous(
    frame,
    previous_box
):
    """
    Run the V2 detector and accept the result only if
    it is spatially consistent with the previous box.
    """

    candidate = detect_scoreboard(frame)

    if candidate is None:
        return None

    frame_height, frame_width = frame.shape[:2]

    if not is_reasonable_movement(
        previous_box,
        candidate,
        frame_width,
        frame_height
    ):
        return None

    return candidate


# ---------------------------------------------------------
# Main tracking pipeline
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("VisionScore - Temporal Tracking V3")
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
    missed_frames = 0

    direct_detections = 0
    tracked_frames = 0
    lost_frames = 0

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

        output = frame.copy()

        # -------------------------------------------------
        # Case 1: No previous tracking information
        # -------------------------------------------------

        if previous_box is None:

            detection = detect_scoreboard(
                frame
            )

            if detection is not None:

                previous_box = detection
                missed_frames = 0
                direct_detections += 1

                x, y, w, h = detection

                status = "DIRECT"

            else:

                detection = None
                status = "LOST"

        # -------------------------------------------------
        # Case 2: Previous box exists
        # -------------------------------------------------

        else:

            detection = detect_near_previous(
                frame,
                previous_box
            )

            if detection is not None:

                previous_box = detection
                missed_frames = 0
                direct_detections += 1

                x, y, w, h = detection

                status = "TRACKED"

            else:

                missed_frames += 1

                if missed_frames <= MAX_MISSED_FRAMES:

                    detection = previous_box

                    tracked_frames += 1

                    x, y, w, h = detection

                    status = (
                        f"TRACKED-HOLD "
                        f"({missed_frames})"
                    )

                else:

                    detection = None
                    previous_box = None
                    lost_frames += 1

                    status = "LOST"

        # -------------------------------------------------
        # Draw result
        # -------------------------------------------------

        if detection is not None:

            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                4
            )

            cv2.putText(
                output,
                f"V3 {status}",
                (x, max(y - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            print(
                f"{image_path.name}: "
                f"x={x}, y={y}, "
                f"width={w}, height={h} "
                f"[{status}]"
            )

        else:

            cv2.putText(
                output,
                "SCOREBOARD LOST",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

            print(
                f"{image_path.name}: "
                f"NO SCOREBOARD "
                f"[{status}]"
            )

        output_path = (
            OUTPUT_DIR /
            image_path.name
        )

        cv2.imwrite(
            str(output_path),
            output
        )

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("TEMPORAL TRACKING SUMMARY")
    print("=" * 60)

    print(
        f"Direct detections: {direct_detections}"
    )

    print(
        f"Tracking holds:    {tracked_frames}"
    )

    print(
        f"Lost frames:       {lost_frames}"
    )

    total_frames = len(image_paths)

    successful = (
        direct_detections +
        tracked_frames
    )

    print(
        f"Successful localization: "
        f"{successful}/{total_frames}"
    )

    if total_frames > 0:

        coverage = (
            successful /
            total_frames
        ) * 100

        print(
            f"Coverage: {coverage:.2f}%"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()