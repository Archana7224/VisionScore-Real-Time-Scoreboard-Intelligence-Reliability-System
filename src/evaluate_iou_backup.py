import cv2
from pathlib import Path


ANNOTATION_DIR = Path("screenshots/annotations")
AUTO_DETECTION_DIR = Path("screenshots/auto_detection_v6")
TRAINING_DIR = Path("screenshots/training_frames")


def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union (IoU).

    Boxes are represented as:
    (x, y, width, height)
    """

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    # Convert to corner coordinates.
    a_x2 = ax + aw
    a_y2 = ay + ah

    b_x2 = bx + bw
    b_y2 = by + bh

    # Intersection rectangle.
    intersection_x1 = max(ax, bx)
    intersection_y1 = max(ay, by)

    intersection_x2 = min(a_x2, b_x2)
    intersection_y2 = min(a_y2, b_y2)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width *
        intersection_height
    )

    # Areas.
    area_a = aw * ah
    area_b = bw * bh

    union_area = (
        area_a +
        area_b -
        intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def extract_box_from_green_rectangle(image):
    """
    Extract the manually annotated rectangle.

    The annotation images contain a green rectangle.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Green color range.
    lower_green = (40, 80, 80)
    upper_green = (90, 255, 255)

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(
        contours,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(
        largest
    )

    if w < 50 or h < 50:
        return None

    return x, y, w, h


def extract_box_from_auto_detection(image):
    """
    Extract the green automatic-detection rectangle.
    """

    return extract_box_from_green_rectangle(image)


def main():

    print("=" * 60)
    print("VisionScore - IoU Evaluation")
    print("=" * 60)

    annotation_files = sorted(
        ANNOTATION_DIR.glob("*.jpg")
    )

    if not annotation_files:
        print("ERROR: No annotation images found.")
        return

    total = 0
    successful = 0
    iou_values = []

    for annotation_path in annotation_files:

        filename = annotation_path.name

        auto_path = (
            AUTO_DETECTION_DIR /
            filename
        )

        if not auto_path.exists():
            print(
                f"{filename}: "
                f"NO AUTOMATIC DETECTION"
            )
            continue

        annotation_image = cv2.imread(
            str(annotation_path)
        )

        auto_image = cv2.imread(
            str(auto_path)
        )

        if (
            annotation_image is None or
            auto_image is None
        ):
            print(
                f"{filename}: "
                f"Could not read images"
            )
            continue

        ground_truth = extract_box_from_green_rectangle(
            annotation_image
        )

        prediction = extract_box_from_auto_detection(
            auto_image
        )

        if ground_truth is None:
            print(
                f"{filename}: "
                f"Could not find ground-truth box"
            )
            continue

        if prediction is None:
            print(
                f"{filename}: "
                f"Could not find predicted box"
            )
            continue

        iou = calculate_iou(
            ground_truth,
            prediction
        )

        total += 1
        successful += 1
        iou_values.append(iou)

        print(
            f"{filename}: "
            f"IoU = {iou:.3f}"
        )

    print()
    print("=" * 60)

    if iou_values:

        average_iou = (
            sum(iou_values) /
            len(iou_values)
        )

        print(
            f"Evaluated: "
            f"{successful}/{total}"
        )

        print(
            f"Average IoU: "
            f"{average_iou:.3f}"
        )

        good = sum(
            1
            for value in iou_values
            if value >= 0.50
        )

        print(
            f"IoU >= 0.50: "
            f"{good}/{len(iou_values)}"
        )

    else:

        print("No valid IoU values.")

    print("=" * 60)


if __name__ == "__main__":
    main()