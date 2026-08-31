import cv2
from pathlib import Path


ANNOTATION_DIR = Path("screenshots/annotations")
AUTO_DETECTION_DIR = Path("screenshots/auto_detection_v2")


def extract_green_box(image):
    """Extract the green annotation/detection rectangle."""

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

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


def calculate_iou(box_a, box_b):

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

    iw = max(
        0,
        ix2 - ix1
    )

    ih = max(
        0,
        iy2 - iy1
    )

    intersection = iw * ih

    area_a = aw * ah
    area_b = bw * bh

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def main():

    print("=" * 60)
    print("VisionScore - Localization V2 Report")
    print("=" * 60)
    print()

    annotation_files = sorted(
        ANNOTATION_DIR.glob("*.jpg")
    )

    total_annotations = len(
        annotation_files
    )

    evaluated = 0
    missed = 0
    ious = []

    difficult_frames = []

    for annotation_path in annotation_files:

        filename = annotation_path.name

        auto_path = (
            AUTO_DETECTION_DIR /
            filename
        )

        if not auto_path.exists():
            missed += 1
            continue

        annotation = cv2.imread(
            str(annotation_path)
        )

        prediction_image = cv2.imread(
            str(auto_path)
        )

        if (
            annotation is None
            or prediction_image is None
        ):
            missed += 1
            continue

        ground_truth = extract_green_box(
            annotation
        )

        prediction = extract_green_box(
            prediction_image
        )

        if (
            ground_truth is None
            or prediction is None
        ):
            missed += 1
            continue

        iou = calculate_iou(
            ground_truth,
            prediction
        )

        evaluated += 1
        ious.append(iou)

        if iou < 0.80:
            difficult_frames.append(
                (filename, iou)
            )

    print(
        f"Ground-truth annotations: "
        f"{total_annotations}"
    )

    print(
        f"Successfully evaluated:    "
        f"{evaluated}"
    )

    print(
        f"Missed predictions:        "
        f"{missed}"
    )

    if ious:

        average_iou = (
            sum(ious) /
            len(ious)
        )

        good = sum(
            1
            for value in ious
            if value >= 0.50
        )

        print(
            f"Average IoU:               "
            f"{average_iou:.3f}"
        )

        print(
            f"IoU >= 0.50:               "
            f"{good}/{len(ious)}"
        )

    print()

    print("Known difficult frames:")

    if difficult_frames:

        for filename, iou in difficult_frames:

            print(
                f"  {filename} → "
                f"IoU {iou:.3f}"
            )

    else:

        print("  None")

    print()

    print("Current method:")
    print(
        "  OpenCV edge + contour "
        "candidate scoring"
    )

    print()

    print("Decision:")

    if ious:

        if (
            average_iou >= 0.85
            and evaluated >= total_annotations * 0.90
        ):

            print(
                "  Localization quality is strong."
            )

        else:

            print(
                "  Continue improving localization."
            )

        print(
            "  Next major improvement: "
            "temporal tracking."
        )

    print("=" * 60)


if __name__ == "__main__":
    main()