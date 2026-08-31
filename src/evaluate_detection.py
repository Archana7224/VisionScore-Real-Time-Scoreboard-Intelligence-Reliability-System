import csv
from pathlib import Path

import cv2


ANNOTATIONS_FILE = Path("configs/annotations.csv")
IMAGE_DIR = Path("screenshots/training_frames")


def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union (IoU)
    between two bounding boxes.

    Box format:
    (x, y, width, height)
    """

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2 = ax + aw
    ay2 = ay + ah

    bx2 = bx + bw
    by2 = by + bh

    intersection_x1 = max(ax, bx)
    intersection_y1 = max(ay, by)

    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

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


def load_annotations():
    if not ANNOTATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Annotations not found: {ANNOTATIONS_FILE}"
        )

    annotations = []

    with ANNOTATIONS_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            annotations.append(
                {
                    "image": row["image"],
                    "box": (
                        int(row["x"]),
                        int(row["y"]),
                        int(row["width"]),
                        int(row["height"]),
                    ),
                }
            )

    return annotations


def main():
    print("=" * 60)
    print("VisionScore - Annotation Validation")
    print("=" * 60)

    annotations = load_annotations()

    print(f"Loaded {len(annotations)} annotations.")

    valid = 0

    for item in annotations:

        image_path = IMAGE_DIR / item["image"]

        if not image_path.exists():
            print(
                f"WARNING: Missing image: {image_path}"
            )
            continue

        frame = cv2.imread(str(image_path))

        if frame is None:
            print(
                f"WARNING: Could not read: {image_path}"
            )
            continue

        height, width = frame.shape[:2]

        x, y, box_width, box_height = item["box"]

        if x < 0 or y < 0:
            print(
                f"INVALID: {item['image']} "
                "has negative coordinates."
            )
            continue

        if box_width <= 0 or box_height <= 0:
            print(
                f"INVALID: {item['image']} "
                "has invalid dimensions."
            )
            continue

        if x + box_width > width:
            print(
                f"INVALID: {item['image']} "
                "exceeds frame width."
            )
            continue

        if y + box_height > height:
            print(
                f"INVALID: {item['image']} "
                "exceeds frame height."
            )
            continue

        valid += 1

        print(
            f"VALID: {item['image']} "
            f"→ {width}x{height} frame"
        )

    print()
    print("=" * 60)
    print(
        f"Validation result: "
        f"{valid}/{len(annotations)} valid"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()