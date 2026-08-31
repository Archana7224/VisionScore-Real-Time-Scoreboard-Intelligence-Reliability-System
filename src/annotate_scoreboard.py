import cv2
from pathlib import Path


INPUT_DIR = Path("screenshots/training_frames")
OUTPUT_DIR = Path("screenshots/annotations")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    images = sorted(
        [
            path
            for path in INPUT_DIR.iterdir()
            if path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )

    if not images:
        raise FileNotFoundError(
            f"No images found in {INPUT_DIR}"
        )

    print("=" * 60)
    print("VisionScore - Scoreboard Annotation Tool")
    print("=" * 60)
    print()
    print("Instructions:")
    print("1. A window will open with an image.")
    print("2. Drag a rectangle around the COMPLETE scoreboard.")
    print("3. Press ENTER or SPACE to confirm.")
    print("4. Press C to cancel/reset the selection.")
    print("5. Press ESC to stop.")
    print()

    for image_path in images:

        frame = cv2.imread(str(image_path))

        if frame is None:
            print(
                f"WARNING: Could not read {image_path.name}"
            )
            continue

        print("-" * 60)
        print(f"Annotating: {image_path.name}")

        window_name = "Draw scoreboard rectangle"

        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            window_name,
            1200,
            700
        )

        rectangle = cv2.selectROI(
            window_name,
            frame,
            fromCenter=False,
            showCrosshair=True
        )

        cv2.destroyWindow(window_name)

        x, y, w, h = rectangle

        if w <= 0 or h <= 0:
            print(
                f"Skipped {image_path.name}: "
                "no valid rectangle selected."
            )
            continue

        # Validate boundaries
        frame_height, frame_width = frame.shape[:2]

        if x < 0 or y < 0:
            print("ERROR: Negative coordinates.")
            continue

        if x + w > frame_width:
            print("ERROR: Rectangle exceeds frame width.")
            continue

        if y + h > frame_height:
            print("ERROR: Rectangle exceeds frame height.")
            continue

        # Draw annotation
        annotated = frame.copy()

        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        cv2.putText(
            annotated,
            "Scoreboard",
            (x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
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
            f"x={x}, y={y}, "
            f"width={w}, height={h}"
        )

        print(
            f"Saved annotation: {output_path}"
        )

    print()
    print("=" * 60)
    print("Annotation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()