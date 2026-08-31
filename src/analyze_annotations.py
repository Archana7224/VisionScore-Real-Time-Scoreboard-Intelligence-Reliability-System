import csv
from pathlib import Path


ANNOTATIONS_FILE = Path("configs/annotations.csv")


def main():
    print("=" * 60)
    print("VisionScore - Scoreboard Movement Analysis")
    print("=" * 60)

    if not ANNOTATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Annotations file not found: {ANNOTATIONS_FILE}"
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
                    "x": int(row["x"]),
                    "y": int(row["y"]),
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                }
            )

    if not annotations:
        raise RuntimeError("No annotations found.")

    print(f"Annotations analyzed: {len(annotations)}")
    print()

    # ---------------------------------------------------------
    # Find ranges
    # ---------------------------------------------------------

    x_values = [item["x"] for item in annotations]
    y_values = [item["y"] for item in annotations]
    width_values = [item["width"] for item in annotations]
    height_values = [item["height"] for item in annotations]

    print("POSITION")
    print("-" * 60)

    print(
        f"X range:      {min(x_values)} → {max(x_values)}"
    )

    print(
        f"Y range:      {min(y_values)} → {max(y_values)}"
    )

    print()

    print("SIZE")
    print("-" * 60)

    print(
        f"Width range:  {min(width_values)} → {max(width_values)}"
    )

    print(
        f"Height range: {min(height_values)} → {max(height_values)}"
    )

    print()

    # ---------------------------------------------------------
    # Calculate approximate normalized coordinates
    # ---------------------------------------------------------

    FRAME_WIDTH = 1920
    FRAME_HEIGHT = 1080

    normalized_x = [
        item["x"] / FRAME_WIDTH
        for item in annotations
    ]

    normalized_y = [
        item["y"] / FRAME_HEIGHT
        for item in annotations
    ]

    normalized_width = [
        item["width"] / FRAME_WIDTH
        for item in annotations
    ]

    normalized_height = [
        item["height"] / FRAME_HEIGHT
        for item in annotations
    ]

    print("NORMALIZED RANGES")
    print("-" * 60)

    print(
        f"X:      {min(normalized_x):.3f} → "
        f"{max(normalized_x):.3f}"
    )

    print(
        f"Y:      {min(normalized_y):.3f} → "
        f"{max(normalized_y):.3f}"
    )

    print(
        f"Width:  {min(normalized_width):.3f} → "
        f"{max(normalized_width):.3f}"
    )

    print(
        f"Height: {min(normalized_height):.3f} → "
        f"{max(normalized_height):.3f}"
    )

    print()

    # ---------------------------------------------------------
    # Identify extreme examples
    # ---------------------------------------------------------

    smallest = min(
        annotations,
        key=lambda item: item["width"] * item["height"]
    )

    largest = max(
        annotations,
        key=lambda item: item["width"] * item["height"]
    )

    leftmost = min(
        annotations,
        key=lambda item: item["x"]
    )

    rightmost = max(
        annotations,
        key=lambda item: item["x"]
    )

    print("EXTREME CASES")
    print("-" * 60)

    print(
        f"Smallest scoreboard: "
        f"{smallest['image']} "
        f"({smallest['width']}x{smallest['height']})"
    )

    print(
        f"Largest scoreboard:  "
        f"{largest['image']} "
        f"({largest['width']}x{largest['height']})"
    )

    print(
        f"Leftmost scoreboard:  "
        f"{leftmost['image']} "
        f"(x={leftmost['x']})"
    )

    print(
        f"Rightmost scoreboard: "
        f"{rightmost['image']} "
        f"(x={rightmost['x']})"
    )

    print()

    # ---------------------------------------------------------
    # Final interpretation
    # ---------------------------------------------------------

    x_range = max(x_values) - min(x_values)
    y_range = max(y_values) - min(y_values)
    width_range = max(width_values) - min(width_values)
    height_range = max(height_values) - min(height_values)

    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if x_range > 100 or y_range > 100:
        print(
            "Scoreboard position changes significantly."
        )
    else:
        print(
            "Scoreboard position is relatively stable."
        )

    if width_range > 200 or height_range > 200:
        print(
            "Scoreboard size changes significantly."
        )
    else:
        print(
            "Scoreboard size is relatively stable."
        )

    print()
    print("Recommendation:")
    print(
        "Use the annotation data to evaluate an "
        "automated localization method before selecting "
        "a heavy object-detection model."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()