import csv
from pathlib import Path

import cv2
import easyocr


INPUT_DIR = Path("screenshots/training_frames")
LOCALIZATION_CSV = Path("outputs/localization_v6.csv")
OUTPUT_DIR = Path("screenshots/ocr")
OUTPUT_CSV = Path("outputs/ocr_results.csv")


def clean_text(text):
    return " ".join(str(text).strip().split())


def extract_score_candidates(text):
    import re

    tokens = re.findall(r'(?<!\d)\d{1,3}(?!\d)', text)

    values = []

    for token in tokens:
        try:
            value = int(token)
        except ValueError:
            continue

        if 0 <= value <= 200:
            values.append(value)

    return values


def main():
    print("=" * 60)
    print("VisionScore - OCR Stage")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("Loading EasyOCR...")
    reader = easyocr.Reader(
        ["en"],
        gpu=False,
        verbose=False
    )
    print("EasyOCR ready.")
    print()

    rows = []

    with open(
        LOCALIZATION_CSV,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:
        detections = list(csv.DictReader(f))

    for row in detections:

        frame_name = row["frame"]
        status = row["status"]

        image_path = INPUT_DIR / frame_name

        if not image_path.exists():
            print(f"{frame_name}: IMAGE NOT FOUND")
            continue

        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"{frame_name}: READ ERROR")
            continue

        # Skip genuinely lost frames.
        if not row["x"]:
            print(f"{frame_name}: {status} - OCR SKIPPED")
            rows.append({
                "frame": frame_name,
                "localization_status": status,
                "ocr_text": "",
                "ocr_confidence": 0.0
            })
            continue

        x = int(float(row["x"]))
        y = int(float(row["y"]))
        w = int(float(row["width"]))
        h = int(float(row["height"]))

        # Safety bounds.
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        w = max(1, min(w, frame.shape[1] - x))
        h = max(1, min(h, frame.shape[0] - y))

        crop = frame[y:y+h, x:x+w]

        if crop.size == 0:
            print(f"{frame_name}: EMPTY CROP")
            continue

        # Save crop for inspection.
        cv2.imwrite(
            str(OUTPUT_DIR / frame_name),
            crop
        )

        # Resize for OCR.
        scale = 1.5
        resized = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        results = reader.readtext(
            resized,
            detail=1,
            paragraph=False,
            text_threshold=0.5,
            low_text=0.3,
            link_threshold=0.3
        )

        texts = []
        confidences = []

        for detection in results:
            text = clean_text(detection[1])
            confidence = float(detection[2])

            if text:
                texts.append(text)
                confidences.append(confidence)

        combined_text = " | ".join(texts)

        average_confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else 0.0
        )

        print(
            f"{frame_name}: "
            f"{len(texts)} text regions | "
            f"conf={average_confidence:.2f} | "
            f"{combined_text}"
        )

        rows.append({
            "frame": frame_name,
            "localization_status": status,
            "ocr_text": combined_text,
            "ocr_confidence": round(
                average_confidence,
                4
            )
        })

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "localization_status",
                "ocr_text",
                "ocr_confidence"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 60)
    print("OCR SUMMARY")
    print("=" * 60)
    print(f"Processed: {len(rows)}/{len(detections)}")
    print(f"CSV report: {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
