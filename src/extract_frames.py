import cv2
from pathlib import Path


VIDEO_PATH = Path("data/bowling_scoreboard.mp4")
OUTPUT_DIR = Path("screenshots/training_frames")


def main():

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(
            "Invalid video metadata."
        )

    duration = frame_count / fps

    print("=" * 60)
    print("VisionScore - Training Frame Extraction")
    print("=" * 60)

    print(f"FPS: {fps:.2f}")
    print(f"Frames: {frame_count}")
    print(f"Duration: {duration:.2f} seconds")
    print()

    # Extract one frame every 2 seconds
    interval = 2.0

    timestamp = 0.0
    index = 1

    while timestamp < duration:

        frame_number = int(
            timestamp * fps
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        success, frame = cap.read()

        if success and frame is not None:

            output_path = (
                OUTPUT_DIR /
                f"train_{index:03d}.jpg"
            )

            cv2.imwrite(
                str(output_path),
                frame
            )

            print(
                f"Saved {output_path} "
                f"(time={timestamp:.2f}s)"
            )

            index += 1

        timestamp += interval

    cap.release()

    print()
    print("=" * 60)
    print(
        f"Extracted {index - 1} training frames."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()