import cv2
from pathlib import Path


VIDEO_PATH = Path("data/bowling_scoreboard.mp4")


def main():
    print("=" * 50)
    print("VisionScore - Video Validation")
    print("=" * 50)

    if not VIDEO_PATH.exists():
        print(f"ERROR: Video not found: {VIDEO_PATH}")
        print("Please make sure the video exists in the data folder.")
        return

    print(f"Video found: {VIDEO_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("ERROR: OpenCV could not open the video.")
        print("Possible causes:")
        print("- Unsupported codec")
        print("- Corrupted video")
        print("- Invalid video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    print(f"FPS: {fps:.2f}")
    print(f"Frames: {frame_count}")
    print(f"Resolution: {width} x {height}")
    print(f"Duration: {duration:.2f} seconds")

    success, frame = cap.read()

    if not success or frame is None:
        print("ERROR: Video opened but the first frame could not be read.")
        cap.release()
        return

    print("First frame successfully read.")
    print(f"Frame shape: {frame.shape}")

    cap.release()

    print("=" * 50)
    print("VIDEO VALIDATION PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()