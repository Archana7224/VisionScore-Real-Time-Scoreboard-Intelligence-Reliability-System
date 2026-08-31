import cv2
import json
import os


REPORT_FILE = "outputs/final_report.json"
FRAME_DIR = "screenshots/training_frames"
OUTPUT_DIR = "outputs/evidence"


# Scoreboard coordinates used throughout the project.
# These are relative to the ORIGINAL frame.
SCOREBOARD_X1 = 260
SCOREBOARD_Y1 = 550
SCOREBOARD_X2 = 840
SCOREBOARD_Y2 = 645


# Player score positions inside the scoreboard ROI.
# Based on the OCR experiments you already performed.
PLAYER_BOXES = {
    "player_1": (30, 5, 140, 95),
    "player_2": (165, 5, 275, 95),
    "player_3": (305, 5, 415, 95),
    "player_4": (445, 5, 550, 95),
}


def load_report():
    with open(
        REPORT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def draw_text(
    image,
    text,
    position,
    scale=0.7,
    thickness=2
):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def generate_evidence(event, index):

    frame_name = event["event_frame"]

    frame_path = os.path.join(
        FRAME_DIR,
        frame_name
    )

    image = cv2.imread(frame_path)

    if image is None:
        print(
            f"WARNING: Could not read {frame_path}"
        )
        return

    # --------------------------------------------------
    # Draw scoreboard boundary
    # --------------------------------------------------

    cv2.rectangle(
        image,
        (SCOREBOARD_X1, SCOREBOARD_Y1),
        (SCOREBOARD_X2, SCOREBOARD_Y2),
        (255, 255, 255),
        3
    )

    # --------------------------------------------------
    # Highlight changed player's score
    # --------------------------------------------------

    player = event["player"]

    if player in PLAYER_BOXES:

        x1, y1, x2, y2 = PLAYER_BOXES[player]

        absolute_x1 = SCOREBOARD_X1 + x1
        absolute_y1 = SCOREBOARD_Y1 + y1
        absolute_x2 = SCOREBOARD_X1 + x2
        absolute_y2 = SCOREBOARD_Y1 + y2

        cv2.rectangle(
            image,
            (absolute_x1, absolute_y1),
            (absolute_x2, absolute_y2),
            (255, 255, 255),
            5
        )

    # --------------------------------------------------
    # Create information panel
    # --------------------------------------------------

    panel_x1 = 40
    panel_y1 = 40
    panel_x2 = 700
    panel_y2 = 220

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (0, 0, 0),
        -1
    )

    image = cv2.addWeighted(
        overlay,
        0.75,
        image,
        0.25,
        0
    )

    # --------------------------------------------------
    # Event information
    # --------------------------------------------------

    draw_text(
        image,
        "VisionScore - Score Change Evidence",
        (60, 75),
        scale=0.8,
        thickness=2
    )

    draw_text(
        image,
        f"Player: {event['player']}",
        (60, 110)
    )

    draw_text(
        image,
        f"Score: {event['old_score']} -> {event['new_score']}",
        (60, 145)
    )

    draw_text(
        image,
        f"Delta: {event['delta']:+d}",
        (60, 180)
    )

    draw_text(
        image,
        f"Reliability: {event['reliability']}%",
        (400, 110)
    )

    draw_text(
        image,
        f"Status: {event['status']}",
        (400, 145)
    )

    draw_text(
        image,
        f"Frame: {frame_name}",
        (400, 180)
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output_name = (
        f"event_{index:02d}_"
        f"{player}_"
        f"{event['old_score']}_"
        f"to_"
        f"{event['new_score']}.jpg"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    cv2.imwrite(
        output_path,
        image
    )

    print(
        f"Evidence created: {output_path}"
    )


def main():

    print("=" * 60)
    print("VisionScore - Visual Evidence Generator")
    print("=" * 60)

    report = load_report()

    events = report["confirmed_events"]

    print(
        f"Confirmed events: {len(events)}"
    )

    print()

    for index, event in enumerate(
        events,
        start=1
    ):

        generate_evidence(
            event,
            index
        )

    print()
    print("=" * 60)
    print(
        f"Evidence generated: {len(events)}"
    )
    print(
        f"Output directory: {OUTPUT_DIR}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()