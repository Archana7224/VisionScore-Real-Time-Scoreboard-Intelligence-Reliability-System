import csv
import os


TEMPORAL_FILE = "outputs/temporal_results.csv"
RELIABILITY_FILE = "outputs/reliability_results.csv"
OUTPUT_FILE = "outputs/event_timeline.csv"


def frame_number(filename):
    """
    Convert train_017.jpg -> 17
    """
    try:
        return int(
            filename.replace("train_", "").replace(".jpg", "")
        )
    except ValueError:
        return -1


def load_temporal_results():
    with open(
        TEMPORAL_FILE,
        newline="",
        encoding="utf-8"
    ) as f:
        return list(csv.DictReader(f))


def load_reliability_results():
    with open(
        RELIABILITY_FILE,
        newline="",
        encoding="utf-8"
    ) as f:
        return list(csv.DictReader(f))


def get_score(row, player):
    try:
        return int(row[player])
    except (ValueError, KeyError):
        return None


def build_timeline(temporal_rows, reliability_rows):

    # Index temporal frames by frame number
    temporal_by_frame = {
        frame_number(row["frame"]): row
        for row in temporal_rows
    }

    events = []

    for event in reliability_rows:

        event_frame = frame_number(event["frame"])
        player = event["player"]

        old_score = int(event["old_score"])
        new_score = int(event["new_score"])

        segment_length = int(event["segment_length"])
        reliability = int(event["reliability"])

        status = event["status"]
        reason = event["reason"]

        # --------------------------------------------------
        # Find the beginning of the score segment
        # --------------------------------------------------

        start_frame = event_frame

        for i in range(event_frame - 1, 0, -1):

            row = temporal_by_frame.get(i)

            if row is None:
                continue

            score = get_score(row, player)

            if score == old_score:
                start_frame = i + 1
                break

        # --------------------------------------------------
        # Find the end of the new-score segment
        # --------------------------------------------------

        end_frame = event_frame

        for i in range(event_frame, len(temporal_rows) + 1):

            row = temporal_by_frame.get(i)

            if row is None:
                continue

            score = get_score(row, player)

            if score == new_score:
                end_frame = i
            else:
                break

        # --------------------------------------------------
        # Store event
        # --------------------------------------------------

        events.append({
            "event_frame": event["frame"],
            "player": player,
            "old_score": old_score,
            "new_score": new_score,
            "delta": new_score - old_score,
            "start_frame": f"train_{start_frame:03d}.jpg",
            "end_frame": f"train_{end_frame:03d}.jpg",
            "segment_length": segment_length,
            "reliability": reliability,
            "status": status,
            "reason": reason
        })

    return events


def save_events(events):

    os.makedirs("outputs", exist_ok=True)

    fields = [
        "event_frame",
        "player",
        "old_score",
        "new_score",
        "delta",
        "start_frame",
        "end_frame",
        "segment_length",
        "reliability",
        "status",
        "reason"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(events)


def print_report(events):

    print("=" * 60)
    print("VisionScore - Event Timeline")
    print("=" * 60)

    print()

    if not events:
        print("No events found.")
        return

    for i, event in enumerate(events, start=1):

        print(f"EVENT {i}")
        print("-" * 60)

        print(f"Player       : {event['player']}")
        print(f"Transition   : {event['old_score']} -> {event['new_score']}")
        print(f"Delta        : {event['delta']:+d}")
        print(f"Start frame  : {event['start_frame']}")
        print(f"Event frame  : {event['event_frame']}")
        print(f"End frame    : {event['end_frame']}")
        print(f"Segment      : {event['segment_length']} frames")
        print(f"Reliability  : {event['reliability']}%")
        print(f"Status       : {event['status']}")
        print(f"Reason       : {event['reason']}")

        print()

    confirmed = sum(
        1 for e in events
        if e["status"] == "CONFIRMED"
    )

    rejected = sum(
        1 for e in events
        if e["status"] == "REJECTED"
    )

    uncertain = sum(
        1 for e in events
        if e["status"] == "UNCERTAIN"
    )

    print("=" * 60)
    print(f"Events analyzed : {len(events)}")
    print(f"Confirmed       : {confirmed}")
    print(f"Rejected        : {rejected}")
    print(f"Uncertain       : {uncertain}")
    print(f"CSV report      : {OUTPUT_FILE}")
    print("=" * 60)


def main():

    print("Loading temporal results...")

    temporal_rows = load_temporal_results()

    print(
        f"Loaded {len(temporal_rows)} temporal frames."
    )

    print("Loading reliability results...")

    reliability_rows = load_reliability_results()

    print(
        f"Loaded {len(reliability_rows)} reliability events."
    )

    events = build_timeline(
        temporal_rows,
        reliability_rows
    )

    save_events(events)

    print_report(events)


if __name__ == "__main__":
    main()