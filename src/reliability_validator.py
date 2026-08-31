import csv
import os

INPUT_FILE = "outputs/temporal_results.csv"
OUTPUT_FILE = "outputs/reliability_results.csv"

PLAYERS = [
    "player_1",
    "player_2",
    "player_3",
    "player_4"
]

MIN_SEGMENT_LENGTH = 3


def load_results():
    rows = []

    with open(INPUT_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def get_segments(rows, player):
    """
    Convert a player's frame-by-frame scores into stable segments.

    Example:

        34,34,31,31,31,34,34

    becomes:

        34 -> frames 1-2
        31 -> frames 3-5
        34 -> frames 6-7
    """

    segments = []

    start = 0
    current_score = int(rows[0][player])

    for i in range(1, len(rows)):

        score = int(rows[i][player])

        if score != current_score:

            segments.append({
                "score": current_score,
                "start": start,
                "end": i - 1,
                "length": i - start
            })

            start = i
            current_score = score

    # Add final segment
    segments.append({
        "score": current_score,
        "start": start,
        "end": len(rows) - 1,
        "length": len(rows) - start
    })

    return segments


def validate_segments(rows, player):

    segments = get_segments(rows, player)

    events = []

    for i in range(1, len(segments)):

        previous = segments[i - 1]
        current = segments[i]

        old_score = previous["score"]
        new_score = current["score"]

        frame = rows[current["start"]]["frame"]

        # --------------------------------------------------
        # Detect temporary bounce:
        #
        # A -> B -> A
        #
        # Example:
        # 34 -> 31 -> 34
        # --------------------------------------------------

        is_bounce = False

        if i + 1 < len(segments):

            next_segment = segments[i + 1]

            if next_segment["score"] == old_score:
                is_bounce = True

        # --------------------------------------------------
        # Calculate reliability
        # --------------------------------------------------

        if is_bounce:

            status = "REJECTED"
            reason = "score_returned_to_previous_value"

            reliability = 10

        elif current["length"] >= MIN_SEGMENT_LENGTH:

            status = "CONFIRMED"
            reason = "stable_new_score"

            reliability = 90

        else:

            status = "UNCERTAIN"
            reason = "short_score_segment"

            reliability = 40

        events.append({
            "frame": frame,
            "player": player,
            "old_score": old_score,
            "new_score": new_score,
            "delta": new_score - old_score,
            "segment_length": current["length"],
            "reliability": reliability,
            "status": status,
            "reason": reason
        })

    return events


def save_results(events):

    os.makedirs("outputs", exist_ok=True)

    fieldnames = [
        "frame",
        "player",
        "old_score",
        "new_score",
        "delta",
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
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(events)


def main():

    print("=" * 60)
    print("VisionScore - Reliability Validator V3")
    print("=" * 60)

    rows = load_results()

    print(f"Loaded {len(rows)} temporal frames.")

    all_events = []

    for player in PLAYERS:

        events = validate_segments(rows, player)

        all_events.extend(events)

    print("\nVALIDATED EVENTS")
    print("-" * 60)

    for event in all_events:

        print(
            f'{event["frame"]} | '
            f'{event["player"]} | '
            f'{event["old_score"]} -> '
            f'{event["new_score"]} | '
            f'segment={event["segment_length"]} frames | '
            f'reliability={event["reliability"]}% | '
            f'{event["status"]} | '
            f'{event["reason"]}'
        )

    save_results(all_events)

    confirmed = sum(
        1 for e in all_events
        if e["status"] == "CONFIRMED"
    )

    rejected = sum(
        1 for e in all_events
        if e["status"] == "REJECTED"
    )

    uncertain = sum(
        1 for e in all_events
        if e["status"] == "UNCERTAIN"
    )

    print("\n" + "=" * 60)
    print(f"Events analyzed : {len(all_events)}")
    print(f"Confirmed       : {confirmed}")
    print(f"Rejected        : {rejected}")
    print(f"Uncertain       : {uncertain}")
    print(f"CSV report      : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()