import csv


INPUT_FILE = "outputs/temporal_results.csv"
OUTPUT_FILE = "outputs/score_changes.csv"

PLAYERS = [
    "player_1",
    "player_2",
    "player_3",
    "player_4"
]

# Number of consecutive frames required
# before accepting a score change.
PERSISTENCE_FRAMES = 2


def read_temporal_results(filename):
    rows = []

    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def detect_changes(rows):

    events = []

    previous_scores = {
        player: None
        for player in PLAYERS
    }

    candidates = {
        player: None
        for player in PLAYERS
    }

    candidate_counts = {
        player: 0
        for player in PLAYERS
    }

    for row in rows:

        frame = row["frame"]

        for player in PLAYERS:

            current_score = int(row[player])

            previous_score = previous_scores[player]

            # First frame: establish baseline.
            if previous_score is None:
                previous_scores[player] = current_score
                continue

            # No change.
            if current_score == previous_score:

                candidates[player] = None
                candidate_counts[player] = 0

                continue

            # New candidate score.
            if candidates[player] != current_score:

                candidates[player] = current_score
                candidate_counts[player] = 1

            else:

                candidate_counts[player] += 1

            # Accept only after persistence.
            if candidate_counts[player] >= PERSISTENCE_FRAMES:

                event = {
                    "frame": frame,
                    "player": player,
                    "previous_score": previous_score,
                    "new_score": current_score,
                    "change": current_score - previous_score,
                    "event": "SCORE_CHANGE"
                }

                events.append(event)

                # New score becomes the trusted state.
                previous_scores[player] = current_score

                candidates[player] = None
                candidate_counts[player] = 0

    return events


def write_events(filename, events):

    fieldnames = [
        "frame",
        "player",
        "previous_score",
        "new_score",
        "change",
        "event"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for event in events:
            writer.writerow(event)


def main():

    print("=" * 60)
    print("VisionScore - Score Change Detector")
    print("=" * 60)

    rows = read_temporal_results(INPUT_FILE)

    print(f"Loaded {len(rows)} temporal frames.")

    events = detect_changes(rows)

    write_events(
        OUTPUT_FILE,
        events
    )

    print()
    print("DETECTED EVENTS")
    print("-" * 60)

    if not events:

        print("No score changes detected.")

    else:

        for event in events:

            print(
                f"{event['frame']} | "
                f"{event['player']} | "
                f"{event['previous_score']} "
                f"-> {event['new_score']} | "
                f"delta={event['change']}"
            )

    print()
    print("=" * 60)
    print(f"Events detected: {len(events)}")
    print(f"CSV report: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()