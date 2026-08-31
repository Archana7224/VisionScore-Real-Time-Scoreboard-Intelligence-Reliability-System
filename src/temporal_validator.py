import csv
from collections import Counter


INPUT_FILE = "outputs/ocr_results_v3.csv"
OUTPUT_FILE = "outputs/temporal_results.csv"

PLAYERS = ["player_1", "player_2", "player_3", "player_4"]

# Number of recent frames used for temporal voting.
WINDOW_SIZE = 5


def read_ocr_results(filename):
    """Read OCR results from CSV."""
    rows = []

    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def temporal_vote(values):
    """
    Select the most frequently observed value
    in the recent temporal window.
    """

    valid_values = [
        int(v)
        for v in values
        if v is not None and str(v).strip() != ""
    ]

    if not valid_values:
        return None

    counts = Counter(valid_values)

    # Most common value.
    return counts.most_common(1)[0][0]


def stabilize_scores(rows):
    """Apply temporal voting independently to each player."""

    results = []

    history = {
        player: []
        for player in PLAYERS
    }

    for row in rows:

        stable_row = {
            "frame": row["frame"]
        }

        for player in PLAYERS:

            current_value = row[player]

            history[player].append(current_value)

            # Keep only the latest WINDOW_SIZE frames.
            if len(history[player]) > WINDOW_SIZE:
                history[player].pop(0)

            stable_value = temporal_vote(history[player])

            stable_row[player] = stable_value

        results.append(stable_row)

    return results


def write_results(filename, results):
    """Write stabilized results to CSV."""

    fieldnames = ["frame"] + PLAYERS

    with open(filename, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)


def print_results(results):

    print("=" * 60)
    print("VisionScore - Temporal Validation")
    print("=" * 60)

    for row in results:

        print(
            f"{row['frame']}: "
            f"P1={row['player_1']} "
            f"P2={row['player_2']} "
            f"P3={row['player_3']} "
            f"P4={row['player_4']}"
        )

    print()
    print("=" * 60)
    print(f"Processed: {len(results)} frames")
    print(f"CSV report: {OUTPUT_FILE}")
    print("=" * 60)


def main():

    print("Loading OCR results...")

    rows = read_ocr_results(INPUT_FILE)

    print(f"Loaded {len(rows)} frames.")

    results = stabilize_scores(rows)

    write_results(
        OUTPUT_FILE,
        results
    )

    print_results(results)


if __name__ == "__main__":
    main()