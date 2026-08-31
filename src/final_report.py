import csv
import json
import os


TEMPORAL_FILE = "outputs/temporal_results.csv"
RELIABILITY_FILE = "outputs/reliability_results.csv"
TIMELINE_FILE = "outputs/event_timeline.csv"

OUTPUT_FILE = "outputs/final_report.json"


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def calculate_average_reliability(events):
    if not events:
        return 0

    values = [
        float(event["reliability"])
        for event in events
        if event["status"] == "CONFIRMED"
    ]

    if not values:
        return 0

    return round(sum(values) / len(values), 2)


def build_report():

    temporal_rows = load_csv(TEMPORAL_FILE)
    reliability_rows = load_csv(RELIABILITY_FILE)
    timeline_rows = load_csv(TIMELINE_FILE)

    confirmed = [
        event
        for event in timeline_rows
        if event["status"] == "CONFIRMED"
    ]

    rejected = [
        event
        for event in timeline_rows
        if event["status"] == "REJECTED"
    ]

    uncertain = [
        event
        for event in timeline_rows
        if event["status"] == "UNCERTAIN"
    ]

    events = []

    for event in confirmed:

        events.append({
            "event_frame": event["event_frame"],
            "player": event["player"],
            "old_score": int(event["old_score"]),
            "new_score": int(event["new_score"]),
            "delta": int(event["delta"]),
            "start_frame": event["start_frame"],
            "end_frame": event["end_frame"],
            "segment_length": int(event["segment_length"]),
            "reliability": float(event["reliability"]),
            "status": event["status"],
            "reason": event["reason"]
        })

    report = {
        "system": "VisionScore",
        "version": "1.0",

        "dataset": {
            "frames_processed": len(temporal_rows)
        },

        "event_summary": {
            "events_analyzed": len(timeline_rows),
            "events_confirmed": len(confirmed),
            "events_rejected": len(rejected),
            "events_uncertain": len(uncertain),
            "average_confirmed_reliability":
                calculate_average_reliability(timeline_rows)
        },

        "confirmed_events": events,

        "rejected_events": [
            {
                "event_frame": event["event_frame"],
                "player": event["player"],
                "old_score": int(event["old_score"]),
                "new_score": int(event["new_score"]),
                "delta": int(event["delta"]),
                "reliability": float(event["reliability"]),
                "status": event["status"],
                "reason": event["reason"]
            }
            for event in rejected
        ]
    }

    return report


def save_report(report):

    os.makedirs("outputs", exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )


def print_report(report):

    summary = report["event_summary"]

    print("=" * 60)
    print("VisionScore - Final Intelligence Report")
    print("=" * 60)

    print()

    print(
        f"Frames processed       : "
        f"{report['dataset']['frames_processed']}"
    )

    print(
        f"Events analyzed        : "
        f"{summary['events_analyzed']}"
    )

    print(
        f"Confirmed events       : "
        f"{summary['events_confirmed']}"
    )

    print(
        f"Rejected events        : "
        f"{summary['events_rejected']}"
    )

    print(
        f"Uncertain events       : "
        f"{summary['events_uncertain']}"
    )

    print(
        f"Average reliability    : "
        f"{summary['average_confirmed_reliability']}%"
    )

    print()
    print("CONFIRMED SCORE CHANGES")
    print("-" * 60)

    for i, event in enumerate(
        report["confirmed_events"],
        start=1
    ):

        print(
            f"{i}. {event['player']} | "
            f"{event['old_score']} -> "
            f"{event['new_score']} | "
            f"delta={event['delta']:+d} | "
            f"reliability={event['reliability']}%"
        )

    print()
    print("REJECTED SCORE CHANGES")
    print("-" * 60)

    for event in report["rejected_events"]:

        print(
            f"{event['player']} | "
            f"{event['old_score']} -> "
            f"{event['new_score']} | "
            f"reliability={event['reliability']}% | "
            f"{event['reason']}"
        )

    print()
    print("=" * 60)
    print(f"JSON report : {OUTPUT_FILE}")
    print("=" * 60)


def main():

    print("Loading VisionScore data...")

    report = build_report()

    save_report(report)

    print_report(report)


if __name__ == "__main__":
    main()