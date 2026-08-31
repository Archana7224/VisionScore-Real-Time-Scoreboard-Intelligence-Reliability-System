import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

LOCALIZATION_FILE = OUTPUTS / "localization_v6.csv"
TEMPORAL_FILE = OUTPUTS / "temporal_results.csv"
OCR_FILE = OUTPUTS / "ocr_results.csv"
RELIABILITY_FILE = OUTPUTS / "reliability_results.csv"
FINAL_REPORT_FILE = OUTPUTS / "final_report.json"
OUTPUT_FILE = OUTPUTS / "system_metrics.json"


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def format_pct(numerator, denominator):
    if denominator == 0:
        return "0.00%"
    return f"{(numerator / denominator) * 100:.2f}%"


def build_metrics():
    localization_rows = load_csv(LOCALIZATION_FILE)
    temporal_rows = load_csv(TEMPORAL_FILE)
    ocr_rows = load_csv(OCR_FILE)
    reliability_rows = load_csv(RELIABILITY_FILE)

    total_frames = len(temporal_rows) if temporal_rows else len(localization_rows)
    successful_localization = sum(
        1 for row in localization_rows if row.get("status", "").strip() != "LOST"
    )
    direct_detections = sum(
        1 for row in localization_rows if row.get("status", "").strip() == "DIRECT"
    )
    tracking_holds = sum(
        1 for row in localization_rows if "TRACKED-HOLD" in row.get("status", "")
    )
    lost_frames = sum(
        1 for row in localization_rows if row.get("status", "").strip() == "LOST"
    )

    ocr_frames_processed = len(ocr_rows) if ocr_rows else 0

    confirmed_events = [
        row for row in reliability_rows if row.get("status", "").strip() == "CONFIRMED"
    ]
    rejected_events = [
        row for row in reliability_rows if row.get("status", "").strip() == "REJECTED"
    ]
    uncertain_events = [
        row for row in reliability_rows if row.get("status", "").strip() == "UNCERTAIN"
    ]

    total_events = len(reliability_rows)
    confirmation_rate = (len(confirmed_events) / total_events * 100) if total_events else 0.0
    rejection_rate = (len(rejected_events) / total_events * 100) if total_events else 0.0

    if confirmed_events:
        average_confirmed_reliability = round(
            sum(safe_float(event.get("reliability", 0)) for event in confirmed_events)
            / len(confirmed_events),
            2,
        )
    else:
        average_confirmed_reliability = 0.0

    confirmed_score_changes = {}
    total_confirmed_increase = 0

    for event in confirmed_events:
        player = event.get("player", "unknown")
        delta = int(float(event.get("delta", 0)))
        confirmed_score_changes[player] = confirmed_score_changes.get(player, 0) + delta
        if delta > 0:
            total_confirmed_increase += delta

    metrics = {
        "frames_processed": total_frames,
        "localization_coverage": (
            (successful_localization / total_frames) * 100 if total_frames else 0.0
        ),
        "direct_detections": direct_detections,
        "tracking_holds": tracking_holds,
        "lost_frames": lost_frames,
        "ocr_frames_processed": ocr_frames_processed,
        "events_detected": total_events,
        "events_confirmed": len(confirmed_events),
        "events_rejected": len(rejected_events),
        "events_uncertain": len(uncertain_events),
        "confirmation_rate": confirmation_rate,
        "rejection_rate": rejection_rate,
        "average_confirmed_reliability": average_confirmed_reliability,
        "confirmed_score_changes": confirmed_score_changes,
        "total_confirmed_increase": total_confirmed_increase,
    }

    return metrics


def save_metrics(metrics):
    OUTPUTS.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def print_metrics(metrics):
    print("=" * 60)
    print("VisionScore - System Performance Metrics")
    print("=" * 60)
    print()
    print("FRAME PROCESSING")
    print("-" * 60)
    print(f"Frames processed        : {metrics['frames_processed']}")
    print(f"Localization coverage   : {metrics['localization_coverage']:.2f}%")
    print(f"Direct detections       : {metrics['direct_detections']}")
    print(f"Tracking holds          : {metrics['tracking_holds']}")
    print(f"Lost frames             : {metrics['lost_frames']}")
    print()
    print("OCR")
    print("-" * 60)
    print(f"Frames processed        : {metrics['ocr_frames_processed']}/{metrics['frames_processed']}")
    print()
    print("EVENT INTELLIGENCE")
    print("-" * 60)
    print(f"Events detected         : {metrics['events_detected']}")
    print(f"Confirmed events        : {metrics['events_confirmed']}")
    print(f"Rejected events         : {metrics['events_rejected']}")
    print(f"Uncertain events        : {metrics['events_uncertain']}")
    print()
    print(f"Confirmation rate       : {metrics['confirmation_rate']:.2f}%")
    print(f"Rejection rate          : {metrics['rejection_rate']:.2f}%")
    print()
    print("Average confirmed")
    print(f"reliability             : {metrics['average_confirmed_reliability']:.2f}%")
    print()
    print("CONFIRMED SCORE IMPACT")
    print("-" * 60)
    for player, delta in metrics["confirmed_score_changes"].items():
        if delta > 0:
            print(f"{player:<22} : +{delta}")
    print()
    print(f"Total confirmed increase: +{metrics['total_confirmed_increase']}")
    print()
    print("=" * 60)
    print(f"JSON report : {OUTPUT_FILE}")
    print("=" * 60)


def main():
    metrics = build_metrics()
    save_metrics(metrics)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
