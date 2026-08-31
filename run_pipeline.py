import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Temporal Localization", "src/localization_v7.py"),
    ("OCR Scoreboard", "src/ocr_scoreboard.py"),
    ("Temporal Validation", "src/temporal_validator.py"),
    ("Score Change Detection", "src/change_detector.py"),
    ("Reliability Validation", "src/reliability_validator.py"),
    ("Final Intelligence Report", "src/final_report.py"),
    ("Visual Evidence Generation", "src/visual_evidence.py"),
    ("System Performance Metrics", "src/system_metrics.py"),
    ("Dashboard Report", "src/dashboard_report.py"),
]


def run_step(name, script):
    print("\n" + "=" * 60)
    print(f"RUNNING: {name}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script)],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        print(f"\n[FAILED] {name}")
        print(f"Script: {script}")
        sys.exit(result.returncode)

    print(f"\n[OK] {name}")


def main():
    print("=" * 60)
    print("VisionScore - Complete Pipeline")
    print("=" * 60)

    for name, script in STEPS:
        if not (PROJECT_ROOT / script).exists():
            print(f"\n[ERROR] Missing script: {script}")
            sys.exit(1)

        run_step(name, script)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated outputs:")

    outputs = [
        "outputs/localization_v6.csv",
        "outputs/ocr_results_v3.csv",
        "outputs/temporal_results.csv",
        "outputs/score_changes.csv",
        "outputs/reliability_results.csv",
        "outputs/final_report.json",
        "outputs/system_metrics.json",
        "outputs/visionscore_dashboard.html",
    ]

    for output in outputs:
        path = PROJECT_ROOT / output
        status = "OK" if path.exists() else "MISSING"
        print(f"[{status}] {output}")

    print("\nEvidence:")
    evidence_dir = PROJECT_ROOT / "outputs" / "evidence"

    if evidence_dir.exists():
        for file in evidence_dir.glob("*"):
            print(f"[OK] {file}")

    print("\nVisionScore processing complete.")


if __name__ == "__main__":
    main()