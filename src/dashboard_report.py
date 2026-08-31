import json
import os
import webbrowser


METRICS_FILE = "outputs/system_metrics.json"
FINAL_REPORT_FILE = "outputs/final_report.json"
EVIDENCE_DIR = "outputs/evidence"
OUTPUT_FILE = "outputs/visionscore_dashboard.html"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe(value):
    if value is None:
        return "-"
    return str(value)


def build_event_rows(report):
    rows = ""

    for event in report.get("confirmed_events", []):
        rows += f"""
        <tr>
            <td>{safe(event["player"])}</td>
            <td>{safe(event["old_score"])} → {safe(event["new_score"])}</td>
            <td>+{safe(event["delta"])}</td>
            <td>
                <span class="badge confirmed">
                    CONFIRMED
                </span>
            </td>
            <td>{safe(event["reliability"])}%</td>
            <td>{safe(event["event_frame"])}</td>
        </tr>
        """

    for event in report.get("rejected_events", []):
        rows += f"""
        <tr>
            <td>{safe(event["player"])}</td>
            <td>{safe(event["old_score"])} → {safe(event["new_score"])}</td>
            <td>{safe(event["delta"])}</td>
            <td>
                <span class="badge rejected">
                    REJECTED
                </span>
            </td>
            <td>{safe(event["reliability"])}%</td>
            <td>{safe(event.get("event_frame", "-"))}</td>
        </tr>
        """

    return rows


def build_evidence_cards(report):
    cards = ""

    confirmed_events = report.get("confirmed_events", [])

    for index, event in enumerate(confirmed_events, start=1):

        player = event["player"]
        old_score = event["old_score"]
        new_score = event["new_score"]

        filename = (
            f"event_{index:02d}_"
            f"{player}_"
            f"{old_score}_to_{new_score}.jpg"
        )

        image_path = os.path.join(EVIDENCE_DIR, filename)

        if os.path.exists(image_path):

            cards += f"""
            <div class="evidence-card">
                <img src="{os.path.basename(EVIDENCE_DIR)}/{filename}"
                     alt="Evidence for {player} score change">

                <div class="evidence-info">
                    <h3>{player}</h3>
                    <p>
                        {old_score} → {new_score}
                        &nbsp; | &nbsp;
                        +{event["delta"]}
                    </p>

                    <span class="badge confirmed">
                        {event["reliability"]}% RELIABLE
                    </span>
                </div>
            </div>
            """

    return cards


def main():

    print("=" * 60)
    print("VisionScore - Dashboard Generator")
    print("=" * 60)

    metrics = load_json(METRICS_FILE)
    report = load_json(FINAL_REPORT_FILE)

    os.makedirs("outputs", exist_ok=True)

    frames = metrics.get("frames_processed", 0)
    localization = metrics.get("localization_coverage", 0)
    direct = metrics.get("direct_detections", 0)
    tracking = metrics.get("tracking_holds", 0)
    lost = metrics.get("lost_frames", 0)

    ocr_processed = metrics.get("ocr_frames_processed", frames)

    events = metrics.get("events_detected", 0)
    confirmed = metrics.get("confirmed_events", 0)
    rejected = metrics.get("rejected_events", 0)
    uncertain = metrics.get("uncertain_events", 0)

    confirmation_rate = metrics.get("confirmation_rate", 0)
    avg_reliability = metrics.get(
        "average_confirmed_reliability",
        0
    )

    total_increase = metrics.get(
        "total_confirmed_score_increase",
        0
    )

    event_rows = build_event_rows(report)
    evidence_cards = build_evidence_cards(report)

    html = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>VisionScore Intelligence Dashboard</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background: #f4f6f8;
    color: #17202a;
}}

.container {{
    max-width: 1200px;
    margin: auto;
    padding: 30px;
}}

.header {{
    background: #111827;
    color: white;
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 25px;
}}

.header h1 {{
    margin: 0 0 8px 0;
    font-size: 32px;
}}

.header p {{
    margin: 0;
    color: #cbd5e1;
}}

.metrics {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(210px, 1fr));

    gap: 16px;
    margin-bottom: 25px;
}}

.metric {{
    background: white;
    padding: 22px;
    border-radius: 14px;

    box-shadow:
        0 2px 8px rgba(0,0,0,0.06);
}}

.metric h3 {{
    margin: 0 0 10px 0;
    font-size: 14px;
    color: #64748b;
}}

.metric .value {{
    font-size: 30px;
    font-weight: bold;
}}

.section {{
    background: white;
    padding: 25px;
    border-radius: 14px;
    margin-bottom: 25px;

    box-shadow:
        0 2px 8px rgba(0,0,0,0.06);
}}

.section h2 {{
    margin-top: 0;
}}

table {{
    width: 100%;
    border-collapse: collapse;
}}

th,
td {{
    padding: 13px;
    text-align: left;
    border-bottom: 1px solid #e5e7eb;
}}

th {{
    background: #f8fafc;
}}

.badge {{
    display: inline-block;
    padding: 5px 9px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: bold;
}}

.confirmed {{
    background: #dcfce7;
    color: #166534;
}}

.rejected {{
    background: #fee2e2;
    color: #991b1b;
}}

.pipeline {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
}}

.pipeline-step {{
    background: #eef2ff;
    padding: 12px 16px;
    border-radius: 10px;
    font-weight: bold;
}}

.arrow {{
    font-size: 20px;
    color: #64748b;
}}

.evidence-grid {{
    display: grid;

    grid-template-columns:
        repeat(auto-fit, minmax(320px, 1fr));

    gap: 20px;
}}

.evidence-card {{
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    overflow: hidden;
    background: #fafafa;
}}

.evidence-card img {{
    width: 100%;
    display: block;
}}

.evidence-info {{
    padding: 16px;
}}

.evidence-info h3 {{
    margin: 0 0 8px 0;
}}

.evidence-info p {{
    margin: 0 0 10px 0;
    color: #475569;
}}

.footer {{
    text-align: center;
    color: #64748b;
    padding: 20px;
}}

@media (max-width: 700px) {{

    .container {{
        padding: 15px;
    }}

    .header h1 {{
        font-size: 25px;
    }}

    table {{
        font-size: 13px;
    }}

    th,
    td {{
        padding: 8px;
    }}

}}

</style>

</head>


<body>

<div class="container">


<div class="header">

<h1>VisionScore</h1>

<p>
Real-Time Scoreboard Intelligence &
Reliability System
</p>

</div>


<div class="metrics">

<div class="metric">

<h3>FRAMES PROCESSED</h3>

<div class="value">
{frames}
</div>

</div>


<div class="metric">

<h3>LOCALIZATION COVERAGE</h3>

<div class="value">
{localization:.2f}%
</div>

</div>


<div class="metric">

<h3>EVENTS DETECTED</h3>

<div class="value">
{events}
</div>

</div>


<div class="metric">

<h3>CONFIRMED EVENTS</h3>

<div class="value">
{confirmed}
</div>

</div>


<div class="metric">

<h3>CONFIRMED RELIABILITY</h3>

<div class="value">
{avg_reliability:.1f}%
</div>

</div>


<div class="metric">

<h3>CONFIRMED SCORE GAIN</h3>

<div class="value">
+{total_increase}
</div>

</div>

</div>


<div class="section">

<h2>System Pipeline</h2>

<div class="pipeline">

<div class="pipeline-step">
Input Frames
</div>

<div class="arrow">→</div>

<div class="pipeline-step">
Localization
</div>

<div class="arrow">→</div>

<div class="pipeline-step">
OCR
</div>

<div class="arrow">→</div>

<div class="pipeline-step">
Temporal Validation
</div>

<div class="arrow">→</div>

<div class="pipeline-step">
Change Detection
</div>

<div class="arrow">→</div>

<div class="pipeline-step">
Reliability Engine
</div>

<div class="arrow">→</div>

<div class="pipeline-step">
Final Intelligence
</div>

</div>

</div>


<div class="section">

<h2>Frame Processing</h2>

<table>

<tr>
<th>Metric</th>
<th>Value</th>
</tr>

<tr>
<td>Frames processed</td>
<td>{frames}</td>
</tr>

<tr>
<td>Direct detections</td>
<td>{direct}</td>
</tr>

<tr>
<td>Tracking holds</td>
<td>{tracking}</td>
</tr>

<tr>
<td>Lost frames</td>
<td>{lost}</td>
</tr>

<tr>
<td>Localization coverage</td>
<td>{localization:.2f}%</td>
</tr>

<tr>
<td>OCR frames processed</td>
<td>{ocr_processed}/{frames}</td>
</tr>

</table>

</div>


<div class="section">

<h2>Event Intelligence</h2>

<table>

<tr>

<th>Player</th>
<th>Score Change</th>
<th>Delta</th>
<th>Status</th>
<th>Reliability</th>
<th>Event Frame</th>

</tr>

{event_rows}

</table>

</div>


<div class="section">

<h2>Event Statistics</h2>

<table>

<tr>
<td>Events detected</td>
<td>{events}</td>
</tr>

<tr>
<td>Confirmed events</td>
<td>{confirmed}</td>
</tr>

<tr>
<td>Rejected events</td>
<td>{rejected}</td>
</tr>

<tr>
<td>Uncertain events</td>
<td>{uncertain}</td>
</tr>

<tr>
<td>Confirmation rate</td>
<td>{confirmation_rate:.2f}%</td>
</tr>

<tr>
<td>Average confirmed reliability</td>
<td>{avg_reliability:.2f}%</td>
</tr>

</table>

</div>


<div class="section">

<h2>Visual Evidence</h2>

<div class="evidence-grid">

{evidence_cards}

</div>

</div>


<div class="footer">

VisionScore v1.0
<br>
Computer Vision • OCR • Temporal Intelligence • Reliability Analysis

</div>


</div>

</body>

</html>
"""
    total_increase = metrics.get(
    "total_confirmed_increase",
    0
)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(html)

    print()
    print("Dashboard created:")
    print(OUTPUT_FILE)

    print()
    print("Opening dashboard...")

    webbrowser.open(
        "file://" +
        os.path.abspath(OUTPUT_FILE)
    )

    print("=" * 60)


if __name__ == "__main__":
    main()