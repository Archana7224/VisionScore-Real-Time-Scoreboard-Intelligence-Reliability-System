"""Generate VisionScore submission documentation PDF."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER
from PIL import Image as PILImage

SHOTS = "/tmp/agent-browser"
OUT = "/vercel/share/v0-project/docs/VisionScore_Documentation.pdf"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

INK = colors.HexColor("#0f172a")
MUT = colors.HexColor("#475569")
ACC = colors.HexColor("#059669")

styles = getSampleStyleSheet()
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold",
                    fontSize=22, textColor=INK, spaceAfter=4, leading=26)
sub = ParagraphStyle("sub", parent=styles["Normal"], fontName="Helvetica",
                     fontSize=11, textColor=MUT, spaceAfter=2, leading=15)
sec = ParagraphStyle("sec", parent=styles["Heading2"], fontName="Helvetica-Bold",
                     fontSize=14, textColor=ACC, spaceBefore=6, spaceAfter=6, leading=18)
body = ParagraphStyle("body", parent=styles["Normal"], fontName="Helvetica",
                      fontSize=10.5, textColor=INK, leading=15, spaceAfter=6)
cap = ParagraphStyle("cap", parent=styles["Normal"], fontName="Helvetica-Oblique",
                     fontSize=9, textColor=MUT, alignment=TA_CENTER, spaceBefore=3,
                     spaceAfter=10)


def img(path, max_w=CONTENT_W, max_h=140 * mm):
    im = PILImage.open(path)
    w, h = im.size
    ratio = min(max_w / w, max_h / h)
    return Image(path, width=w * ratio, height=h * ratio)


story = []

# ---- Title page ----
story.append(Spacer(1, 30 * mm))
story.append(Paragraph("VisionScore", h1))
story.append(Paragraph("Real-Time Scoreboard Intelligence &amp; Reliability System", sub))
story.append(Spacer(1, 4 * mm))
story.append(HRFlowable(width="100%", thickness=1.2, color=ACC))
story.append(Spacer(1, 6 * mm))
story.append(Paragraph(
    "Project documentation. This document walks through the complete working "
    "solution with screenshots of the input video, the project running, the "
    "scoreboard being detected, and the final extracted scoreboard data.", body))
story.append(Spacer(1, 4 * mm))
story.append(Paragraph(
    "<b>Pipeline:</b> Video &rarr; Frame extraction &rarr; Scoreboard localization "
    "(OpenCV) &rarr; OCR (EasyOCR) &rarr; Temporal series &rarr; Reliability analysis "
    "&rarr; Reliable structured data.", body))
story.append(Spacer(1, 4 * mm))
story.append(Paragraph(
    "<b>Tech stack:</b> Python, OpenCV, NumPy, EasyOCR, Flask (dashboard).", body))
story.append(Spacer(1, 10 * mm))
story.append(img(f"{SHOTS}/03-fixed-landing.png", max_h=95 * mm))
story.append(Paragraph("VisionScore dashboard - the single-screen console.", cap))
story.append(PageBreak())

# ---- Section 1: Input ----
story.append(Paragraph("1. Input Video / Frame", sec))
story.append(Paragraph(
    "The input is a bowling broadcast clip (<b>bowling_scoreboard.mp4</b>) containing "
    "an on-screen scoreboard. The system extracts frames from this video as the "
    "starting point. Below is a representative input frame showing the four players "
    "(J, V, P, T) and their running totals in the TTL column.", body))
story.append(img(f"{SHOTS}/doc-input-frame.jpg", max_h=110 * mm))
story.append(Paragraph("A single input frame extracted from bowling_scoreboard.mp4.", cap))
story.append(PageBreak())

# ---- Section 2: Code running ----
story.append(Paragraph("2. Code / Project Running", sec))
story.append(Paragraph(
    "Clicking <b>Run bundled demo</b> (or uploading a file) starts the real pipeline. "
    "The dashboard streams live progress: the input video plays on the left, and the "
    "<b>Pipeline Runtime</b> console logs each frame's localization result "
    "(DIRECT / TRACKED) and the OCR-read scores in real time.", body))
story.append(img(f"{SHOTS}/doc-running.png", max_h=125 * mm))
story.append(Paragraph(
    "The pipeline running live - progress bar, per-frame console output, and "
    "the current detection preview.", cap))
story.append(PageBreak())

# ---- Section 3: Detection ----
story.append(Paragraph("3. Scoreboard Being Detected / Extracted", sec))
story.append(Paragraph(
    "Each processed frame is annotated with a green bounding box around the localized "
    "scoreboard, its detection mode (DIRECT vs. TRACKED), and a confidence score. "
    "The close-up below shows a single detection at 97% confidence.", body))
story.append(img(f"{SHOTS}/06-lightbox.png", max_h=100 * mm))
story.append(Paragraph("Detected scoreboard with bounding box and confidence label.", cap))
story.append(Spacer(1, 3 * mm))
story.append(Paragraph(
    "The full detection grid shows every frame's result side by side, giving a clear "
    "view of localization coverage across the whole clip.", body))
story.append(img(f"{SHOTS}/doc-extracted.png", max_h=95 * mm))
story.append(Paragraph("Per-frame detection grid alongside the extracted data panel.", cap))
story.append(PageBreak())

# ---- Section 4: Extracted data ----
story.append(Paragraph("4. Extracted Scoreboard Data (Final Output)", sec))
story.append(Paragraph(
    "The final output is the reliable per-player score, produced after temporal "
    "reasoning across all frames: <b>P1 = 20, P2 = 34, P3 = 48, P4 = 54</b>. The "
    "Intelligence Report summarizes the run - 29 frames processed, 96.5% localization "
    "coverage, and the validated score-change events.", body))
story.append(img(f"{SHOTS}/doc-report.png", max_h=120 * mm))
story.append(Paragraph(
    "Intelligence Report: metrics plus the validated events table. Note how the "
    "reliability layer marks stable changes CONFIRMED (90%) and OCR flickers "
    "REJECTED (10%).", cap))
story.append(Spacer(1, 3 * mm))
story.append(Paragraph(
    "<b>Why this matters:</b> a naive per-frame OCR reader would flip P2 to 31 on a "
    "misread frame. VisionScore rejects that flicker because the value reverts, and "
    "keeps the stable, correct result - turning noisy OCR into trustworthy data.", body))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                        topMargin=MARGIN, bottomMargin=MARGIN,
                        title="VisionScore Documentation")
doc.build(story)
print("PDF written to", OUT)
print("size:", os.path.getsize(OUT), "bytes")
