from pathlib import Path

path = Path("src/localization_v7.py")

text = path.read_text(encoding="utf-8")

old = """        # =================================================
        # TEMPORAL HOLD
        # =================================================

        if box is None:

            if (
                previous_box is not None
                and hold_frames < MAX_HOLD_FRAMES
            ):"""

new = """        # =================================================
        # V7.1 RECOVERY SEARCH
        # =================================================

        if box is None and previous_box is not None and RECOVERY_ENABLED:

            recovered = recover_scoreboard(
                frame,
                previous_box
            )

            if recovered is not None:

                box = recovered
                previous_box = box
                hold_frames = 0
                status = "RECOVERED"
                confidence = 0.65

        # =================================================
        # TEMPORAL HOLD
        # =================================================

        if box is None:

            if (
                previous_box is not None
                and hold_frames < MAX_HOLD_FRAMES
            ):"""

if old not in text:
    print("ERROR: Target block not found.")
    print("NO CHANGES MADE.")
    raise SystemExit(1)

backup = Path("src/localization_v7_before_v71_connection.py")
backup.write_text(text, encoding="utf-8")

path.write_text(
    text.replace(old, new, 1),
    encoding="utf-8"
)

print("V7.1 connection added successfully.")
print("Backup created:", backup)