import csv
from datetime import datetime
from pathlib import Path


def mark_attendance(name, attendance_file):
    path = Path(attendance_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_names = set()
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader, None)
            existing_names = {row[0] for row in reader if row}

    if name in existing_names:
        return False

    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["name", "date", "time"])

        now = datetime.now()
        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])

    return True
