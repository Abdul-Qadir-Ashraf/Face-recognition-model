import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.attendance_record import mark_attendance


def test_mark_attendance_creates_csv(tmp_path):
    attendance_file = tmp_path / "register.csv"

    was_marked = mark_attendance("Abdul", attendance_file)

    assert was_marked is True
    assert attendance_file.exists()

    with attendance_file.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    assert rows[0] == ["name", "date", "time"]
    assert rows[1][0] == "Abdul"


def test_mark_attendance_prevents_duplicate_names(tmp_path):
    attendance_file = tmp_path / "register.csv"

    first_mark = mark_attendance("Abdul", attendance_file)
    second_mark = mark_attendance("Abdul", attendance_file)

    with attendance_file.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    assert first_mark is True
    assert second_mark is False
    assert len(rows) == 2
