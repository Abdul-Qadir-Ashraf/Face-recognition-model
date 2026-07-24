import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
import face_recognition
import numpy as np

from src.face_utils import draw_face_box, iter_image_files, load_rgb_image


def load_known_faces(employees_dir):
    known_encodings = []
    known_names = []

    for image_path in iter_image_files(employees_dir):
        image = load_rgb_image(image_path)
        encodings = face_recognition.face_encodings(image)

        if not encodings:
            print(f"Skipping {image_path.name}: no face detected")
            continue

        known_encodings.append(encodings[0])
        known_names.append(image_path.stem)

    if not known_encodings:
        raise ValueError(
            f"No valid employee faces found in {employees_dir}. "
            "Add clear face images named like Abdul.jpg."
        )

    return known_encodings, known_names


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


def recognize_from_frame(frame, known_encodings, known_names, tolerance=0.6):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = []
    for encoding, location in zip(face_encodings, face_locations):
        distances = face_recognition.face_distance(known_encodings, encoding)
        best_match_index = int(np.argmin(distances))
        best_distance = float(distances[best_match_index])

        if best_distance <= tolerance:
            name = known_names[best_match_index]
            status = "recognized"
        else:
            name = "Unknown"
            status = "unknown"

        results.append(
            {
                "name": name,
                "status": status,
                "distance": best_distance,
                "location": location,
            }
        )

    return results


def run_attendance(
    employees_dir="Employees",
    attendance_file="attendance/register.csv",
    camera_index=0,
    tolerance=0.6,
):
    known_encodings, known_names = load_known_faces(employees_dir)
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not capture.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions or camera index.")

    print("Camera started. Press 'q' to quit.")

    while True:
        success, frame = capture.read()
        if not success:
            print("Could not read frame from webcam")
            break

        results = recognize_from_frame(frame, known_encodings, known_names, tolerance)
        for result in results:
            color = (0, 255, 0) if result["status"] == "recognized" else (0, 0, 255)
            label = f"{result['name']} ({result['distance']:.2f})"
            draw_face_box(frame, result["location"], label, color=color)

            if result["status"] == "recognized":
                if mark_attendance(result["name"], attendance_file):
                    print(f"Attendance marked for {result['name']}")

        cv2.imshow("Face Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Face recognition attendance system.")
    parser.add_argument("--employees-dir", default="Employees", help="Folder of known employee face images")
    parser.add_argument("--attendance-file", default="attendance/register.csv", help="CSV output path")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index")
    parser.add_argument("--tolerance", type=float, default=0.6, help="Lower is stricter")
    return parser.parse_args()


def main():
    args = parse_args()
    run_attendance(
        employees_dir=args.employees_dir,
        attendance_file=args.attendance_file,
        camera_index=args.camera_index,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()
