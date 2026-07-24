from pathlib import Path

import cv2
import face_recognition


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_rgb_image(image_path):
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = face_recognition.load_image_file(str(path))
    if image is None:
        raise ValueError(f"Could not load image: {path}")

    return image


def find_single_face_encoding(image, label):
    locations = face_recognition.face_locations(image)
    if not locations:
        raise ValueError(f"No face detected in {label}")

    encodings = face_recognition.face_encodings(image, locations)
    if not encodings:
        raise ValueError(f"Could not encode face in {label}")

    return encodings[0], locations[0]


def draw_face_box(image, location, label, color=(0, 255, 0)):
    top, right, bottom, left = location
    cv2.rectangle(image, (left, top), (right, bottom), color, 2)
    cv2.rectangle(image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    cv2.putText(
        image,
        label,
        (left + 6, bottom - 6),
        cv2.FONT_HERSHEY_COMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )


def iter_image_files(directory):
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    for image_path in sorted(path.iterdir()):
        if image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            yield image_path
