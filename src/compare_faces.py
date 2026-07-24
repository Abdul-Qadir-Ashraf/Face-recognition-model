import argparse

import cv2
import face_recognition

from src.face_utils import draw_face_box, find_single_face_encoding, load_rgb_image


def compare_faces(reference_image_path, test_image_path, tolerance=0.6, show=True):
    reference_image = load_rgb_image(reference_image_path)
    test_image = load_rgb_image(test_image_path)

    reference_encoding, reference_location = find_single_face_encoding(
        reference_image,
        reference_image_path,
    )
    test_encoding, test_location = find_single_face_encoding(test_image, test_image_path)

    matches = face_recognition.compare_faces(
        [reference_encoding],
        test_encoding,
        tolerance=tolerance,
    )
    distance = face_recognition.face_distance([reference_encoding], test_encoding)[0]
    is_match = bool(matches[0])

    reference_display = cv2.cvtColor(reference_image, cv2.COLOR_RGB2BGR)
    test_display = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

    draw_face_box(reference_display, reference_location, "Reference")
    label = f"{'MATCH' if is_match else 'NO MATCH'} | distance: {distance:.3f}"
    draw_face_box(test_display, test_location, label, color=(0, 180, 255))

    if show:
        cv2.imshow("Reference Image", reference_display)
        cv2.imshow("Test Image", test_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {"match": is_match, "distance": float(distance), "tolerance": tolerance}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare faces from two images.")
    parser.add_argument("--reference", default="PictureC.jpg", help="Known/reference face image")
    parser.add_argument("--test", default="PictureB.jpg", help="Test face image")
    parser.add_argument("--tolerance", type=float, default=0.6, help="Lower is stricter")
    parser.add_argument("--no-display", action="store_true", help="Run without opening image windows")
    return parser.parse_args()


def main():
    args = parse_args()
    result = compare_faces(
        args.reference,
        args.test,
        tolerance=args.tolerance,
        show=not args.no_display,
    )

    print(f"Match: {result['match']}")
    print(f"Distance: {result['distance']:.3f}")
    print(f"Tolerance: {result['tolerance']}")


if __name__ == "__main__":
    main()
