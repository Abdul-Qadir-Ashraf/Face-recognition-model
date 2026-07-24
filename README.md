# Face Recognition Attendance System 2.0

A Python-based face recognition project that compares faces from images and marks attendance from webcam input.

This upgraded version is structured as a portfolio-ready computer vision project with safer error handling, command-line usage, reusable modules, and clear setup instructions.

## Features

- Compare two face images and display match confidence
- Recognize known people from a webcam feed
- Mark attendance once per recognized person in a CSV file
- Handle missing images, missing faces, and webcam errors gracefully
- Store known faces using a simple `Employees/` image folder

## Project Structure

```text
.
├── src/
│   ├── attendance_system.py
│   ├── compare_faces.py
│   └── face_utils.py
├── Employees/
│   └── .gitkeep
├── attendance/
│   └── .gitkeep
├── Face_recognition.py
├── work_attendance_real.py
├── PictureA.jpg
├── PictureB.jpg
├── PictureC.jpg
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> On Windows, `face-recognition` may require Visual Studio Build Tools and CMake because it depends on `dlib`.

## Compare Two Faces

```bash
python Face_recognition.py --reference PictureC.jpg --test PictureB.jpg
```

Run without opening image windows:

```bash
python Face_recognition.py --reference PictureC.jpg --test PictureB.jpg --no-display
```

## Attendance System

1. Add clear face images inside `Employees/`.
2. Name each image after the person, for example:

```text
Employees/
├── Abdul.jpg
├── Sara.png
└── Rahul.jpeg
```

3. Start webcam attendance:

```bash
python work_attendance_real.py
```

4. Press `q` to stop the webcam window.

Attendance is saved to:

```text
attendance/register.csv
```

## Commands

```bash
python Face_recognition.py --help
python work_attendance_real.py --help
```

## Run Tests

```bash
python -m pytest
```

## How It Works

1. Known faces are loaded from the `Employees/` folder.
2. Each known face is encoded using the `face_recognition` library.
3. Webcam frames are scanned for faces.
4. The closest face encoding is selected using face distance.
5. If the distance is below the tolerance threshold, attendance is marked.

Default tolerance is `0.6`. Lower values are stricter.

## Limitations

- Accuracy depends on lighting, image quality, camera angle, and face visibility.
- This is a learning/portfolio project, not a production biometric security system.
- For real deployment, add consent handling, privacy controls, anti-spoofing, and secure storage.

## Future Improvements

- Add a Streamlit or Django dashboard
- Add employee registration from webcam
- Add attendance reports and charts
- Add more tests for image loading and recognition edge cases
- Add model evaluation with multiple images per person
- Add face anti-spoofing checks
