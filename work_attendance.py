import face_recognition as fr
import cv2
import os

import numpy

# create database
path = 'Employees'
my_image = []
employees_name = []
employees_list = os.listdir(path)

# Image list creation
for employees in employees_list:
    this_image = cv2.imread(f'{path}//{employees}')
    this_image = cv2.cvtColor(this_image, cv2.COLOR_BGR2RGB)
    my_image.append(this_image)
    employees_name.append(employees.replace('.jpg', ''))


# encoding image
def encode_image(images):
    encoded_list = []
    for image in images:
        image_code = fr.face_encodings(image)[0]
        encoded_list.append(image_code)

    return encoded_list


encode_image_list = encode_image(my_image)

# record attendance
def record_attendance(person):
    f = open('register.csv', 'r+')
    data_list = f.readlines()
    register_names = []

    for line in data_list:
        newcomer = line.split(',')
        register_names.append(newcomer[0])

    if person not in register_names:
        right_now = datetime.now()
        string_right_now = right_now.strftime('%H:%M:%S')
        f.writelines(f'\n{person},{string_right_now}')


# video capture
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# read the image
success, image = capture.read()

# matching
if not success:
    print("Oops failed to capture the photo!")
else:
    # recognise face
    captured_face = fr.face_locations(image)

    # encode captured face
    image_code = fr.face_encodings(image, captured_face)

    # compare
    for face, location in zip(image_code, captured_face):
        matches = fr.compare_faces(encode_image_list, face)
        distance = fr.face_distance(encode_image_list, face)

        match_index = numpy.argmin(distance)

        if distance[match_index].round(2) > 0.6:
            print("Does not match")
            print(distance)
        else:
            employe = employees_name[match_index]

            y1, x2, x1, y2 = location

            # setting rectangle
            cv2.rectangle(image,
                          (x1, y1),
                          (x2, y2),
                          (0, 255, 0),
                          2)
            cv2.putText(image,
                        employe,
                        (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0), 2)
            cv2.imshow('My name', image)
            cv2.waitKey(0)
