import face_recognition as fr
import cv2


# load images
control_picture = fr.load_image_file('PictureC.jpg')
test_picture = fr.load_image_file('PictureB.jpg')

# Transform image into RGB
control_picture = cv2.cvtColor(control_picture, cv2.COLOR_BGR2RGB)
test_picture = cv2.cvtColor(test_picture, cv2.COLOR_BGR2RGB)

# image details
coded_face_A = fr.face_encodings(control_picture)[0]
face_location_A = fr.face_locations(control_picture)[0]

coded_face_B = fr.face_encodings(test_picture)[0]
face_location_B = fr.face_locations(test_picture)[0]

# show rectangles
cv2.rectangle(control_picture,
              (face_location_A[3], face_location_A[0]),
              (face_location_A[1], face_location_A[2]),
              (0, 255, 0),
              2)
cv2.rectangle(test_picture,
              (face_location_B[3], face_location_B[0]),
              (face_location_B[1], face_location_B[2]),
              (0, 255, 0),
              2)

# face compare
result = fr.compare_faces([coded_face_A], coded_face_B)

# measurement of distance
distance = fr.face_distance([coded_face_A], coded_face_B)

# show results
cv2.putText(test_picture, f'{result} {distance}',
            (50, 50), cv2.FONT_HERSHEY_COMPLEX,
            1, (0, 255, 0),
            2)

# show image
cv2.imshow('My Control Picture', control_picture)
cv2.imshow('My test picture', test_picture)


# Hold screen
cv2.waitKey(0)
