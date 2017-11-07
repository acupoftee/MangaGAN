import cv2
import numpy as np

# add cascades
face_cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect_anine_face(image):
    img = cv2.imread(image)

    # convert image to grayscale for easier processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_color)

    return faces
