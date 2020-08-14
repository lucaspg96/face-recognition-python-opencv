import numpy as np
import cv2
import os
from time import sleep

import utils as ut

try:
    model.read(r'model.yml')
    model = cv2.face.LBPHFaceRecognizer_create()
except Exception as e:
    model = None

video = cv2.VideoCapture(0)

names = {
    1: ""
}

while True:
    ret, img = video.read()
    faces, gray_img = ut.face_detection(img)

    for face in faces:
        (x, y, w, h) = face

        face_gray = gray_img[y:y+w, x:x+h]
        if model is None:
            label, confidence = ("Unknown", 1)
        else:
            label, confidence = model.predict(face_gray)
        name = names.get(label, str(label))
        print("label: {}, confidence: {}".
              format(name, confidence))
        ut.draw_rect(img, face)
        if confidence > 0.8:
            ut.put_text(img, name, x, y)

    resized_img = cv2.resize(img, (1000, 700))

    cv2.imshow('Video', resized_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
