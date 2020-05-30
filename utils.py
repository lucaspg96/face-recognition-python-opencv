import numpy as np
import cv2
import os

face_haar = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


def face_detection(img):
    gray_scale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_haar.detectMultiScale(img,
                                       scaleFactor=1.3,
                                       minNeighbors=3)
    return faces, gray_scale_img


def load_training_data(directory):
    faces = []
    faces_ids = []

    for path, subdirnames, filenames in os.walk(directory):
        for file in filenames:
            if file.startswith("."):
                print("Skipping system file: "+file)
                continue

            face_id = os.path.basename(path)
            img_path = os.path.join(path, file)

            img = cv2.imread(img_path)
            if img is None:
                print("File not load properly: {} . Skipping...".format(img_path))
                continue

            faces_rect, gray_image = face_detection(img)
            (x, y, w, h) = faces_rect[0]
            face = gray_image[y:y+w, x:x+h]

            faces.append(face)
            faces_ids.append(int(face_id))

    return faces, np.array(faces_ids)


def train_classifier(faces, ids):
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faces, ids)
    return model


def draw_rect(img, rect):
    (x, y, w, h) = rect
    return cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)


def put_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                1, (255, 255, 255), 1)

# (faces, ids) = load_training_data("train_images/captures")

# print(len(faces), len(ids))
