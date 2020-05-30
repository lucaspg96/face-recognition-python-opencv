import numpy as np
import cv2
import os

import utils as ut

print("Loading images...")
faces, ids = ut.load_training_data("train_images")
print("{} images loaded".format(len(faces)))

print("Starting training...")
model = ut.train_classifier(faces, ids)
print("Model trained. Saving...")
model.save("model.yml")
print("Model saved")
