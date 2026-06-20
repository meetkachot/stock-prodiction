import cv2
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")

def predict_image(path):
    img = cv2.imread(path, 0)

    if img is None:
        print("Image not found!")
        return

    img = cv2.resize(img, (100, 100))
    img = img.flatten().reshape(1, -1)

    result = model.predict(img)

    if result[0] == 0:
        print("Genuine Signature ✅")
    else:
        print("Forged Signature ❌")


# Test image path
predict_image("dataset/genuine/g1.jpg")
