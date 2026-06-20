import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load images safely
def load_images(folder):
    images = []
    labels = []

    categories = ["genuine", "forged"]

    for label, category in enumerate(categories):
        path = os.path.join(folder, category)

        if not os.path.exists(path):
            print(f"Folder missing: {path}")
            continue

        for file in os.listdir(path):
            img_path = os.path.join(path, file)

            img = cv2.imread(img_path, 0)

            if img is None:
                print("Skipping:", img_path)
                continue

            img = cv2.resize(img, (100, 100))
            img = img.flatten()

            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)


# Load dataset
X, y = load_images("dataset")

print("Total images:", len(X))
print("Unique labels:", set(y))

# Check minimum classes
if len(set(y)) < 2:
    print("ERROR: Need both genuine and forged images!")
    exit()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

# ✅ Accuracy in percentage (2 decimal places)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Save model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")


