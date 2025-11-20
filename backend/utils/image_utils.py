import os
import base64
import cv2
import numpy as np
import re

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_PATH = os.path.join(BASE_DIR, "..", "models", "haarcascade_frontalface_default.xml")

# Load Haar Cascade once
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)
if FACE_CASCADE.empty():
    raise FileNotFoundError(f"Haar cascade not found at {CASCADE_PATH}")


def decode_image(base64_string: str):
    """
    Convert base64 image string → NumPy BGR image.
    Accepts: data:image/jpeg;base64,XXXX...
    """
    try:
        # Remove prefix
        base64_string = re.sub(r"^data:image/.+;base64,", "", base64_string)

        # Base64 → bytes
        img_bytes = base64.b64decode(base64_string)

        # Bytes → NumPy array
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)

        # NumPy → OpenCV BGR image
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
        return img

    except Exception as e:
        print("Decode Error:", e)
        return None


def detect_faces(img):
    """
    Detect faces and return bounding boxes as a list of dicts:
    [{"x": , "y": , "w": , "h": }, ...]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        print("No faces detected")
        return []

    boxes = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in faces]
    print(f"Detected {len(boxes)} face(s): {boxes}")
    return boxes
