import os
import cv2
import numpy as np
import tensorflow as tf

# Emotion labels used by Mini-Xception on FER2013
EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]

# Paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "emotion_model.tflite")
CASCADE_PATH = os.path.join(BASE_DIR, "..", "models", "haarcascade_frontalface_default.xml")

# Load TFLite model
INTERPRETER = tf.lite.Interpreter(model_path=MODEL_PATH)
INTERPRETER.allocate_tensors()
input_details = INTERPRETER.get_input_details()
output_details = INTERPRETER.get_output_details()

# Load Haar cascade
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)
if FACE_CASCADE.empty():
    raise FileNotFoundError(f"Haar cascade not found at {CASCADE_PATH}")


def preprocess_face(img):
    """
    Convert face to grayscale, resize to 48x48, normalize, and expand dims.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized.astype("float32") / 255.0
    expanded = np.expand_dims(normalized, axis=0)
    expanded = np.expand_dims(expanded, axis=-1)
    return expanded


def predict_emotion(img):
    """
    Detects the largest face in the image and predicts its emotion.
    Returns "no_face" if none found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        print("No face detected")
        return "no_face"

    # Pick largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]

    input_tensor = preprocess_face(face_img)
    print("Input tensor shape:", input_tensor.shape)

    # Run inference
    INTERPRETER.set_tensor(input_details[0]["index"], input_tensor)
    INTERPRETER.invoke()
    output = INTERPRETER.get_tensor(output_details[0]["index"])[0]

    emotion_id = int(np.argmax(output))
    emotion_label = EMOTION_LABELS[emotion_id]
    print("Predicted emotion:", emotion_label)

    return emotion_label
