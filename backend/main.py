import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.image_utils import decode_image, detect_faces
from utils.emotion_utils import predict_emotion

app = FastAPI(title="Face Detection & Emotion Recognition API")

# --- CORS (for frontend hosted separately) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" with your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Serve the frontend (folder at same level as backend) ---
FRONTEND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")

# --- API endpoints ---
ALLOWED_TASKS = {"detection", "emotion"}

class ImageRequest(BaseModel):
    image: str   # base64 string
    task: str    # "detection" or "emotion"

@app.post("/process")
def process_image(data: ImageRequest):
    # Validate task
    if data.task not in ALLOWED_TASKS:
        raise HTTPException(status_code=400, detail="Invalid task")

    # Decode base64 image
    img = decode_image(data.image)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # --- Face detection ---
    if data.task == "detection":
        boxes = detect_faces(img)
        return {
            "task": "detection",
            "faces": boxes
        }

    # --- Emotion recognition ---
    if data.task == "emotion":
        emotion = predict_emotion(img)
        return {
            "task": "emotion",
            "emotion": emotion
        }

# Optional health check
@app.get("/health")
def health():
    return {"status": "backend running"}
