import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from utils.image_utils import decode_image, detect_faces
from utils.emotion_utils import predict_emotion

app = FastAPI(title="Face Detection & Emotion Recognition API")

# --- CORS (allow frontend from anywhere; adjust in production) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Serve frontend at root ---
FRONTEND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.exists(FRONTEND_PATH):
    app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")
else:
    print("Warning: frontend folder not found, skipping static mount")

# --- API endpoints ---
ALLOWED_TASKS = {"detection", "emotion"}

class ImageRequest(BaseModel):
    image: str   # base64 string
    task: str    # "detection" or "emotion"

@app.post("/process")
def process_image(data: ImageRequest):
    if data.task not in ALLOWED_TASKS:
        raise HTTPException(status_code=400, detail="Invalid task")

    img = decode_image(data.image)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    if data.task == "detection":
        boxes = detect_faces(img)
        return {"task": "detection", "faces": boxes}

    if data.task == "emotion":
        emotion = predict_emotion(img)
        return {"task": "emotion", "emotion": emotion}

@app.get("/health")
def health():
    return {"status": "backend running"}

# --- Catch-all for frontend routes (avoids conflicts with API paths) ---
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    # Protect API routes
    if full_path.startswith("process") or full_path.startswith("health"):
        raise HTTPException(status_code=404)

    index_path = os.path.join(FRONTEND_PATH, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        raise HTTPException(status_code=404, detail="Frontend not found")
