import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.image_utils import decode_image, detect_faces
from utils.emotion_utils import predict_emotion

app = FastAPI(title="Face Detection & Emotion Recognition API")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# --- Serve frontend at /frontend --- (important: avoid conflict with API routes)
FRONTEND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.exists(FRONTEND_PATH):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")
else:
    print("Warning: frontend folder not found, skipping static mount")

# --- Run via Uvicorn on Render ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
