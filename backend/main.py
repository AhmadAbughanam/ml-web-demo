import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from utils.image_utils import decode_image, detect_faces
from utils.emotion_utils import predict_emotion

app = FastAPI(title="Face Detection & Emotion Recognition API")

# --- CORS - MORE PERMISSIVE FOR DEPLOYMENT ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_PATH = os.path.join(BASE_DIR, "..", "frontend")

# --- API Models ---
ALLOWED_TASKS = {"detection", "emotion"}

class ImageRequest(BaseModel):
    image: str   # base64 string
    task: str    # "detection" or "emotion"

# --- API Endpoints (MUST BE BEFORE STATIC FILES) ---

@app.get("/health")
def health():
    return {"status": "backend running"}

@app.post("/process")
async def process_image(data: ImageRequest):
    """
    Process image for face detection or emotion recognition
    """
    if data.task not in ALLOWED_TASKS:
        raise HTTPException(status_code=400, detail="Invalid task. Must be 'detection' or 'emotion'")
    
    img = decode_image(data.image)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    if data.task == "detection":
        boxes = detect_faces(img)
        return JSONResponse(content={"task": "detection", "faces": boxes})
    
    if data.task == "emotion":
        emotion = predict_emotion(img)
        confidence = "High" if emotion != "no_face" else "N/A"
        return JSONResponse(content={
            "task": "emotion", 
            "emotion": emotion,
            "confidence": confidence
        })

# --- Serve Static Frontend Files ---
@app.get("/")
async def read_root():
    """Serve the main HTML file"""
    index_path = os.path.join(FRONTEND_PATH, "index.html")
    if os.path.exists(index_path):
        return FileResponse(
            index_path,
            media_type="text/html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    else:
        raise HTTPException(status_code=404, detail="Frontend index.html not found")

# Mount static files AFTER API routes
if os.path.exists(FRONTEND_PATH):
    app.mount(
        "/static", 
        StaticFiles(directory=FRONTEND_PATH, html=True), 
        name="static"
    )
    print(f"✓ Frontend mounted from: {FRONTEND_PATH}")
else:
    print(f"⚠ Warning: Frontend directory not found at {FRONTEND_PATH}")

# --- Catch-all for SPA routing (MUST BE LAST) ---
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """
    Serve index.html for any unmatched routes (SPA support)
    This enables client-side routing
    """
    # Don't intercept API routes
    if full_path.startswith(("process", "health", "static", "api")):
        raise HTTPException(status_code=404, detail="Not found")
    
    # Check if requesting a specific file
    file_path = os.path.join(FRONTEND_PATH, full_path)
    if os.path.isfile(file_path):
        return FileResponse(
            file_path,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    
    # Otherwise serve index.html (for SPA routing)
    index_path = os.path.join(FRONTEND_PATH, "index.html")
    if os.path.exists(index_path):
        return FileResponse(
            index_path,
            media_type="text/html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    else:
        raise HTTPException(status_code=404, detail="Frontend not found")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
