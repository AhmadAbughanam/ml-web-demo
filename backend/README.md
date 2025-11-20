# Backend — Face Detection + Emotion Recognition API

This backend provides two lightweight ML services:

- **Face Detection** (OpenCV Haar Cascade)
- **Emotion Recognition** (Mini-Xception TFLite)

It receives a base64 image from the frontend and returns:

- bounding boxes for face detection
- emotion label for emotion recognition

---

## 🚀 1. How to Run Locally

### Install dependencies

```bash
pip install -r requirements.txt
Start the server
bash
Copy code
uvicorn main:app --reload --host 0.0.0.0 --port 8000
Test in browser
Open:

cpp
Copy code
http://127.0.0.1:8000
Test with cURL
bash
Copy code
curl -X POST http://127.0.0.1:8000/process \
    -H "Content-Type: application/json" \
    -d "{\"image\":\"BASE64_STRING\",\"task\":\"detection\"}"
📂 2. Folder Structure
css
Copy code
backend/
│
├── main.py
├── requirements.txt
│
├── models/
│   ├── emotion_model.tflite
│   └── haarcascade_frontalface_default.xml
│
└── utils/
    ├── image_utils.py
    └── emotion_utils.py
🧠 3. API Route
POST /process
Request:

json
Copy code
{
  "image": "data:image/jpeg;base64,...",
  "task": "detection"
}
Response (detection):

json
Copy code
{
  "task": "detection",
  "faces": [
    {"x":120, "y":80, "w":150, "h":150}
  ]
}
Response (emotion):

json
Copy code
{
  "task": "emotion",
  "emotion": "happy"
}
📦 4. Model Files
Place both of these inside backend/models/:

haarcascade_frontalface_default.xml
→ from OpenCV:
https://github.com/opencv/opencv/blob/master/data/haarcascades/

emotion_model.tflite
→ lightweight Mini-Xception TFLite model
(I'll provide a ready-to-download link or export instructions)

☁️ 5. Deploying to Render (Free Tier)
Steps
Push backend folder to GitHub

On Render:

Create Web Service

Runtime: Python

Build Command:

bash
Copy code
pip install -r backend/requirements.txt
Start Command:

nginx
Copy code
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
Important
Render free tier sleeps after 15 minutes.
Cold start = 10–20 seconds due to model loading.

🟢 Status Endpoint
Check if backend is alive:

sql
Copy code
GET /
```
