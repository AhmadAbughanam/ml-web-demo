// DOM Elements
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const resultCanvas = document.getElementById("resultCanvas");
const captureBtn = document.getElementById("captureBtn");
const detectBtn = document.getElementById("detectBtn");
const emotionBtn = document.getElementById("emotionBtn");
const emotionBadge = document.getElementById("emotionBadge");
const emptyState = document.getElementById("emptyState");
const faceCountEl = document.getElementById("faceCount");
const confidenceEl = document.getElementById("confidence");
const processingTimeEl = document.getElementById("processingTime");
const navItems = document.querySelectorAll(".nav-item");

let capturedImage = null;
let processingStartTime = 0;

// Initialize webcam
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, facingMode: "user" }
    });
    video.srcObject = stream;
    video.play();
    console.log("Camera initialized successfully");
  } catch (err) {
    console.error("Camera error:", err);
    alert("Unable to access camera. Please check permissions.");
  }
}

// Navigation handling
navItems.forEach((item) => {
  item.addEventListener("click", () => {
    navItems.forEach((nav) => nav.classList.remove("active"));
    item.classList.add("active");
    
    const step = item.dataset.step;
    if (step === "capture") {
      captureBtn.scrollIntoView({ behavior: "smooth", block: "center" });
    } else if (step === "detect" && !detectBtn.disabled) {
      detectBtn.scrollIntoView({ behavior: "smooth", block: "center" });
    } else if (step === "emotion" && !emotionBtn.disabled) {
      emotionBtn.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  });
});

// Capture photo
captureBtn.addEventListener("click", () => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  capturedImage = canvas.toDataURL("image/jpeg", 0.9);

  // Enable analysis buttons
  detectBtn.disabled = false;
  emotionBtn.disabled = false;

  // Clear previous results
  emotionBadge.textContent = "";
  emptyState.classList.remove("hidden");
  faceCountEl.textContent = "0";
  confidenceEl.textContent = "--";
  processingTimeEl.textContent = "--";
  
  const rctx = resultCanvas.getContext("2d");
  rctx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);

  // Visual feedback
  const originalHTML = captureBtn.innerHTML;
  captureBtn.innerHTML = `
    <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <polyline points="20 6 9 17 4 12"></polyline>
    </svg>
    Captured!
  `;
  captureBtn.style.background = "linear-gradient(135deg, #10b981, #059669)";

  setTimeout(() => {
    captureBtn.innerHTML = originalHTML;
    captureBtn.style.background = "";
  }, 2000);

  console.log("Photo captured successfully");
});

// Send image to backend
async function sendToBackend(task, btn) {
  if (!capturedImage) {
    alert("Please capture a photo first!");
    return;
  }

  processingStartTime = performance.now();
  const originalHTML = btn.innerHTML;
  
  btn.disabled = true;
  btn.innerHTML = `
    <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="12" cy="12" r="10"></circle>
      <path d="M12 6v6l4 2"></path>
    </svg>
    <span>Processing...</span>
  `;

  try {
    const response = await fetch("/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: capturedImage, task }),
    });

    const processingTime = ((performance.now() - processingStartTime) / 1000).toFixed(2);
    processingTimeEl.textContent = `${processingTime}s`;

    if (!response.ok) {
      const err = await response.json();
      console.error("Backend error:", err);
      alert("Error: " + (err.detail || "Unknown error occurred"));
      return;
    }

    const data = await response.json();

    if (task === "detection") {
      displayFaces(data.faces);
      faceCountEl.textContent = data.faces.length;
      confidenceEl.textContent = data.faces.length > 0 ? "High" : "N/A";
      
      // Success feedback
      btn.innerHTML = `
        <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
        <span>Detection Complete</span>
      `;
      
      setTimeout(() => {
        btn.innerHTML = originalHTML;
        btn.disabled = false;
      }, 2000);
      
    } else if (task === "emotion") {
      emotionBadge.textContent = data.emotion;
      confidenceEl.textContent = data.confidence || "N/A";
      
      // Success feedback
      btn.innerHTML = `
        <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
        <span>Analysis Complete</span>
      `;
      
      setTimeout(() => {
        btn.innerHTML = originalHTML;
        btn.disabled = false;
      }, 2000);
    }

  } catch (err) {
    console.error("Network error:", err);
    alert("Connection error. Please check if the backend is running.");
    processingTimeEl.textContent = "Error";
  } finally {
    if (btn.disabled) {
      btn.disabled = false;
      btn.innerHTML = originalHTML;
    }
  }
}

// Display face detection results
function displayFaces(faces) {
  const img = new Image();
  img.src = capturedImage;

  img.onload = () => {
    emptyState.classList.add("hidden");
    
    resultCanvas.width = img.width;
    resultCanvas.height = img.height;
    const ctx = resultCanvas.getContext("2d");
    
    // Draw image
    ctx.drawImage(img, 0, 0);

    if (faces.length === 0) {
      // Overlay message for no faces
      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      ctx.fillRect(0, 0, resultCanvas.width, resultCanvas.height);
      
      ctx.fillStyle = "white";
      ctx.font = "24px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No faces detected", resultCanvas.width / 2, resultCanvas.height / 2);
      return;
    }

    // Draw face boxes
    faces.forEach((face, index) => {
      const padding = 10;
      const x = face.x - padding;
      const y = face.y - padding;
      const w = face.w + padding * 2;
      const h = face.h + padding * 2;

      // Gradient box
      const gradient = ctx.createLinearGradient(x, y, x + w, y + h);
      gradient.addColorStop(0, "#6366f1");
      gradient.addColorStop(1, "#8b5cf6");

      // Shadow
      ctx.strokeStyle = "rgba(0, 0, 0, 0.3)";
      ctx.lineWidth = 6;
      ctx.strokeRect(x + 3, y + 3, w, h);

      // Main box
      ctx.strokeStyle = gradient;
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, w, h);

      // Corner accents
      const cornerSize = 25;
      ctx.lineWidth = 5;
      ctx.strokeStyle = "#8b5cf6";

      // Top-left
      ctx.beginPath();
      ctx.moveTo(x, y + cornerSize);
      ctx.lineTo(x, y);
      ctx.lineTo(x + cornerSize, y);
      ctx.stroke();

      // Top-right
      ctx.beginPath();
      ctx.moveTo(x + w - cornerSize, y);
      ctx.lineTo(x + w, y);
      ctx.lineTo(x + w, y + cornerSize);
      ctx.stroke();

      // Bottom-left
      ctx.beginPath();
      ctx.moveTo(x, y + h - cornerSize);
      ctx.lineTo(x, y + h);
      ctx.lineTo(x + cornerSize, y + h);
      ctx.stroke();

      // Bottom-right
      ctx.beginPath();
      ctx.moveTo(x + w - cornerSize, y + h);
      ctx.lineTo(x + w, y + h);
      ctx.lineTo(x + w, y + h - cornerSize);
      ctx.stroke();

      // Label
      ctx.fillStyle = gradient;
      ctx.fillRect(x, y - 30, 80, 30);
      ctx.fillStyle = "white";
      ctx.font = "bold 14px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(`Face ${index + 1}`, x + 40, y - 15);
    });

    console.log(`Rendered ${faces.length} face(s)`);
  };

  img.onerror = () => {
    console.error("Failed to load captured image");
    alert("Error displaying results");
  };
}

// Button event listeners
detectBtn.addEventListener("click", () => sendToBackend("detection", detectBtn));
emotionBtn.addEventListener("click", () => sendToBackend("emotion", emotionBtn));

// Initialize
initCamera();