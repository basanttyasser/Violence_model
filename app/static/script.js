// Tab Elements
const tabWebcam = document.getElementById('tabWebcam');
const tabUpload = document.getElementById('tabUpload');
const webcamMode = document.getElementById('webcamMode');
const uploadMode = document.getElementById('uploadMode');

// Webcam Elements
const video = document.getElementById('webcam');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusOverlayWebcam = document.getElementById('status-overlay-webcam');

// Upload Elements
const videoUpload = document.getElementById('videoUpload');
const uploadedVideo = document.getElementById('uploadedVideo');
const analyzeBtn = document.getElementById('analyzeBtn');
const statusOverlayUpload = document.getElementById('status-overlay-upload');

// Shared Log Element
const logText = document.getElementById('logText');

let stream;
let isDetecting = false;
let ws; // WebSocket for live stream
let frameInterval; 

// ==========================================
// TAB SWITCHING LOGIC
// ==========================================
tabWebcam.addEventListener('click', () => {
    tabWebcam.classList.add('active');
    tabUpload.classList.remove('active');
    webcamMode.style.display = 'block';
    uploadMode.style.display = 'none';
    setupCamera();
    logText.innerText = "Switched to Webcam mode.";
});

tabUpload.addEventListener('click', () => {
    tabUpload.classList.add('active');
    tabWebcam.classList.remove('active');
    webcamMode.style.display = 'none';
    uploadMode.style.display = 'block';
    stopCamera(); 
    logText.innerText = "Switched to Upload mode. Please select a video.";
});

// ==========================================
// WEBCAM LOGIC (Continuous Real-Time via WebSockets)
// ==========================================
async function setupCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
    } catch (err) {
        logText.innerText = "Error accessing webcam: " + err.message;
    }
}

function stopCamera() {
    if (isDetecting) stopBtn.click();
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }
}

function startContinuousDetection() {
    // Dynamically get correct WebSocket URL (handles localhost and ngrok automatically)
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/live`;
    
    ws = new WebSocket(wsUrl);

    // Create a hidden canvas to grab frames
    const canvas = document.createElement('canvas');
    canvas.width = 224; 
    canvas.height = 224;
    const ctx = canvas.getContext('2d');

    ws.onopen = () => {
        logText.innerText = "Live connection established. Buffering frames...";
        statusOverlayWebcam.innerText = "BUFFERING...";
        statusOverlayWebcam.className = "status warning";

        // Capture and send a frame 10 times a second (10 FPS)
        frameInterval = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN && video.readyState === video.HAVE_ENOUGH_DATA) {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const base64Frame = canvas.toDataURL('image/jpeg', 0.6); // Compress slightly for speed
                ws.send(base64Frame);
            }
        }, 100); 
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.status === "success") {
            if (data.is_violence) {
                statusOverlayWebcam.innerText = `VIOLENCE DETECTED (${data.confidence}%)`;
                statusOverlayWebcam.className = "status danger";
                logText.innerText = `[${new Date().toLocaleTimeString()}] ALERT: Violence detected! Confidence: ${data.confidence}%`;
                logText.style.color = "#ff4c4c";
            } else {
                statusOverlayWebcam.innerText = "SAFE";
                statusOverlayWebcam.className = "status safe";
                logText.innerText = `[${new Date().toLocaleTimeString()}] Normal activity.`;
                logText.style.color = "#a4a4a4";
            }
        }
    };

    ws.onclose = () => {
        clearInterval(frameInterval);
    };
}

startBtn.addEventListener('click', () => {
    isDetecting = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    startContinuousDetection();
});

stopBtn.addEventListener('click', () => {
    isDetecting = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    logText.innerText = "Live detection stopped.";
    if (ws) ws.close();
    if (frameInterval) clearInterval(frameInterval);
    
    statusOverlayWebcam.innerText = "SAFE";
    statusOverlayWebcam.className = "status safe";
});

// ==========================================
// UPLOAD LOGIC (Standard Video File processing)
// ==========================================
let selectedFile = null;

videoUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        selectedFile = file;
        const fileURL = URL.createObjectURL(file);
        uploadedVideo.src = fileURL;
        analyzeBtn.disabled = false;
        statusOverlayUpload.style.display = 'none';
        logText.innerText = `Loaded: ${file.name}. Ready to analyze.`;
        logText.style.color = "#a4a4a4";
    }
});

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    statusOverlayUpload.style.display = 'block';
    statusOverlayUpload.innerText = "ANALYZING...";
    statusOverlayUpload.className = "status warning";
    logText.innerText = `Analyzing ${selectedFile.name}... This may take a moment.`;
    logText.style.color = "#ff9800";

    const formData = new FormData();
    formData.append("file", selectedFile, "video.mp4");

    try {
        const response = await fetch("/predict", { method: "POST", body: formData });
        const data = await response.json();
        
        if (data.status === "success") {
            if (data.is_violence) {
                statusOverlayUpload.innerText = `VIOLENCE (${data.confidence}%)`;
                statusOverlayUpload.className = "status danger";
            } else {
                statusOverlayUpload.innerText = `SAFE (${data.confidence}%)`;
                statusOverlayUpload.className = "status safe";
            }
            logText.innerText = "Analysis complete.";
        } else {
            logText.innerText = "Server Error: " + data.message;
        }
    } catch (error) {
        console.error("Error:", error);
        logText.innerText = "Connection error.";
    }
    analyzeBtn.disabled = false;
});

// Start camera on load
window.onload = setupCamera;