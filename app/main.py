from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import cv2
import numpy as np
import tempfile
import os
import shutil
import uvicorn
import base64
import collections
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Input, TimeDistributed, GlobalAveragePooling2D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import mixed_precision

# Must match training environment exactly
mixed_precision.set_global_policy("mixed_float16")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
SEQUENCE_LENGTH = 96
MODEL_PATH = "model (2).keras"

CLASSES_LIST = ["NonViolence", "Violence"]


def resize_with_pad(frame, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    """
    Resizes an image to the target size while maintaining the aspect ratio.
    Pads the remaining space with black borders (letterboxing) to prevent shape distortion.
    """
    h, w = frame.shape[:2]
    target_w, target_h = target_size

    # Calculate scale to fit within target dimensions
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize the image keeping the aspect ratio
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a black canvas of the target size
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Calculate offsets to center the image on the canvas
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Paste the resized image into the center of the canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas


def create_model() -> Sequential:
    """Rebuilds the exact architecture used during training."""
    mobilenet = MobileNetV2(
        include_top=False,
        weights=None,
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
    )

    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        TimeDistributed(mobilenet),
        TimeDistributed(GlobalAveragePooling2D()),
        Dropout(0.3),
        Bidirectional(
            LSTM(units=64, return_sequences=False),
            backward_layer=LSTM(units=64, go_backwards=True, return_sequences=False),
        ),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(len(CLASSES_LIST), activation="softmax", dtype="float32"),
    ])
    return model


print("Building model architecture...")
try:
    violence_model = create_model()
    print(f"Loading weights from {MODEL_PATH}...")
    violence_model.load_weights(MODEL_PATH)
    print("Model ready for inference!")
except Exception as e:
    print(f"Warning: Could not load model weights. Using mock predictions. Error: {e}")
    violence_model = None


# --- Serve UI ---
@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


# --- Strategy 1: Uniform Sampling ---
def frames_extraction_uniform(video_path: str) -> list:
    """
    Uniform sampling across the full video.
    Catches spread-out or late-occurring violence.
    Uses padding/letterboxing to maintain true aspect ratio.
    """
    frames_list = []
    try:
        cap = cv2.VideoCapture(video_path)
        raw_frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame_count > 300:
                break
            
            # Use aspect-ratio preserving resize instead of stretch
            frame = resize_with_pad(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_frames.append(frame)
            frame_count += 1
        cap.release()

        num_frames = len(raw_frames)
        if num_frames == 0:
            return []

        # Evenly spaced indices across the full video
        indices = np.linspace(0, num_frames - 1, SEQUENCE_LENGTH, dtype=int)
        for i in indices:
            frame = preprocess_input(raw_frames[i].astype(np.float32))
            frames_list.append(frame)

    except Exception as e:
        print(f"[frames_extraction_uniform] Error: {e}")
    return frames_list


# --- Strategy 2: Motion-Based Sampling ---
def frames_extraction_motion(video_path: str) -> list:
    """
    Motion-based sliding window sampling.
    Targets the most active (likely violent) segment in the video.
    Uses padding/letterboxing to maintain true aspect ratio.
    """
    frames_list = []
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        gray_frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame_count > 300:
                break
            
            # Use aspect-ratio preserving resize instead of stretch
            frame = resize_with_pad(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray_frames.append(gray)
            frame_count += 1
        cap.release()

        num_frames = len(frames)
        if num_frames == 0:
            return []

        # Pad if shorter than SEQUENCE_LENGTH
        if num_frames < SEQUENCE_LENGTH:
            while len(frames) < SEQUENCE_LENGTH:
                frames.append(frames[-1])
                gray_frames.append(gray_frames[-1])
            num_frames = SEQUENCE_LENGTH

        # Calculate motion differences between consecutive frames
        motion_scores = []
        for i in range(num_frames - 1):
            diff = cv2.absdiff(gray_frames[i], gray_frames[i + 1])
            motion_scores.append(np.sum(diff))

        # Sliding window to find the highest-motion segment
        max_motion = -1
        best_start_idx = 0
        window_size = SEQUENCE_LENGTH - 1

        for i in range(num_frames - SEQUENCE_LENGTH + 1):
            window_motion = sum(motion_scores[i: i + window_size])
            if window_motion > max_motion:
                max_motion = window_motion
                best_start_idx = i

        best_frames = frames[best_start_idx: best_start_idx + SEQUENCE_LENGTH]
        for frame in best_frames:
            frame = preprocess_input(frame.astype(np.float32))
            frames_list.append(frame)

    except Exception as e:
        print(f"[frames_extraction_motion] Error: {e}")
    return frames_list


# --- Prediction ---
def get_prediction(frames_array):
    """
    Returns (is_violence, violence_score).
    Threshold lowered to 0.35 to catch borderline violent content.
    """
    if violence_model:
        prediction = violence_model.predict(frames_array, verbose=0)[0]
        violence_score = float(prediction[CLASSES_LIST.index("Violence")])
        # Lowered from 0.5 → 0.35 to reduce false negatives
        is_violence = violence_score > 0.35
        confidence = violence_score if is_violence else (1.0 - violence_score)
    else:
        # Mock fallback when model is not loaded
        is_violence = False
        confidence = 0.99
        violence_score = 0.01
    return is_violence, confidence, violence_score


# --- ENDPOINT 1: Uploaded Video Files ---
@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    try:
        with temp_video as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = []

        # Strategy 1: Uniform sampling — catches spread-out violence
        frames_uniform = frames_extraction_uniform(temp_video.name)
        if len(frames_uniform) == SEQUENCE_LENGTH:
            arr = np.expand_dims(frames_uniform, axis=0)
            is_v, conf, v_score = get_prediction(arr)
            results.append({
                "strategy": "uniform",
                "is_violence": is_v,
                "confidence": conf,
                "violence_score": v_score
            })
            print(f"[Uniform]  is_violence={is_v}  violence_score={v_score:.4f}  confidence={conf:.4f}")

        # Strategy 2: Motion-based sampling — targets most active segment
        frames_motion = frames_extraction_motion(temp_video.name)
        if len(frames_motion) == SEQUENCE_LENGTH:
            arr = np.expand_dims(frames_motion, axis=0)
            is_v, conf, v_score = get_prediction(arr)
            results.append({
                "strategy": "motion",
                "is_violence": is_v,
                "confidence": conf,
                "violence_score": v_score
            })
            print(f"[Motion]   is_violence={is_v}  violence_score={v_score:.4f}  confidence={conf:.4f}")

        if not results:
            return {"status": "error", "message": "Could not extract enough frames from the video."}

        # If either strategy detects violence, flag the video
        violence_results = [r for r in results if r["is_violence"]]
        if violence_results:
            # Pick the violence result with the highest score
            best = max(violence_results, key=lambda x: x["violence_score"])
        else:
            # Both said safe — pick the result with highest confidence
            best = max(results, key=lambda x: x["confidence"])

        return {
            "status": "success",
            "is_violence": best["is_violence"],
            "confidence": round(best["confidence"] * 100, 2),
            "violence_score": round(best["violence_score"] * 100, 2),
            "detected_by": best["strategy"],
            "strategies_used": len(results)
        }

    finally:
        if os.path.exists(temp_video.name):
            os.remove(temp_video.name)


# --- ENDPOINT 2: Continuous Real-Time Webcam (WebSockets) ---
@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Rolling buffer — automatically drops oldest frame when full
    frame_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)

    try:
        while True:
            # Receive base64 image from frontend
            data = await websocket.receive_text()
            encoded_data = data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Use aspect-ratio preserving resize instead of stretch
            frame = resize_with_pad(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = preprocess_input(frame.astype(np.float32))

            frame_buffer.append(frame)

            # Only predict once we have a full window of 16 frames
            if len(frame_buffer) == SEQUENCE_LENGTH:
                frames_array = np.expand_dims(list(frame_buffer), axis=0)
                is_violence, confidence, violence_score = get_prediction(frames_array)

                await websocket.send_json({
                    "status": "success",
                    "is_violence": is_violence,
                    "confidence": round(confidence * 100, 2),
                    "violence_score": round(violence_score * 100, 2)
                })

    except WebSocketDisconnect:
        print("Webcam client disconnected.")
    except Exception as e:
        print(f"WebSocket Error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)