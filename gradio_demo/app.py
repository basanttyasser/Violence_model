import os
import cv2
import numpy as np
import tensorflow as tf
import gradio as gr
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Input, TimeDistributed, GlobalAveragePooling2D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import mixed_precision

# Set this to match your training environment
mixed_precision.set_global_policy("mixed_float16")

# ==========================================
# 1. CONFIGURATION 
# ==========================================
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
SEQUENCE_LENGTH = 16
CLASSES_LIST =["NonViolence", "Violence"]
MODEL_PATH = "modelv2.keras"

# ==========================================
# 2. BUILD ARCHITECTURE & LOAD WEIGHTS
# ==========================================
def create_model():
    """
    Rebuilds the exact same architecture used during training.
    """
    mobilenet = MobileNetV2(
        include_top=False,
        weights=None, # We don't need 'imagenet' here because we are loading our own weights
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
    model = create_model()
    print("Loading weights from model.keras...")
    
    # By loading just the weights, we bypass Keras version bugs!
    model.load_weights(MODEL_PATH)
    print("Model ready for inference!")
except Exception as e:
    print(f"Error loading model weights: {e}")
    print("Ensure 'model.keras' is in the same directory.")

# ==========================================
# 3. FRAME EXTRACTION
# ==========================================
def frames_extraction(video_path):
    """
    Extracts the most action-dense 16 frames from the video.
    """
    frames_list =[]
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        gray_frames =[]

        # Read frames (Limit to first 300 frames / ~10 seconds to prevent RAM crashes)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_count > 300: 
                break
            
            frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR to RGB
            frames.append(frame)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray_frames.append(gray)
            frame_count += 1

        cap.release()

        num_frames = len(frames)
        if num_frames == 0:
            return[]

        # Pad if shorter than 16 frames
        if num_frames < SEQUENCE_LENGTH:
            while len(frames) < SEQUENCE_LENGTH:
                frames.append(frames[-1])
                gray_frames.append(gray_frames[-1])
            num_frames = SEQUENCE_LENGTH

        # Calculate motion
        motion_scores =[]
        for i in range(num_frames - 1):
            diff = cv2.absdiff(gray_frames[i], gray_frames[i+1])
            motion_scores.append(np.sum(diff))

        # Sliding window for highest motion
        max_motion = -1
        best_start_idx = 0
        window_size = SEQUENCE_LENGTH - 1

        for i in range(num_frames - SEQUENCE_LENGTH + 1):
            window_motion = sum(motion_scores[i : i + window_size])
            if window_motion > max_motion:
                max_motion = window_motion
                best_start_idx = i

        # Extract frames
        best_frames = frames[best_start_idx : best_start_idx + SEQUENCE_LENGTH]

        # Preprocess for MobileNetV2 [-1, 1]
        for frame in best_frames:
            frame = preprocess_input(frame.astype(np.float32))
            frames_list.append(frame)

    except Exception as e:
        print(f"Error processing video: {e}")

    return np.array(frames_list)

# ==========================================
# 4. PREDICTION FUNCTION
# ==========================================
def predict_video(video_file_path):
    if video_file_path is None:
        return {"Error: No video uploaded": 1.0}
    
    # 1. Extract frames
    frames = frames_extraction(video_file_path)
    
    if len(frames) != SEQUENCE_LENGTH:
        return {"Error: Could not read video correctly": 1.0}
    
    # 2. Expand dimensions to make it a batch of 1
    frames_batch = np.expand_dims(frames, axis=0)
    
    # 3. Predict
    predictions = model.predict(frames_batch)[0]
    
    # 4. Format output
    result_dict = {CLASSES_LIST[i]: float(predictions[i]) for i in range(len(CLASSES_LIST))}
    
    return result_dict

# ==========================================
# 5. GRADIO INTERFACE SETUP
# ==========================================
interface = gr.Interface(
    fn=predict_video,
    inputs=gr.Video(label="Upload Video (mp4, avi, mov)"),
    outputs=gr.Label(num_top_classes=2, label="Prediction Confidence"),
    title="🛡️ Real-Time Violence Detection",
    description="Upload a short video. The system will isolate the sequence with the highest movement and use the MobileNetV2+LSTM model to determine if violence is occurring."
)

if __name__ == "__main__":
    # Launch the web app
    interface.launch(share=True, debug=True, theme="default")