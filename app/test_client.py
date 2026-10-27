import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- 1. DEFINE CONSTANTS ---
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
SEQUENCE_LENGTH = 40  # Number of frames to extract

# --- 2. YOUR PREPROCESSING FUNCTIONS ---
def resize_with_pad(frame, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    """
    Resizes an image to the target size while maintaining the aspect ratio.
    Pads the remaining space with black borders (letterboxing).
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

def frames_extraction(video_path: str) -> list:
    """
    Finds the continuous SEQUENCE_LENGTH frames with the highest motion.
    Uses padding/letterboxing to maintain aspect ratio, and MobileNetV2 preprocessing [-1, 1].
    """
    frames_list = []
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        gray_frames = []

        # 1. Read all frames, apply resize_with_pad, convert to RGB
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = resize_with_pad(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Crucial: MobileNet needs RGB
            frames.append(frame)
            
            # Grayscale for fast motion calculation
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray_frames.append(gray)

        cap.release()

        num_frames = len(frames)
        if num_frames == 0:
            print("No frames found in video.")
            return []

        # 2. Pad video if shorter than SEQUENCE_LENGTH
        if num_frames < SEQUENCE_LENGTH:
            while len(frames) < SEQUENCE_LENGTH:
                frames.append(frames[-1])
                gray_frames.append(gray_frames[-1])
            num_frames = SEQUENCE_LENGTH

        # 3. Calculate motion differences
        motion_scores = []
        for i in range(num_frames - 1):
            diff = cv2.absdiff(gray_frames[i], gray_frames[i+1])
            motion_scores.append(np.sum(diff))

        # 4. Sliding window for highest motion segment
        max_motion = -1
        best_start_idx = 0
        window_size = SEQUENCE_LENGTH - 1

        for i in range(num_frames - SEQUENCE_LENGTH + 1):
            window_motion = sum(motion_scores[i : i + window_size])
            if window_motion > max_motion:
                max_motion = window_motion
                best_start_idx = i

        # 5. Extract exactly the frames containing the action
        best_frames = frames[best_start_idx : best_start_idx + SEQUENCE_LENGTH]

        # 6. Apply MobileNetV2 preprocess_input [-1, 1]
        for frame in best_frames:
            frame = preprocess_input(frame.astype(np.float32))
            frames_list.append(frame)

    except Exception as e:
        print(f"[frames_extraction] Error reading {video_path}: {e}")

    return frames_list

# --- 3. VISUALIZATION FUNCTION ---
def visualize_extracted_frames(frames):
    """
    Plots the extracted frames. Reverses the [-1, 1] normalization so 
    matplotlib can display them properly.
    """
    if not frames:
        print("No frames to visualize.")
        return

    num_frames = len(frames)
    
    # Setup matplotlib grid (2 rows)
    cols = (num_frames + 1) // 2
    fig, axes = plt.subplots(2, cols, figsize=(15, 6))
    axes = axes.flatten()

    for i, frame in enumerate(frames):
        # Denormalize: Convert from [-1, 1] back to [0, 1] for visualization
        display_frame = (frame + 1.0) / 2.0 
        
        # Clip values to ensure they stay strictly inside [0.0, 1.0]
        display_frame = np.clip(display_frame, 0.0, 1.0)

        axes[i].imshow(display_frame)
        axes[i].set_title(f"Frame {i+1}")
        axes[i].axis('off')

    # Turn off any empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Highest Motion Frames (Aspect-Ratio Preserved & Letterboxed)", fontsize=16)
    plt.tight_layout()
    plt.show()

# --- 4. RUN THE PIPELINE ---
if __name__ == "__main__":
    # ---> PUT YOUR VIDEO PATH HERE <---
    video_path = "basant.mp4" 

    print(f"Processing video: {video_path}...")
    extracted_frames = frames_extraction(video_path)
    
    if extracted_frames:
        print(f"Successfully extracted {len(extracted_frames)} frames.")
        print(f"Frame shape: {extracted_frames[0].shape}, Min Val: {np.min(extracted_frames[0]):.2f}, Max Val: {np.max(extracted_frames[0]):.2f}")
        visualize_extracted_frames(extracted_frames)
    else:
        print("Extraction failed. Please check the video path.")