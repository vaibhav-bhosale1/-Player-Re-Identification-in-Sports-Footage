import cv2
from ultralytics import YOLO
import torch
import numpy as np

# --- Configuration Section ---
# Ensure these paths are correct relative to where you run the script
MODEL_PATH = 'best.pt'      # Your downloaded fine-tuned model
VIDEO_PATH = '15sec_input_720p.mp4'        # Your input video
OUTPUT_VIDEO_PATH = 'output_tracked_players_with_id.mp4' # Name for the output video file

# --- Load the YOLOv11 Model ---
print("Attempting to load the YOLOv11 model...")
try:
    # Check for GPU availability and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load your specific fine-tuned model
    model = YOLO(MODEL_PATH).to(device)
    print(f"Successfully loaded YOLOv11 model from '{MODEL_PATH}'.")

    # Print model class names to understand the mapping (e.g., 0: 'player', 1: 'ball')
    # This helps confirm which class ID corresponds to 'player'.
    print("Model's detected class names:", model.names)

except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please ensure '{MODEL_PATH}' is in the same directory as this script.")
    exit() # Exit if the model cannot be loaded

# --- Determine Player Class ID ---
# BASED ON THE OUTPUT OF "Model's detected class names:" ABOVE, ADJUST THIS LINE:
# If 'player' maps to 0 in model.names, keep PLAYER_CLASS_ID = 0.
# If 'player' maps to 1 in model.names, change PLAYER_CLASS_ID = 1, etc.
PLAYER_CLASS_ID = 2  # <--- ***CHECK THIS VALUE AFTER RUNNING AND LOOKING AT TERMINAL OUTPUT***

# --- Open the Input Video File ---
print(f"Attempting to open video file: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file '{VIDEO_PATH}'.")
    print("Please ensure the video file is in the same directory and is not corrupted.")
    exit() # Exit if video cannot be opened

# Get video properties for creating the output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Input video properties: {frame_width}x{frame_height} @ {fps} FPS")

# --- Set up Video Writer for Output ---
# 'mp4v' is a common codec for .mp4 files. If you encounter issues, try 'XVID' or 'MJPG'.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    print(f"Error: Could not create output video file '{OUTPUT_VIDEO_PATH}'.")
    print("Check codec, file path, and disk space.")
    cap.release() # Release input video capture
    exit()

# --- Helper Function to Generate Consistent Colors for Track IDs ---
def get_color(idx):
    """Generates a pseudo-random, consistent color for a given track ID."""
    # Use a fixed seed based on the ID for reproducible colors
    rng = np.random.default_rng(seed=int(idx * 12345))
    return tuple(map(int, rng.integers(0, 255, size=3))) # Generate RGB tuple

# --- Main Video Processing Loop ---
print("Starting video processing...")
frame_count = 0
while True:
    ret, frame = cap.read() # Read a frame from the video
    if not ret:
        break # Break the loop if no more frames (end of video)

    frame_count += 1
    # Optional: Print progress for long videos
    # if frame_count % 100 == 0:
    #     print(f"Processing frame {frame_count}...")

    # Perform detection and tracking using the YOLO model's built-in `track` method
    # - `persist=True`: This is critical for maintaining track IDs across frames and re-identification.
    # - `tracker='bytetrack.yaml'`: Specifies the ByteTrack algorithm for tracking.
    # - `conf=0.4`: Minimum confidence score for a detection to be considered by the tracker.
    # - `iou=0.5`: Intersection Over Union threshold for Non-Maximum Suppression.
    # - `verbose=False`: Prevents `ultralytics` from printing verbose output for each frame.
    results = model.track(frame, persist=True, tracker='bytetrack.yaml', conf=0.4, iou=0.5, verbose=False)

    # --- Process and Visualize Tracking Results ---
    for r in results: # 'r' represents results for a single source (our frame)
        # Check if tracking IDs are available (they will be if objects are being tracked)
        if r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()          # Bounding box coordinates (x1, y1, x2, y2)
            track_ids = r.boxes.id.cpu().numpy().astype(int) # Unique tracking IDs (e.g., 1, 2, 3...)
            class_ids = r.boxes.cls.cpu().numpy().astype(int) # Class IDs (e.g., 0 for player, 1 for ball)
            confidences = r.boxes.conf.cpu().numpy()    # Confidence scores for detections

            # Iterate through each detected/tracked object in the current frame
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i]) # Get integer coordinates
                track_id = track_ids[i]             # Get the unique track ID
                cls_id = class_ids[i]               # Get the class ID
                conf = confidences[i]               # Get the confidence score

                if cls_id == PLAYER_CLASS_ID:
                    # If it's a player, draw their bounding box and ID
                    color = get_color(track_id) # Get a unique color based on the ID
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID: {track_id} C:{conf:.2f}" # Label with ID and confidence
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                else:
                    # If it's the ball (or another non-player object), just draw a box
                    # The task focuses on player re-identification, so ball doesn't need an ID.
                    ball_color = (0, 0, 255) # Red color for the ball
                    cv2.rectangle(frame, (x1, y1), (x2, y2), ball_color, 2)
                    cv2.putText(frame, f"Ball C:{conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ball_color, 2, cv2.LINE_AA)

    # --- Display and Save Frame ---
    # Display the processed frame (optional - opens a window)
    cv2.imshow('Player Re-identification', frame)

    # Write the current frame to the output video file
    out.write(frame)

    # Check for 'q' key press to quit the display window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Release Resources ---
print(f"Finished processing all {frame_count} frames. Output video saved to: '{OUTPUT_VIDEO_PATH}'")
cap.release()    # Release the video capture object
out.release()    # Release the video writer object
cv2.destroyAllWindows() # Close all OpenCV windows