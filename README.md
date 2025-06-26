Markdown

# Player Re-Identification in Sports Footage

This project implements a solution for player re-identification in sports video footage, specifically focusing on a single camera feed. It utilizes a fine-tuned Ultralytics YOLOv11 model for object detection and integrates with ByteTrack for robust multi-object tracking and re-identification.

---

## 1. Project Objective

The objective is to identify each player in a given 15-second video clip (`15sec_input_720p.mp4`) and ensure that players who temporarily leave the frame (due to occlusion or going out of bounds) and reappear are assigned the same consistent identity (ID) as before. The solution aims to simulate real-time re-identification and player tracking.

---

## 2. Setup and Installation

Follow these steps to set up the project environment and install the necessary dependencies.

### 2.1. Place Model and Video Files

Ensure the following files are placed directly in the root directory of your project folder:

* **`best.pt`**: This is the fine-tuned Ultralytics YOLOv11 model for player and ball detection. It's a basic fine-tuned version of Ultralytics YOLOv11, trained specifically on player and ball detection.
* **`15sec_input_720p.mp4`**: The input video clip for re-identification.

### 2.2. Create a Python Virtual Environment (Recommended)

Using a virtual environment is highly recommended to manage project-specific dependencies and avoid conflicts with other Python projects.

```bash
python -m vvenv venv_reid_project
2.3. Activate the Virtual Environment
Before installing dependencies or running the script, activate your virtual environment:

On macOS/Linux:

Bash

source venv_reid_project/bin/activate
On Windows (Command Prompt):

DOS

venv_reid_project\Scripts\activate.bat
On Windows (PowerShell):

PowerShell

.\venv_reid_project\Scripts\Activate.ps1
Your terminal prompt should change to (venv_reid_project) indicating the environment is active.

2.4. Install Dependencies
With the virtual environment activated, install the required Python libraries using pip:

Bash

pip install ultralytics opencv-python numpy
ultralytics: Provides the YOLO model functionality and integrates with tracking algorithms like ByteTrack.

opencv-python: Used for video file I/O (reading frames, writing output video) and drawing visualizations (bounding boxes, text).

numpy: Essential for numerical operations and array manipulation, often used internally by ultralytics and opencv.

3. How to Run the Code
Once the setup is complete, you can run the re-identification script.

3.1. Verify Player Class ID
The model (best.pt) has specific class IDs for 'ball', 'goalkeeper', 'player', and 'referee'. It's crucial to confirm the PLAYER_CLASS_ID in the reid_script.py matches the actual 'player' class ID from your model's output.

Open reid_script.py in a text editor.

Locate the line PLAYER_CLASS_ID = 2 (or whatever value it currently holds).

When you run the script, the terminal output will show a line similar to Model's detected class names: {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}.

From this example, 'player' corresponds to ID 2. Therefore, PLAYER_CLASS_ID should be set to 2.

If your model's output is different, adjust PLAYER_CLASS_ID accordingly.

3.2. Execute the Script
Ensure your virtual environment is active, then run the script from your project's root directory:

Bash

python reid_script.py
3.3. Expected Output
Upon running the script:

A video window titled "Player Re-identification" will open, displaying the processed video frames in real-time.

Players will be enclosed in bounding boxes with a unique color for each ID and labeled with "ID: [Number] C:[Confidence]".

The ball will be enclosed in a red bounding box and labeled with "Ball C:[Confidence]".

A new video file named output_tracked_players_with_id.mp4 will be saved in your project directory.

The terminal will display messages regarding model loading, video processing progress, and completion.

To stop the live display, press the q key while the video window is active.

4. Code Structure and Methodology
The reid_script.py integrates a fine-tuned YOLOv11 model with Ultralytics' built-in tracking capabilities (specifically ByteTrack).

Object Detection: The best.pt model is loaded using ultralytics.YOLO. It is configured to run on GPU (cuda) if available, otherwise on CPU.

Multi-Object Tracking (MOT): The model.track() method is used for combined detection and tracking.

persist=True: This crucial parameter enables the tracker to maintain object identities across frames, facilitating re-identification even if players are briefly occluded or leave and re-enter the field of view.

tracker='bytetrack.yaml': Specifies ByteTrack as the underlying tracking algorithm. ByteTrack is known for its robustness in crowded scenes and handling occlusions.

Class Filtering: The script specifically targets 'player' detections (based on PLAYER_CLASS_ID) for ID assignment and unique coloring, while the 'ball' is detected but not assigned a persistent track ID for simplicity, as per the assignment's focus on player re-identification.

Visualization: OpenCV (cv2) is used to draw bounding boxes, track IDs, and confidence scores directly onto the video frames. A helper function get_color() ensures consistent, distinct colors for each player's ID.

Video Output: The processed frames are written to an output MP4 file, allowing for later review and evaluation.

5. Troubleshooting
ModuleNotFoundError: If you see errors like No module named 'cv2' or similar, ensure you have activated your virtual environment and installed all dependencies (pip install ultralytics opencv-python numpy).

Model/Video Not Found: Double-check that best.pt and 15sec_input_720p.mp4 are in the same directory as reid_script.py, or update the MODEL_PATH and VIDEO_PATH variables in the script.

Incorrect Labels ("Ball" for players): This is due to an incorrect PLAYER_CLASS_ID. Verify the Model's detected class names: output in your terminal and adjust PLAYER_CLASS_ID in your script accordingly.

Output Video Issues: If the output video is not saving or is corrupted, try changing the fourcc codec in the script (e.g., from 'mp4v' to 'XVID' or 'MJPG'). Ensure sufficient disk space.

Git Warnings about venv: Warnings like "LF will be replaced by CRLF" related to venv_reid_project/ indicate line ending conversions. Ensure your .gitignore includes venv_reid_project/ to prevent Git from tracking these files, and run git rm -r --cached venv_reid_project/ if they were accidentally added to the cache.
