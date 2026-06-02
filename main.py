import cv2
import tkinter as tk
import math
import time
from collections import deque
from mediapipe.python.solutions import face_mesh

mp_face_mesh = face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

cap = cv2.VideoCapture(0)

# SCREEN DIMENSIONS
WIDTH = 1200
HEIGHT = 700

root = tk.Tk()
root.title("EyeType AI Standardized Calibration")

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black", highlightthickness=0)
canvas.pack()

# UI Elements
calibration_target = canvas.create_oval(0, 0, 0, 0, fill="red", state="hidden")
calibration_text = canvas.create_text(WIDTH // 2, HEIGHT // 2 - 50, text="", fill="white", font=("Arial", 16), justify="center")

cursor = canvas.create_oval(
    WIDTH // 2 - 15, HEIGHT // 2 - 15, WIDTH // 2 + 15, HEIGHT // 2 + 15,
    fill="cyan", state="hidden"
)

# --- CALIBRATION CONFIGURATION ---
# Fixed comfortable baselines determined during your calibration countdown
x_min, x_max = 0.45, 0.55
y_min, y_max = -0.02, 0.06

# Advanced filters
DEAD_ZONE = 0.035
SMOOTH_FRAMES = 6
x_history = deque(maxlen=SMOOTH_FRAMES)
y_history = deque(maxlen=SMOOTH_FRAMES)

last_x_ratio = 0.5
last_y_ratio = 0.5
is_calibrated = False

def get_eye_data():
    """Helper function to extract raw eye ratios using the stable nose baseline."""
    success, frame = cap.read()
    if not success:
        return None
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        
        # Pull key landmarks
        left_corner = landmarks[33]
        right_corner = landmarks[133]
        top_lid = landmarks[159]
        bottom_lid = landmarks[145]
        iris = landmarks[468]
        nose_bridge = landmarks[6]
        
        # Conversion
        lc_x, lc_y = int(left_corner.x * w), int(left_corner.y * h)
        rc_x, rc_y = int(right_corner.x * w), int(right_corner.y * h)
        iris_x, iris_y = int(iris.x * w), int(iris.y * h)
        nose_y = int(nose_bridge.y * h)
        
        eye_width = rc_x - lc_x
        eyelid_dist = int(bottom_lid.y * h) - int(top_lid.y * h)
        
        # Check for blinks to filter bad calibration data
        if eye_width > 0 and (eyelid_dist / eye_width) < 0.13:
            return "BLINK"
            
        if eye_width > 0:
            ratio = (iris_x - lc_x) / eye_width
            vertical_offset = (iris_y - nose_y) / eye_width
            return ratio, vertical_offset
            
    return None

def run_calibration():
    """Runs a structured calibration wizard mapping look ranges directly to limits."""
    global x_min, x_max, y_min, y_max, is_calibrated
    
    # --- PHASE 1: TOP LEFT TARGET ---
    canvas.itemconfig(calibration_text, text="LOOK DIRECTLY AT THE RED DOT\n(Top-Left Corner)")
    canvas.coords(calibration_target, 40 - 20, 40 - 20, 40 + 20, 40 + 20)
    canvas.itemconfig(calibration_target, state="normal")
    root.update()
    time.sleep(1.5) # Time to guide eye focus
    
    collected_x, collected_y = [], []
    start_time = time.time()
    while time.time() - start_time < 2.0:
        data = get_eye_data()
        if data and data != "BLINK":
            collected_x.append(data[0])
            collected_y.append(data[1])
        root.update()
        
    if collected_x:
        x_min = sum(collected_x) / len(collected_x)
        y_min = sum(collected_y) / len(collected_y)

    # --- PHASE 2: BOTTOM RIGHT TARGET ---
    canvas.itemconfig(calibration_text, text="NOW LOOK DIRECTLY AT THE RED DOT\n(Bottom-Right Corner)")
    canvas.coords(calibration_target, WIDTH - 40 - 20, HEIGHT - 40 - 20, WIDTH - 40 + 20, HEIGHT - 40 + 20)
    root.update()
    time.sleep(1.5)
    
    collected_x, collected_y = [], []
    start_time = time.time()
    while time.time() - start_time < 2.0:
        data = get_eye_data()
        if data and data != "BLINK":
            collected_x.append(data[0])
            collected_y.append(data[1])
        root.update()
        
    if collected_x:
        x_max = sum(collected_x) / len(collected_x)
        y_max = sum(collected_y) / len(collected_y)

    # Clear Calibration UI Elements
    canvas.itemconfig(calibration_target, state="hidden")
    canvas.itemconfig(calibration_text, state="hidden")
    canvas.itemconfig(cursor, state="normal")
    is_calibrated = True

def move_cursor(x_ratio, y_ratio):
    global last_x_ratio, last_y_ratio
    
    # 1. Add raw normalized values straight to the moving average queue
    x_history.append(x_ratio)
    y_history.append(y_ratio)
    
    avg_x = sum(x_history) / len(x_history)
    avg_y = sum(y_history) / len(y_history)

    # 2. Distance Dead-Zone Calculation
    dx = avg_x - last_x_ratio
    dy = avg_y - last_y_ratio
    distance = math.sqrt(dx*dx + dy*dy)
    
    if distance < DEAD_ZONE:
        # Snap completely still to allow keyboard target dwelling
        smoothed_x = last_x_ratio
        smoothed_y = last_y_ratio
    else:
        # Panning smoothing coefficient
        alpha = 0.10 if distance < 0.15 else 0.35
        smoothed_x = (last_x_ratio * (1 - alpha)) + (avg_x * alpha)
        smoothed_y = (last_y_ratio * (1 - alpha)) + (avg_y * alpha)
    
    # Strict Screen Boundary Caps
    smoothed_x = max(0.0, min(1.0, smoothed_x))
    smoothed_y = max(0.0, min(1.0, smoothed_y))
    
    last_x_ratio = smoothed_x
    last_y_ratio = smoothed_y

    # Translate ratios to layout pixels
    x = int(smoothed_x * WIDTH)
    y = int(smoothed_y * HEIGHT)

    canvas.coords(cursor, x - 15, y - 15, x + 15, y + 15)
    root.update()

# Start the Calibration sequence on application launch
root.after(500, run_calibration)

while True:
    if not is_calibrated:
        root.update()
        continue
        
    data = get_eye_data()
    
    if data:
        if data == "BLINK":
            # Retain current positions safely during blinks
            move_cursor(last_x_ratio, last_y_ratio)
        else:
            ratio, vertical_offset = data
            
            # Normalize measurements using custom calibrated limits
            x_range = (x_max - x_min) if (x_max - x_min) > 0 else 0.01
            y_range = (y_max - y_min) if (y_max - y_min) > 0 else 0.01
            
            normalized_ratio = (ratio - x_min) / x_range
            normalized_vertical = (vertical_offset - y_min) / y_range
            
            # Clamp limits safely before updating coordinates
            normalized_ratio = max(0.0, min(1.0, normalized_ratio))
            normalized_vertical = max(0.0, min(1.0, normalized_vertical))
            
            move_cursor(normalized_ratio, normalized_vertical)
            
    # Polling break to safely update UI loops
    key = cv2.waitKey(1)
    if key == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()
root.destroy()