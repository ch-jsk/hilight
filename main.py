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
root.title("EyeType AI Complete Robust Tracking System")

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black", highlightthickness=0)
canvas.pack()

# UI Elements
calibration_target = canvas.create_oval(0, 0, 0, 0, fill="red", state="hidden")
calibration_text = canvas.create_text(WIDTH // 2, HEIGHT // 2 - 50, text="", fill="white", font=("Arial", 16), justify="center")

cursor = canvas.create_oval(
    WIDTH // 2 - 15, HEIGHT // 2 - 15, WIDTH // 2 + 15, HEIGHT // 2 + 15,
    fill="cyan", state="hidden"
)

# --- ADVANCED DRIFT & JITTER FILTER PANEL ---
GAIN_X = 1.6          
GAIN_Y = 1.8          

CENTER_DEADZONE_X = 0.07  
CENTER_DEADZONE_Y = 0.08  

# JITTER BOX: The radius of the invisible circle that freezes localized shaking
HYSTERESIS_RADIUS = 0.022  

# FRICTION BRAKE: Minimum required velocity to accept eye tracking input
DRIFT_SPEED_THRESHOLD = 0.012  

SMOOTH_FRAMES = 5
x_history = deque(maxlen=SMOOTH_FRAMES)
y_history = deque(maxlen=SMOOTH_FRAMES)

# Calibration baselines (Will auto-update during wizard execution)
x_min, x_max = 0.44, 0.54
y_min, y_max = -0.04, 0.05

last_x_ratio = 0.5
last_y_ratio = 0.5
is_calibrated = False

def get_eye_data(show_preview=True):
    success, frame = cap.read()
    if not success:
        return None

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    data = None

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        left_corner = landmarks[33]
        right_corner = landmarks[133]
        top_lid = landmarks[159]
        bottom_lid = landmarks[145]
        iris = landmarks[468]
        nose_bridge = landmarks[6]

        lc_x = int(left_corner.x * w)
        lc_y = int(left_corner.y * h)
        rc_x = int(right_corner.x * w)
        rc_y = int(right_corner.y * h)
        iris_x = int(iris.x * w)
        iris_y = int(iris.y * h)
        nose_y = int(nose_bridge.y * h)

        eye_width = rc_x - lc_x
        eyelid_dist = int(bottom_lid.y * h) - int(top_lid.y * h)

        # Draw overlays for live monitoring window
        cv2.circle(frame, (iris_x, iris_y), 4, (0, 255, 255), -1)
        cv2.circle(frame, (lc_x, lc_y), 2, (0, 255, 0), -1)
        cv2.circle(frame, (rc_x, rc_y), 2, (0, 255, 0), -1)
        cv2.circle(frame, (int(nose_bridge.x * w), nose_y), 3, (255, 0, 0), -1)

        if eye_width > 0:
            if (eyelid_dist / eye_width) < 0.13:
                data = "BLINK"
            else:
                ratio = (iris_x - lc_x) / eye_width
                vertical_offset = (iris_y - nose_y) / eye_width
                data = (ratio, vertical_offset)

    # Placed safely outside landmark checking block to prevent window lockups
    if show_preview:
        preview = cv2.resize(frame, (320, 240))
        cv2.imshow("EyeType Camera Feed", preview)

    return data

def run_calibration():
    global x_min, x_max, y_min, y_max, is_calibrated
    
    points = [
        ("LOOK DIRECTLY AT THE RED DOT\n(Top-Left Corner)", 60, 60, "min"),
        ("NOW LOOK DIRECTLY AT THE RED DOT\n(Bottom-Right Corner)", WIDTH - 60, HEIGHT - 60, "max")
    ]
    
    for msg, target_x, target_y, mode in points:
        canvas.itemconfig(calibration_text, text=msg)
        canvas.coords(calibration_target, target_x - 20, target_y - 20, target_x + 20, target_y + 20)
        canvas.itemconfig(calibration_target, state="normal")
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
            if mode == "min":
                x_min = sum(collected_x) / len(collected_x)
                y_min = sum(collected_y) / len(collected_y)
            else:
                x_max = sum(collected_x) / len(collected_x)
                y_max = sum(collected_y) / len(collected_y)

    canvas.itemconfig(calibration_target, state="hidden")
    canvas.itemconfig(calibration_text, text="CALIBRATION SUCCESSFUL!\nFocus on the middle to lock tracking...")
    root.update()
    
    # Flush calibration data arrays completely
    x_history.clear()
    y_history.clear()
    time.sleep(1.5)  
    
    canvas.itemconfig(calibration_text, state="hidden")
    canvas.itemconfig(cursor, state="normal")
    is_calibrated = True

def move_cursor(x_ratio, y_ratio):
    global last_x_ratio, last_y_ratio
    
    # 1. Map around tracking midpoint
    offset_x = x_ratio - 0.5
    offset_y = y_ratio - 0.5
    
    # 2. Base Center Box Constraint
    if abs(offset_x) < CENTER_DEADZONE_X and abs(offset_y) < CENTER_DEADZONE_Y:
        adjusted_x = 0.5
        adjusted_y = 0.5
    else:
        adjusted_x = 0.5 + (offset_x * GAIN_X)
        adjusted_y = 0.5 + (offset_y * GAIN_Y)
    
    # 3. ANTI-JITTER LOCALIZED HYSTERESIS BOX
    # Filter circular micro-tremors within targeted areas
    jitter_dx = adjusted_x - last_x_ratio
    jitter_dy = adjusted_y - last_y_ratio
    gaze_distance_from_cursor = math.sqrt(jitter_dx*jitter_dx + jitter_dy*jitter_dy)
    
    if gaze_distance_from_cursor < HYSTERESIS_RADIUS:
        smoothed_x = last_x_ratio
        smoothed_y = last_y_ratio
    else:
        # 4. VELOCITY FRICTION LOCK
        # If the eye motion breaks the hysteresis boundary but is extremely slow, trap it as drift
        input_dx = adjusted_x - last_x_ratio
        input_dy = adjusted_y - last_y_ratio
        gaze_velocity = math.sqrt(input_dx*input_dx + input_dy*input_dy)
        
        if gaze_velocity < DRIFT_SPEED_THRESHOLD:
            smoothed_x = last_x_ratio
            smoothed_y = last_y_ratio
        else:
            # Deliberate look vector shift confirmed -> calculate sliding average updates
            x_history.append(adjusted_x)
            y_history.append(adjusted_y)
            
            avg_x = sum(x_history) / len(x_history)
            avg_y = sum(y_history) / len(y_history)
            
            alpha = 0.25  
            smoothed_x = (last_x_ratio * (1 - alpha)) + (avg_x * alpha)
            smoothed_y = (last_y_ratio * (1 - alpha)) + (avg_y * alpha)
    
    # Keep coordinate calculations inside actual screen dimensions
    smoothed_x = max(0.0, min(1.0, smoothed_x))
    smoothed_y = max(0.0, min(1.0, smoothed_y))
    
    last_x_ratio = smoothed_x
    last_y_ratio = smoothed_y

    x = int(smoothed_x * WIDTH)
    y = int(smoothed_y * HEIGHT)

    canvas.coords(cursor, x - 15, y - 15, x + 15, y + 15)
    root.update()

# Run Calibration sequence shortly after window initializes
root.after(500, run_calibration)

# --- RE-ENGINEERED HIGH PERFORMANCE SYSTEM LOOP ---
while True:
    if not is_calibrated:
        root.update()
        continue
        
    data = get_eye_data()
    
    if data:
        if data == "BLINK":
            move_cursor(last_x_ratio, last_y_ratio)
        else:
            ratio, vertical_offset = data
            
            x_range = (x_max - x_min) if (x_max - x_min) > 0 else 0.01
            y_range = (y_max - y_min) if (y_max - y_min) > 0 else 0.01
            
            normalized_ratio = (ratio - x_min) / x_range
            normalized_vertical = (vertical_offset - y_min) / y_range
            
            normalized_ratio = max(0.0, min(1.0, normalized_ratio))
            normalized_vertical = max(0.0, min(1.0, normalized_vertical))
            
            move_cursor(normalized_ratio, normalized_vertical)
            
    key = cv2.waitKey(1)
    if key == 27: # ESC closes the program execution clean
        break

cap.release()
cv2.destroyAllWindows()
root.destroy()