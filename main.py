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
root.title("EyeType AI Discrete Grid Interface")

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="#111", highlightthickness=0)
canvas.pack()

# --- GRID CONFIGURATION ---
ROWS, COLS = 3, 3
BOX_WIDTH = WIDTH // COLS
BOX_HEIGHT = HEIGHT // ROWS

# Labels matching your layout layout map
GRID_LABELS = [
    ["A", "B", "C"],
    ["D", "E", "F"],
    ["G", "H", "I"]
]

# Track UI element IDs for clean visual state updates
grid_rectangles = {}
grid_texts = {}

def build_discrete_grid():
    """Generates the static bounding boxes and text vectors for the 3x3 layout."""
    for r in range(ROWS):
        for c in range(COLS):
            x1 = c * BOX_WIDTH
            y1 = r * BOX_HEIGHT
            x2 = x1 + BOX_WIDTH
            y2 = y1 + BOX_HEIGHT
            
            # Draw baseline unselected card borders
            rect_id = canvas.create_rectangle(
                x1 + 10, y1 + 10, x2 - 10, y2 - 10, 
                fill="#1A1A1A", outline="#444", width=3
            )
            
            # Standard labels placed precisely at row/col midpoints
            text_id = canvas.create_text(
                (x1 + x2) // 2, (y1 + y2) // 2, 
                text=GRID_LABELS[r][c], fill="#888", font=("Arial", 36, "bold")
            )
            
            grid_rectangles[(r, c)] = rect_id
            grid_texts[(r, c)] = text_id

# Initialize layout structures
build_discrete_grid()

# UI Calibration Vectors
calibration_target = canvas.create_oval(0, 0, 0, 0, fill="red", state="hidden")
calibration_text = canvas.create_text(WIDTH // 2, HEIGHT // 2 - 50, text="", fill="white", font=("Arial", 16), justify="center")

# --- ADVANCED SYSTEM FILTERS ---
GAIN_X = 1.6          
GAIN_Y = 1.8          

CENTER_DEADZONE_X = 0.07  
CENTER_DEADZONE_Y = 0.08  
HYSTERESIS_RADIUS = 0.022  
DRIFT_SPEED_THRESHOLD = 0.012  

SMOOTH_FRAMES = 5
x_history = deque(maxlen=SMOOTH_FRAMES)
y_history = deque(maxlen=SMOOTH_FRAMES)

x_min, x_max = 0.44, 0.54
y_min, y_max = -0.04, 0.05

last_x_ratio = 0.5
last_y_ratio = 0.5
active_cell = None
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

        # Video Demo Monitoring Markers
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

    if show_preview:
        preview = cv2.resize(frame, (320, 240))
        cv2.imshow("EyeType Camera Feed", preview)

    return data

def run_calibration():
    global x_min, x_max, y_min, y_max, is_calibrated
    
    # Temporarily overlay calibration text clearly over the grid layout
    canvas.tag_raise(calibration_text)
    canvas.tag_raise(calibration_target)
    
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
    canvas.itemconfig(calibration_text, text="CALIBRATION SUCCESSFUL!\nInitializing Discrete Selection Array...")
    root.update()
    
    x_history.clear()
    y_history.clear()
    time.sleep(1.5)  
    
    canvas.itemconfig(calibration_text, state="hidden")
    is_calibrated = True

def update_grid_highlights(target_x_ratio, target_y_ratio):
    """Translates smooth spatial coordinates directly into discrete bounded card states."""
    global active_cell
    
    # Convert normalized scale parameters back to layout frame pixels
    px_x = int(target_x_ratio * WIDTH)
    px_y = int(target_y_ratio * HEIGHT)
    
    # Calculate discrete row and column cell index
    col_idx = max(0, min(COLS - 1, px_x // BOX_WIDTH))
    row_idx = max(0, min(ROWS - 1, px_y // BOX_HEIGHT))
    current_cell = (row_idx, col_idx)
    
    # Trigger UI updates only when moving into a brand-new cell area
    if current_cell != active_cell:
        # Revert old active cell colors back to standard unselected styling
        if active_cell in grid_rectangles:
            canvas.itemconfig(grid_rectangles[active_cell], fill="#1A1A1A", outline="#444")
            old_label = GRID_LABELS[active_cell[0]][active_cell[1]]
            canvas.itemconfig(grid_texts[active_cell], text=old_label, fill="#888")
            
        # Transform newly targeted cell into the active highlighted style bracket [A]
        if current_cell in grid_rectangles:
            canvas.itemconfig(grid_rectangles[current_cell], fill="#003333", outline="#00FFF0")
            new_label = f"[{GRID_LABELS[row_idx][col_idx]}]"
            canvas.itemconfig(grid_texts[current_cell], text=new_label, fill="#00FFF0")
            
        active_cell = current_cell

def process_gaze_trajectory(x_ratio, y_ratio):
    global last_x_ratio, last_y_ratio
    
    offset_x = x_ratio - 0.5
    offset_y = y_ratio - 0.5
    
    if abs(offset_x) < CENTER_DEADZONE_X and abs(offset_y) < CENTER_DEADZONE_Y:
        adjusted_x = 0.5
        adjusted_y = 0.5
    else:
        adjusted_x = 0.5 + (offset_x * GAIN_X)
        adjusted_y = 0.5 + (offset_y * GAIN_Y)
    
    jitter_dx = adjusted_x - last_x_ratio
    jitter_dy = adjusted_y - last_y_ratio
    gaze_distance_from_cursor = math.sqrt(jitter_dx*jitter_dx + jitter_dy*jitter_dy)
    
    if gaze_distance_from_cursor < HYSTERESIS_RADIUS:
        smoothed_x = last_x_ratio
        smoothed_y = last_y_ratio
    else:
        input_dx = adjusted_x - last_x_ratio
        input_dy = adjusted_y - last_y_ratio
        gaze_velocity = math.sqrt(input_dx*input_dx + input_dy*input_dy)
        
        if gaze_velocity < DRIFT_SPEED_THRESHOLD:
            smoothed_x = last_x_ratio
            smoothed_y = last_y_ratio
        else:
            x_history.append(adjusted_x)
            y_history.append(adjusted_y)
            
            avg_x = sum(x_history) / len(x_history)
            avg_y = sum(y_history) / len(y_history)
            
            alpha = 0.25  
            smoothed_x = (last_x_ratio * (1 - alpha)) + (avg_x * alpha)
            smoothed_y = (last_y_ratio * (1 - alpha)) + (avg_y * alpha)
    
    smoothed_x = max(0.0, min(1.0, smoothed_x))
    smoothed_y = max(0.0, min(1.0, smoothed_y))
    
    last_x_ratio = smoothed_x
    last_y_ratio = smoothed_y

    # Pass the calculated coordinate to the box highlight calculation engine
    update_grid_highlights(smoothed_x, smoothed_y)
    root.update()

# Launch Calibration Sequence shortly after thread initiation
root.after(500, run_calibration)

# --- SYSTEM OPERATIONAL POLLING LOOP ---
while True:
    if not is_calibrated:
        root.update()
        continue
        
    data = get_eye_data()
    
    if data:
        if data == "BLINK":
            process_gaze_trajectory(last_x_ratio, last_y_ratio)
        else:
            ratio, vertical_offset = data
            
            x_range = (x_max - x_min) if (x_max - x_min) > 0 else 0.01
            y_range = (y_max - y_min) if (y_max - y_min) > 0 else 0.01
            
            normalized_ratio = (ratio - x_min) / x_range
            normalized_vertical = (vertical_offset - y_min) / y_range
            
            normalized_ratio = max(0.0, min(1.0, normalized_ratio))
            normalized_vertical = max(0.0, min(1.0, normalized_vertical))
            
            process_gaze_trajectory(normalized_ratio, normalized_vertical)
            
    key = cv2.waitKey(1)
    if key == 27: # ESC safely breaks out
        break

cap.release()
cv2.destroyAllWindows()
root.destroy()