import cv2
import tkinter as tk
from mediapipe.python.solutions import face_mesh

mp_face_mesh = face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1
)

# Webcam.
cap = cv2.VideoCapture(0)


# CURSOR WINDOW
WIDTH = 1200
HEIGHT = 700

root = tk.Tk()
root.title("EyeType AI Cursor")

canvas = tk.Canvas(
    root,
    width=WIDTH,
    height=HEIGHT,
    bg="black",
    highlightthickness=0
)

canvas.pack()

cursor = canvas.create_oval(
    WIDTH // 2 - 20,
    HEIGHT // 2 - 20,
    WIDTH // 2 + 20,
    HEIGHT // 2 + 20,
    fill="white"
)

def move_cursor(ratio):

    x = int(ratio * WIDTH)

    canvas.coords(
        cursor,
        x - 20,
        HEIGHT // 2 - 20,
        x + 20,
        HEIGHT // 2 + 20
    )

    root.update()

smoothed_ratio = 0.5

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    gaze_text = "NO FACE"

    if results.multi_face_landmarks:

        landmarks = results.multi_face_landmarks[0].landmark

        h, w, _ = frame.shape

        # LEFT EYE CORNERS
        left_corner = landmarks[33]
        right_corner = landmarks[133]

        # LEFT IRIS CENTER
        iris = landmarks[468]

        left_x = int(left_corner.x * w)
        right_x = int(right_corner.x * w)
        iris_x = int(iris.x * w)

        # Draw eye landmarks
        cv2.circle(frame, (left_x, int(left_corner.y * h)), 3, (0, 255, 0), -1)
        cv2.circle(frame, (right_x, int(right_corner.y * h)), 3, (0, 255, 0), -1)
        cv2.circle(frame, (iris_x, int(iris.y * h)), 5, (0, 0, 255), -1)

        eye_width = right_x - left_x

        if eye_width > 0:

            ratio = (iris_x - left_x) / eye_width

            ratio = max(0, min(1, ratio))
            smoothed_ratio = (
                     smoothed_ratio * 0.8
                     + ratio * 0.2
                     )
            move_cursor(smoothed_ratio)

            # Tune these thresholds later
            if ratio < 0.40:
                gaze_text = "LEFT"

            elif ratio > 0.60:
                gaze_text = "RIGHT"

            else:
                gaze_text = "CENTER"

            cv2.putText(
                frame,
                f"Ratio: {ratio:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

    cv2.putText(
        frame,
        gaze_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("EyeType AI - Phase 1", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
root.destroy()