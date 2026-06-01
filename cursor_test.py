import tkinter as tk

root = tk.Tk()

root.title("EyeType AI")
root.configure(bg="black")

WIDTH = 1200
HEIGHT = 700

canvas = tk.Canvas(
    root,
    width=WIDTH,
    height=HEIGHT,
    bg="black",
    highlightthickness=0
)
canvas.pack()

dot = canvas.create_oval(
    580,
    330,
    620,
    370,
    fill="white"
)

cursor_x = WIDTH // 2

def update_cursor(ratio):
    global cursor_x

    cursor_x = int(ratio * WIDTH)

    canvas.coords(
        dot,
        cursor_x - 20,
        HEIGHT // 2 - 20,
        cursor_x + 20,
        HEIGHT // 2 + 20
    )

# TEST ANIMATION
ratio = 0

def animate():
    global ratio

    ratio += 0.01

    if ratio > 1:
        ratio = 0

    update_cursor(ratio)

    root.after(20, animate)

animate()

root.mainloop()