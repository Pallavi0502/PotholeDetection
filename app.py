import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO


# ---------------------------------------------------
# Load trained YOLOv8 model
# ---------------------------------------------------

model = YOLO(r"runs\detect\train\weights\best.pt")


# ---------------------------------------------------
# Colors
# ---------------------------------------------------

BG_COLOR = "#f4f6f8"
HEADER_COLOR = "#1f2937"
CARD_COLOR = "#ffffff"
TEXT_COLOR = "#1f2937"
SUBTEXT_COLOR = "#6b7280"
BUTTON_COLOR = "#2563eb"
BUTTON_HOVER = "#1d4ed8"
SUCCESS_COLOR = "#15803d"


# ---------------------------------------------------
# Main window
# ---------------------------------------------------

root = tk.Tk()

root.title("Pothole Detection | YOLOv8")
root.geometry("1100x750")
root.configure(bg=BG_COLOR)
root.resizable(False, False)


# ---------------------------------------------------
# Variables
# ---------------------------------------------------

current_image = None


# ---------------------------------------------------
# Functions
# ---------------------------------------------------

def select_image():

    global current_image

    file_path = filedialog.askopenfilename(
        title="Select Road Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png")
        ]
    )

    if not file_path:
        return

    try:

        status_label.config(
            text="Processing image...",
            fg=BUTTON_COLOR
        )

        root.update()

        # Run YOLO prediction
        results = model.predict(
            source=file_path,
            conf=0.25,
            verbose=False
        )

        result = results[0]

        # Draw bounding boxes
        plotted_image = result.plot()

        # Convert BGR → RGB
        plotted_image = plotted_image[:, :, ::-1]

        # Convert to PIL
        image = Image.fromarray(plotted_image)

        # Resize for display
        image.thumbnail((650, 470))

        current_image = ImageTk.PhotoImage(image)

        image_label.config(
            image=current_image,
            text=""
        )

        # Number of potholes
        count = len(result.boxes)

        if count == 0:

            result_title.config(
                text="No potholes detected",
                fg=SUCCESS_COLOR
            )

            result_label.config(
                text="The model did not detect any potholes\n"
                     "in the selected image."
            )

        else:

            result_title.config(
                text=f"{count} Pothole(s) Detected",
                fg=SUCCESS_COLOR
            )

            output = ""

            for i, box in enumerate(result.boxes):

                confidence = float(box.conf[0]) * 100

                output += (
                    f"Pothole {i + 1}"
                    f"        {confidence:.2f}% confidence\n\n"
                )

            result_label.config(
                text=output
            )

        status_label.config(
            text="Detection completed",
            fg=SUCCESS_COLOR
        )

    except Exception as e:

        messagebox.showerror(
            "Error",
            f"Something went wrong:\n\n{str(e)}"
        )

        status_label.config(
            text="Detection failed",
            fg="red"
        )


def clear_result():

    global current_image

    current_image = None

    image_label.config(
        image="",
        text="Select an image to begin detection",
        fg=SUBTEXT_COLOR
    )

    result_title.config(
        text="Detection Result",
        fg=TEXT_COLOR
    )

    result_label.config(
        text="No image selected"
    )

    status_label.config(
        text="Ready",
        fg=SUBTEXT_COLOR
    )


# ---------------------------------------------------
# Header
# ---------------------------------------------------

header = tk.Frame(
    root,
    bg=HEADER_COLOR,
    height=110
)

header.pack(
    fill="x"
)

title = tk.Label(
    header,
    text="POTHOLE DETECTION",
    font=("Segoe UI", 25, "bold"),
    fg="white",
    bg=HEADER_COLOR
)

title.pack(
    pady=(20, 2)
)

subtitle = tk.Label(
    header,
    text="YOLOv8 Computer Vision System",
    font=("Segoe UI", 12),
    fg="#d1d5db",
    bg=HEADER_COLOR
)

subtitle.pack()


# ---------------------------------------------------
# Main content
# ---------------------------------------------------

content = tk.Frame(
    root,
    bg=BG_COLOR
)

content.pack(
    fill="both",
    expand=True,
    padx=30,
    pady=25
)


# ---------------------------------------------------
# Left image card
# ---------------------------------------------------

image_card = tk.Frame(
    content,
    bg=CARD_COLOR,
    width=680,
    height=500,
    highlightbackground="#e5e7eb",
    highlightthickness=1
)

image_card.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 15)
)

image_card.pack_propagate(False)


image_heading = tk.Label(
    image_card,
    text="Road Image",
    font=("Segoe UI", 16, "bold"),
    fg=TEXT_COLOR,
    bg=CARD_COLOR
)

image_heading.pack(
    anchor="w",
    padx=20,
    pady=(18, 10)
)


image_label = tk.Label(
    image_card,
    text="Select an image to begin detection",
    font=("Segoe UI", 12),
    fg=SUBTEXT_COLOR,
    bg="#f9fafb"
)

image_label.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=(0, 20)
)


# ---------------------------------------------------
# Right result card
# ---------------------------------------------------

result_card = tk.Frame(
    content,
    bg=CARD_COLOR,
    width=330,
    height=500,
    highlightbackground="#e5e7eb",
    highlightthickness=1
)

result_card.pack(
    side="right",
    fill="y"
)

result_card.pack_propagate(False)


result_heading = tk.Label(
    result_card,
    text="Detection Result",
    font=("Segoe UI", 16, "bold"),
    fg=TEXT_COLOR,
    bg=CARD_COLOR
)

result_heading.pack(
    anchor="w",
    padx=20,
    pady=(18, 15)
)


result_title = tk.Label(
    result_card,
    text="Detection Result",
    font=("Segoe UI", 15, "bold"),
    fg=TEXT_COLOR,
    bg=CARD_COLOR
)

result_title.pack(
    pady=(20, 15)
)


result_label = tk.Label(
    result_card,
    text="No image selected",
    font=("Segoe UI", 12),
    fg=SUBTEXT_COLOR,
    bg=CARD_COLOR,
    justify="left"
)

result_label.pack(
    anchor="w",
    padx=30,
    pady=20
)


# ---------------------------------------------------
# Status
# ---------------------------------------------------

status_label = tk.Label(
    result_card,
    text="Ready",
    font=("Segoe UI", 11),
    fg=SUBTEXT_COLOR,
    bg=CARD_COLOR
)

status_label.pack(
    side="bottom",
    pady=20
)


# ---------------------------------------------------
# Buttons
# ---------------------------------------------------

button_frame = tk.Frame(
    root,
    bg=BG_COLOR
)

button_frame.pack(
    pady=(0, 25)
)


select_button = tk.Button(
    button_frame,
    text="  SELECT IMAGE  ",
    command=select_image,
    font=("Segoe UI", 12, "bold"),
    fg="white",
    bg=BUTTON_COLOR,
    activebackground=BUTTON_HOVER,
    activeforeground="white",
    relief="flat",
    cursor="hand2",
    padx=20,
    pady=10
)

select_button.pack(
    side="left",
    padx=8
)


clear_button = tk.Button(
    button_frame,
    text="  CLEAR  ",
    command=clear_result,
    font=("Segoe UI", 12, "bold"),
    fg=TEXT_COLOR,
    bg="#e5e7eb",
    activebackground="#d1d5db",
    relief="flat",
    cursor="hand2",
    padx=20,
    pady=10
)

clear_button.pack(
    side="left",
    padx=8
)


# ---------------------------------------------------
# Footer
# ---------------------------------------------------

footer = tk.Label(
    root,
    text="Powered by YOLOv8 • Pothole Detection",
    font=("Segoe UI", 9),
    fg=SUBTEXT_COLOR,
    bg=BG_COLOR
)

footer.pack(
    pady=(0, 10)
)


# ---------------------------------------------------
# Start application
# ---------------------------------------------------

root.mainloop()