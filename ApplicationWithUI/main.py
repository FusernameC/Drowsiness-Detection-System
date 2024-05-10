from tkinter import Tk, Canvas, Entry, Button, PhotoImage, filedialog, Label, StringVar
from PIL import Image, ImageTk
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
import timm
import torch.nn as nn

# Define
class DrowsinessClassifier(nn.Module):
        def __init__(self, numclasses=2):
            super(DrowsinessClassifier, self).__init__()
            self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
            self.features = nn.Sequential(*list(self.base_model.children())[:-1])

            enet_out_site=1280
            # Make a classifier
            self.classifier = nn.Linear(enet_out_site, numclasses)
            
        def forward(self, x):
            x = self.features(x)
            output = self.classifier(x)
            return output

# Khởi tạo đường dẫn tài nguyên
ASSETS_PATH = os.path.join(os.getcwd(), "assets")

def relative_to_assets(path: str) -> str:
    return os.path.join(ASSETS_PATH, path)

# Khởi tạo giao diện
window = Tk()
window.title("Object Detection GUI")
# Lấy kích thước màn hình
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Tính toán vị trí cho cửa sổ
x = int((screen_width / 2) - (1219 / 2))
y = int((screen_height / 2) - (741 / 2))

# Cài đặt kích thước và vị trí của cửa sổ
window.geometry("1219x741")
window.configure(bg="#FFFFFF")

canvas = Canvas(window, bg="#FFFFFF", height=500, width=900, bd=0, highlightthickness=0, relief="ridge")
canvas.pack()

# Khởi tạo model nhận diện khuôn mặt và mệt mỏi
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_face = YOLO(os.getcwd() + "\\yolov8n-face.pt")
model_drowsiness = torch.load(os.getcwd() + "\\drow.pt", map_location=torch.device('cpu'))
model_drowsiness.to(device)
model_drowsiness.eval()

TRANSFORM_IMG = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Trạng thái camera và video
cap = None
is_camera_video_open = False
image_widget = None

# Chức năng mở file hình ảnh
def open_image():
    if is_camera_video_open:
        close_camera_video()
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    if file_path:  # Make sure a file was selected
        image = cv2.imread(file_path)
        detect_and_display(image)

# Chức năng mở file video
def open_video():
    global cap, is_camera_video_open
    if is_camera_video_open:
        close_camera_video()
    file_path = filedialog.askopenfilename(
        title="Select a video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    if file_path:  # Make sure a file was selected
        cap = cv2.VideoCapture(file_path)
        is_camera_video_open = True
        read_and_process_video(cap)

# Chức năng mở camera
def open_camera():
    global cap, is_camera_video_open
    if is_camera_video_open:
        close_camera_video()
    cap = cv2.VideoCapture(0)
    is_camera_video_open = True
    read_and_process_video(cap)

def close_camera_video():
    global cap, is_camera_video_open, image_widget
    entry_1_var.set("")
    if cap:
        cap.release()
    cap = None
    is_camera_video_open = False
    cv2.destroyAllWindows()
    if image_widget:
        canvas.delete(image_widget)
        image_widget = None

# Hiển thị hình ảnh lên canvas
def display_image(image):
    global image_widget
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = ImageTk.PhotoImage(image)
    canvas.image = image
    if image_widget:
        canvas.itemconfig(image_widget, image=image)
    else:
        image_widget = canvas.create_image(606, 422, image=image)

# Nhận diện và hiển thị
def detect_and_display(frame):
    frame = cv2.resize(frame, (763, 534))
    results = model_face.predict(frame)
    annotated_frame = results[0].plot()
    display_image(annotated_frame)
    detect_drowsiness(frame)

# Nhận diện mệt mỏi
def detect_drowsiness(frame):
    results = model_face.predict(frame)
    boxes = results[0].boxes.xyxy
    if len(boxes) > 0:
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box)
        crop_img = frame[y1:y2, x1:x2]
        crop_img = TRANSFORM_IMG(crop_img).unsqueeze(0).to(device)
        output = model_drowsiness(crop_img)
        index = output.argmax()
        if index == 0:
            entry_1_var.set("DROWSY")
        elif index == 1:
            entry_1_var.set("AWAKE")
    else:
        entry_1_var.set("NO FACE DETECTED")

# Xử lý video và camera
def read_and_process_video(cap):
    def process_frame():
        ret, frame = cap.read()
        if ret and is_camera_video_open:
            detect_and_display(frame)
            window.after(10, process_frame)
        else:
            cap.release()
            cv2.destroyAllWindows()
            if image_widget:
                canvas.delete(image_widget)

    process_frame()

# Các nút bấm trên GUI
canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 741,
    width = 1219,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.create_rectangle(
    0.0,
    0.0,
    1219.0,
    111.0,
    fill="#2E6A95",
    outline="")

canvas.create_text(
    329.0,
    37.0,
    anchor="nw",
    text="DROWSY DETECTION",
    fill="#FFFFFF",
    font=("Inter BlackItalic", 50 * -1)
)

canvas.place(x = 0, y = 0)
button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=open_camera,
    relief="flat"
)
button_1.place(
    x=20.0,
    y=481.0,
    width=173.0,
    height=46.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=open_video,
    relief="flat"
)
button_2.place(
    x=20.0,
    y=370.0,
    width=173.0,
    height=46.0
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=open_image,
    relief="flat"
)
button_3.place(
    x=20.0,
    y=260.0,
    width=173.0,
    height=46.0
)

image_image_1 = PhotoImage(
    file=relative_to_assets("image.png"))
image_1 = canvas.create_image(
    606.0,
    422.0,
    image=image_image_1
)

entry_1_var = StringVar()

entry_1 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0,
    textvariable=entry_1_var,
    state='disabled',
    justify='center')
entry_1.pack()
entry_1.place(
    x=1036.0,
    y=373.0,
    width=139.0,
    height=39.0
)

canvas.create_text(
    1046.0,
    338.0,
    anchor="nw",
    text="Result",
    fill="#1C4574",
    font=("Inter Bold", 20 * -1)
)

window.mainloop()
