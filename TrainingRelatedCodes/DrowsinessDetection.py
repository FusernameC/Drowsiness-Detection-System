# %%
# Import
import torch
import numpy as np
import cv2
from ultralytics import YOLO

import os

from roboflow import Roboflow
#%%
print(os.getcwd() + "\\yolov8n-face.pt")

# %%
# Load model
if __name__ == '__main__':
    # new training
    # model = YOLO("yolov8s.yaml")
    # model.train(data="Drowsiness-2\\data.yaml", epochs=100, batch=8, optimizer="Adam", amp=False)
    
    # load model (phai co model da train, sua lai duong dan theo may)
    model_face = YOLO(os.getcwd() + "\\yolov8n-face.pt") #\\runs\\detect\\train23\\weights\\best.pt
    model_drowsiness = torch.load(os.getcwd() + "\\class_saves\\drow.pt")
    # model.train(data="Drowsiness-2\\data.yaml", epochs=200, batch=8, optimizer="Adam", amp=False)
    
    # download dataset (tu len roboflow lay code)
    # rf = Roboflow(api_key="QPbl2J3ZyAhqkpABpfMD")
    # project = rf.workspace("dip-d0jtp").project("drowsiness-jwd6q")
    # version = project.version(1)
    # dataset = version.download("yolov8")

    
    # test val accuracy
    # metrics = model.val()
    # print("val start")
    # print(metrics.box.map50) 
    # print("val end")

    # test webcam
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = cv2.resize(frame, (640, 640))

        if success:
            # Face detection
            # Run YOLOv8 inference on the frame
            results = model_face.predict(frame, max_det=1)
            crops = results.crop()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Drwosiness detection
            if (len(crops) > 0):
                output = model_drowsiness(crops[0])
                print(output)
                

            # Break the loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    
    # test bang anh (tu sua duong dan)
    # img = cv2.imread(os.getcwd() + "\\awake-3a5ea1aa-f587-11ee-b094-3c7c3f1933fd_jpg.rf.4dcac6d2cd86d703e7475882160f024c.jpg")
    # # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # # img = cv2.resize(img, (640, 640))
    # results = model.predict(img, max_det=1)
    # annotated_frame = results[0].plot()
    # cv2.imshow("YOLOv8 Inference", annotated_frame)
    # cv2.waitKey(0)

# %%
