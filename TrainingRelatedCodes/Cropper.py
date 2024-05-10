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
    model_face = YOLO(os.getcwd() + "\\yolov8n-face.pt") #\\runs\\detect\\train23\\weights\\best.pt
    
    folder = os.getcwd() + "\\Drowsiness-2\\train\\images"
    folderDest = os.getcwd() + "\\CustomClassification"
    for filename in os.listdir(folder):
        img = cv2.imread(folder + "\\" + filename)
        results = model_face.predict(img, max_det=1)
        boxes = results[0].boxes.xyxy.tolist()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # Crop the object using the bounding box coordinates
            ultralytics_crop_object = img[int(y1):int(y2), int(x1):int(x2)]
            # Save the cropped object as an image
            if ("drowsy" in filename):
                cv2.imwrite(folderDest + "\\1. Drowsy\\" + filename, ultralytics_crop_object)
            elif ("awake" in filename):
                cv2.imwrite(folderDest + "\\2. Awake\\" + filename, ultralytics_crop_object)