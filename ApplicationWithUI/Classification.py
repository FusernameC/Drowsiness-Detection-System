#%%
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import timm
import cv2
from ultralytics import YOLO
import torch.nn.functional as F


#%%
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EPOCHS = 10
    BATCH_SIZE = 8
    LEARNING_RATE = 0.003
    TRAIN_DATA_PATH = "./Drow_Class/dataset/train/"
    TEST_DATA_PATH = "./Drow_Class/dataset/test/"
    TRANSFORM_IMG = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])


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

    train_mode = 0
    if (train_mode == 1):
        dataset = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
        total_count=len(dataset)
        train_count = int(0.7 * total_count)
        valid_count = int(0.2 * total_count)
        test_count = total_count - train_count - valid_count

        train_data, val_data, test_data = torch.utils.data.random_split(dataset, (train_count, valid_count, test_count))
        train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=8, pin_memory=True)
        val_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=8, pin_memory=True)
        test_data_loader  = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

        print("Number of train samples: ", len(train_data))
        print("Number of val samples: ", len(val_data))
        print("Number of test samples: ", len(test_data))
        print("Detected Classes are: ", dataset.class_to_idx) # classes are detected by folder structure
        model = DrowsinessClassifier(numclasses=2)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_losses, val_losses = [], []
        
        
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for images, labels in tqdm(train_data_loader, desc='Training loop'):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * labels.size(0)
            train_loss = running_loss / len(train_data_loader.dataset)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for images, labels in tqdm(val_data_loader, desc='Validation loop'):
                    # Move inputs and labels to the device
                    images, labels = images.to(device), labels.to(device)
                
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * labels.size(0)
            val_loss = running_loss / len(val_data_loader.dataset)
            val_losses.append(val_loss)
            print(f"Epoch {epoch+1}/{EPOCHS} - Train loss: {train_loss}, Validation loss: {val_loss}")
            torch.save(model, os.getcwd() + "\\class_saves\\drow.pt")

    model_face = YOLO(os.getcwd() + "\\yolov8n-face.pt") #\\runs\\detect\\train23\\weights\\best.pt
    # model_drowsiness = torch.load(os.getcwd() + "\\drow.pt")
    model_drowsiness = torch.load(os.getcwd() + "\\drow.pt", map_location=torch.device('cpu'))

    model_drowsiness.to(device)
    model_drowsiness.eval()
    
    # test bang webcam
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
            boxes = results[0].boxes.xyxy.tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            # Gan vao o hien hinh anh
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Drwosiness detection
            if (len(boxes) > 0):
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    # Crop the object using the bounding box coordinates
                    ultralytics_crop_object = frame[int(y1):int(y2), int(x1):int(x2)]
                    new_frame = TRANSFORM_IMG(frame)
                    new_frame = new_frame.to(device)
                output = model_drowsiness(new_frame.unsqueeze(0))
                index = output.data.cpu().numpy().argmax()
                # Gan vao o Result, doi print thanh function hien o o Result
                if (index == 0):
                    print("DROWSY")
                elif (index == 1):
                    print("AWAKE")
            else:
                print("NO FACE DETECTED")
                    
                

            # Break the loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    
    
# %%
