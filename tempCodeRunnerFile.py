from ultralytics import YOLO
import cv2
import winsound
import os
from datetime import datetime

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Cannot open webcam. Try changing index to 1 or remove CAP_DSHOW.")
    exit()

# Set webcam resolution
cap.set(3,640)
cap.set(4,480)