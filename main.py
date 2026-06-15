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

# Create folder for screenshots
save_folder = "Detected_Mobile_Images"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Alert and screenshot control
alert_played = False
screenshot_taken = False

print("System started. Press 'q' to quit.")

while True:

    ret, frame = cap.read()

    # Retry a few times before giving up
    if not ret or frame is None:
        print("Warning: Failed to grab frame, retrying...")
        cap.release()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Error: Could not reconnect to webcam.")
            break
        continue

    # Resize frame for faster performance
    frame = cv2.resize(frame, (640, 480))

    # Run YOLO detection
    try:
        results = model(frame, imgsz=320, verbose=False)
    except Exception as e:
        print(f"YOLO Error: {e}")
        continue

    # Get detected objects
    boxes = results[0].boxes

    mobile_detected = False

    for box in boxes:

        # Class ID
        cls = int(box.cls[0])

        # Get class name
        class_name = model.names[cls]

        # Confidence score
        conf = float(box.conf[0])

        # Detect only cell phone
        if class_name == "cell phone" and conf > 0.55:

            mobile_detected = True

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw rectangle
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            # Label
            label = f"Mobile Detected {conf:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 10)),  # Prevent label going off screen
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            # Save screenshot only once
            if not screenshot_taken:

                # Create timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Screenshot filename
                filename = os.path.join(save_folder, f"mobile_{timestamp}.jpg")

                # Save screenshot
                success = cv2.imwrite(filename, frame)

                if success:
                    print(f"Screenshot Saved: {filename}")
                else:
                    print("Warning: Failed to save screenshot.")

                screenshot_taken = True

    # Play alert sound
    if mobile_detected and not alert_played:

        print("WARNING: MOBILE PHONE DETECTED!")

        try:
            winsound.Beep(1000, 700)
        except Exception as e:
            print(f"Sound Error: {e}")

        alert_played = True

    # Reset alert and screenshot flag
    if not mobile_detected:
        alert_played = False
        screenshot_taken = False

    # Show webcam output
    cv2.imshow("Mobile Phone Detection Alert System", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("System stopped.")