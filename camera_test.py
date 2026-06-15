import cv2

# Try different camera indexes
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera 0 failed")

    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Camera 1 failed")

    cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("No webcam detected")
    exit()

print("Webcam opened successfully")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()