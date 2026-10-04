import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("video_assets/2.mp4")

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    results = model(frame, classes=[2], verbose=False)

    annotated_frame = results[0].plot()

    display_frame = cv2.resize(
        annotated_frame,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow("Car Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()