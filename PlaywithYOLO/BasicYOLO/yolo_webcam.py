import cv2
from ultralytics import YOLO

model = YOLO("yolov8m.pt")

video_path = 0 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

print("Processing video stream. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
      
    frame = cv2.flip(frame, 1)

    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confidence = float(box.conf[0])
            
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if confidence > 0.50 and class_name == "person":
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{class_name} {confidence:.2f}"

                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Advanced YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
