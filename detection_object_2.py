from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")  # tiny & fast, or use yolov8s.pt for more accuracy

cap = cv2.VideoCapture("853889-hd_1920_1080_25fps.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # Run detection
    results = model(frame)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        if model.names[cls] == 'person':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

    cv2.imshow("YOLOv8 Person Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
