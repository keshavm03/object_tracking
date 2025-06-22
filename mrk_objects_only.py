import cv2
import numpy as np

# ✅ Function to detect moving objects
def get_tracked_objects(frame, fgbg):
    mask = fgbg.apply(frame)

    # Clean noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:  # Filter small movements
            x, y, w, h = cv2.boundingRect(cnt)
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            objects.append((cx, cy))

    return objects

# 📌 STEP 1: Video Path
video_path = "853889-hd_1920_1080_25fps.mp4"  # Replace with your video file
cap = cv2.VideoCapture(video_path)

# ✅ Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# 📽️ Process video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # ✅ Get moving object centers
    objects = get_tracked_objects(frame, fgbg)

    # ✅ Mark the objects
    for (x, y) in objects:
        cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)  # Green dot

    # ✅ Show the frame
    cv2.imshow("Moving Objects", frame)
    if cv2.waitKey(30) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
