import cv2
from ultralytics import YOLO
from config import settings
from config import BASE_DIR

VIDEO_PATH = str(settings.VIDEO_PATH.resolve())

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for r in results.boxes:  
        cls = int(r.cls[0])
        conf = float(r.conf[0])
        if cls == 0:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.imshow("YOLO People Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
