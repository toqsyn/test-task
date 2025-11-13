import cv2
import json
import time
from ultralytics import YOLO
import numpy as np
from config import settings
from config import BASE_DIR

VIDEO_PATH = str(settings.VIDEO_PATH.resolve())
ZONES_JSON = str(settings.JSON_PATH.resolve())

model = YOLO("yolov8n.pt")

with open(ZONES_JSON, "r") as f:
    zones = json.load(f)["zones"]

cap = cv2.VideoCapture(VIDEO_PATH)
alarm = False
last_seen = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for z in zones:
        pts = np.array(z["points"], dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
        if "name" in z:
            cv2.putText(frame, z["name"], tuple(z["points"][0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    alarm_now = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            inside_zone = False
            for z in zones:
                pts = np.array(z["points"], dtype=np.int32)
                if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                    inside_zone = True
                    break

            color = (0, 0, 255) if inside_zone else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)

            if inside_zone:
                alarm_now = True

    if alarm_now:
        alarm = True
        last_seen = time.time()

    if alarm and time.time() - last_seen > 3:
        alarm = False

    if alarm:
        cv2.putText(frame, "ALERT: Person in restricted zone!",
                    (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 10)

    cv2.imshow("Restricted Area Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
