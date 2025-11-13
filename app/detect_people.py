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

    results = model(frame)[0]

    for z in zones:
        pts = np.array(z["points"], dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    alarm_now = False

    for r in results.boxes:
        cls = int(r.cls[0])
        if cls != 0:
            continue
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        for z in zones:
            pts = np.array(z["points"], dtype=np.int32)
            inside = cv2.pointPolygonTest(pts, (cx, cy), False)
            if inside >= 0:
                alarm_now = True

    if alarm_now:
        alarm = True
        last_seen = time.time()

    if alarm and time.time() - last_seen > 3:
        alarm = False

    if alarm:
        cv2.putText(frame, "ALARM!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    cv2.imshow("Restricted Area Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
