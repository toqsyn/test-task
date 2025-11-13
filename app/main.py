import cv2
import numpy as np
import json
from pathlib import Path
import sys
from config import settings
from config import BASE_DIR

VIDEO_PATH = str(settings.VIDEO_PATH.resolve())
OUT_JSON = (settings.BASE_DIR / "restricted_zones.json").resolve()

points = []
zones = []

def load_zones():
    if OUT_JSON.exists():
        try:
            data = json.loads(OUT_JSON.read_text())
            return data.get("zones", [])
        except Exception:
            return []
    return []

def save_zone(points):
    z = load_zones()
    next_id = (max([zz["id"] for zz in z]) + 1) if z else 1
    zone = {"id": next_id, "name": f"zone_{next_id}", "points": [[int(x), int(y)] for (x,y) in points]}
    z.append(zone)
    OUT_JSON.write_text(json.dumps({"zones": z}, indent=2))
    print(f"[saved] zone id={next_id} points={len(points)}")

def mouse_cb(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()

def draw_overlay(frame, pts, exist_zones):
    overlay = frame.copy()
    if len(pts) > 0:
        arr = np.array(pts, dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [arr], color=(0, 255, 0, 50))
        cv2.polylines(overlay, [arr], isClosed=False if len(pts) < 3 else True, color=(0, 255, 0), thickness=2)
        for p in pts:
            cv2.circle(overlay, p, 4, (0, 255, 0), -1)
    
    for z in exist_zones:
        pts_z = np.array(z["points"], dtype=np.int32)
        cv2.polylines(overlay, [pts_z], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(overlay, f"Z{id}", tuple(pts_z[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    cv2.putText(frame, "LMB add point | RMB undo | s save | r reset | q quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def main():
    global points
    cap = cv2.VideoCapture(VIDEO_PATH)
    # cap = cv2.VideoCapture("../assets/test.mp4")
    if not cap.isOpened():
        print("Can't open video:", VIDEO_PATH)
        return
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Can't read frame from video")
        return

    cv2.namedWindow("annotate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("annotate", mouse_cb)

    zones_on_disk = load_zones()

    while True:
        disp = frame.copy()
        draw_overlay(disp, points, zones_on_disk)
        cv2.imshow("annotate", disp)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("r"):
            points = []
            print("[reset]")
        elif key == ord("s"):
            if len(points) >= 3:
                save_zone(points)
                zones_on_disk = load_zones()
                points = []
            else:
                print("Need >= 3 points to save a polygon")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
