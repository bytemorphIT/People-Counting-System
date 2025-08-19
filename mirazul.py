"""
===========================================================================
                        PEOPLE COUNTING SYSTEM - TODOs
===========================================================================

1. Code Review & Refactoring
   - Review the entire modular codebase:
     config.py, video_stream.py, detector.py, tracker_state.py,
     counting.py, vizualization.py, logger.py, main.py, utils.py.
   - Ensure clean separation of concerns and consistent naming.
   - Apply PEP8 formatting and best practices.
   - Handle exceptions (e.g., missing RTSP feed, failed model load).

2. Multi-Line Entry/Exit Detection
   - Add support for 2 additional lines for entry/exit detection.
   - Make line positions configurable in config.py.
   - Ensure accurate counting when a person crosses any line.
   - Implement thresholds and dead zones for line crossing detection.

3. Pseudocode for Line Crossing Logic
   - Detect upward and downward movement across each line.
   - Respect COUNT_COOLDOWN between successive counts.
   - Remove stale IDs based on STALE_TIME.
   - Track which line was crossed and update counts accordingly.

4. Logging & Debugging Enhancements
   - Improve debug logs to indicate the line that triggered entry/exit.
   - Record crossing timestamps and trajectory information for analysis.


===========================================================================
"""
import cv2
import threading
import numpy as np
import os
from ultralytics import YOLO
from datetime import datetime, timedelta
from collections import deque

# ─────────── CONFIG ───────────
RTSP_URL = "rtsp://admin:123456@192.168.1.77/H264?ch=1&subtype=1"
MODEL = YOLO("best-3.pt")  # your trained model
FRAME_W, FRAME_H = 1280, 720
CONFIDENCE_THRESHOLD = 0.5

# ─────────── STATE ────────────
person_states = {}
entry_count = 0
exit_count = 0

total_detection_conf_sum = 0.0
total_detection_conf_count = 0
total_person_conf_sum = 0.0
total_person_conf_count = 0

# ─────────── LINE & ZONES ───────
THRESHOLD = 180  # distance from center
middle_line_y = FRAME_H // 2
upper_line_y = middle_line_y - THRESHOLD   # Entry trigger line
lower_line_y = middle_line_y + THRESHOLD   # Exit trigger line

line_pts_middle = [(0, middle_line_y), (FRAME_W, middle_line_y)]
line_pts_upper = [(0, upper_line_y), (FRAME_W, upper_line_y)]
line_pts_lower = [(0, lower_line_y), (FRAME_W, lower_line_y)]

# ────── DEBUG LOG ──────
debug_path = "debug.txt"
with open(debug_path, "w") as f:
    f.write("DEBUG LOG for YOLO Person Counter\n")
    f.write("Purpose: Track detection and crossing accuracy.\n")
    f.write("=" * 60 + "\n")
    f.write(f"[{datetime.now()}] Debugging session started.\n")
    f.write("-" * 60 + "\n")

def save_count(ent, ext):
    fname = datetime.now().strftime("count_log_%Y-%m-%d.txt")
    with open(fname, "w") as f:
        f.write(f"People Entered: {ent}\nPeople Exited: {ext}\n")

# ────── VIDEO STREAM ─────
class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            print("Error: Unable to open video stream")
        self.frame = None
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def read(self):
        return self.frame is not None, self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

# ──────── MAIN LOOP ────────
stream = VideoStream(RTSP_URL)
cv2.namedWindow("Person Counter")

MAX_POSITION_HISTORY = 6
COUNT_COOLDOWN = timedelta(milliseconds=500)
STALE_TIME = timedelta(seconds=2)

while True:
    ok, frame = stream.read()
    if not ok or frame is None:
        print("Waiting for video feed...")
        cv2.waitKey(10)
        continue

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    results = MODEL.track(frame, persist=True, conf=CONFIDENCE_THRESHOLD, iou=0.5)[0]

    detection_confidences = []
    person_confidences = []
    current_time = datetime.now()

    if results.boxes.id is not None:
        for box, tid_raw in zip(results.boxes, results.boxes.id):
            cls_id = int(box.cls[0])
            class_name = MODEL.names[cls_id]
            confidence = float(box.conf[0])
            tid = int(tid_raw)

            if class_name != "person" or confidence < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detection_confidences.append(confidence)
            person_confidences.append(confidence)

            if tid not in person_states:
                person_states[tid] = {
                    'last_positions': deque([cy], maxlen=MAX_POSITION_HISTORY),
                    'last_count_time': datetime.min,
                    'last_seen': current_time,
                    'state': 'unknown'
                }
                continue

            state_info = person_states[tid]
            state_info['last_seen'] = current_time
            positions = state_info['last_positions']
            positions.append(cy)

            if len(positions) < 2:
                continue

            last_cy = positions[-2]
            current_cy = positions[-1]
            can_count = current_time - state_info['last_count_time'] > COUNT_COOLDOWN
            state = state_info['state']

            # Logging movement
            with open(debug_path, "a") as f:
                f.write(f"[{current_time}] ID {tid} | last_cy={last_cy}, current_cy={current_cy}, state={state}\n")

            # ENTRY (upward across upper line)
            if last_cy > upper_line_y >= current_cy and can_count:
                entry_count += 1
                state_info['state'] = "inside"
                state_info['last_count_time'] = current_time
                print(f"[ENTRY] ID {tid} | Total Entry: {entry_count}")
                with open(debug_path, "a") as f:
                    f.write(f"--> ENTRY COUNTED!\n")
                    f.write(f"    Entry Count: {entry_count}, Exit Count: {exit_count}\n")
                    f.write("-" * 40 + "\n")

            # EXIT (downward across lower line)
            elif last_cy < lower_line_y <= current_cy and can_count:
                exit_count += 1
                state_info['state'] = "outside"
                state_info['last_count_time'] = current_time
                print(f"[EXIT] ID {tid} | Total Exit: {exit_count}")
                with open(debug_path, "a") as f:
                    f.write(f"--> EXIT COUNTED!\n")
                    f.write(f"    Entry Count: {entry_count}, Exit Count: {exit_count}\n")
                    f.write("-" * 40 + "\n")

            # Draw trajectory
            for i in range(1, len(positions)):
                cv2.line(frame, (cx, positions[i - 1]), (cx, positions[i]), (200, 100, 255), 2)

            # Draw detections
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            cv2.putText(frame, f"ID {tid} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, class_name, (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Remove stale IDs
    stale_ids = [tid for tid, s in person_states.items() if current_time - s['last_seen'] > STALE_TIME]
    for tid in stale_ids:
        person_states.pop(tid)

    # Confidence tracking
    total_detection_conf_sum += sum(detection_confidences)
    total_detection_conf_count += len(detection_confidences)
    total_person_conf_sum += sum(person_confidences)
    total_person_conf_count += len(person_confidences)

    avg_detection_conf = (total_detection_conf_sum / total_detection_conf_count * 100) if total_detection_conf_count else 0
    avg_person_conf = (total_person_conf_sum / total_person_conf_count * 100) if total_person_conf_count else 0

    # ─────── UI / Overlay ───────
    # Draw reference lines
    cv2.line(frame, line_pts_middle[0], line_pts_middle[1], (0, 255, 255), 2)   # middle
    cv2.line(frame, line_pts_upper[0], line_pts_upper[1], (0, 255, 0), 2)       # entry trigger
    cv2.line(frame, line_pts_lower[0], line_pts_lower[1], (0, 0, 255), 2)       # exit trigger

    cv2.putText(frame, "ENTRY ↑", (10, upper_line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "EXIT ↓", (FRAME_W - 130, lower_line_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, f"Entered: {entry_count}", (10, FRAME_H - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Exited:  {exit_count}", (FRAME_W - 230, FRAME_H - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"Detection Conf (All): {avg_detection_conf:.1f}%", (10, FRAME_H - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Person Conf: {avg_person_conf:.1f}%", (10, FRAME_H - 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Person Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ────── CLEANUP ──────
save_count(entry_count, exit_count)
final_avg_detection_conf = (total_detection_conf_sum / total_detection_conf_count * 100) if total_detection_conf_count > 0 else 0
final_avg_person_conf = (total_person_conf_sum / total_person_conf_count * 100) if total_person_conf_count > 0 else 0

with open(debug_path, "a") as f:
    f.write("\n=== FINAL STATS ===\n")
    f.write(f"Total Crossings: {entry_count + exit_count}\n")
    f.write(f"Avg Detection Confidence (all): {final_avg_detection_conf:.2f}%\n")
    f.write(f"Avg Person Detection Confidence: {final_avg_person_conf:.2f}%\n")
    f.write("=" * 60 + "\n")

stream.stop()
cv2.destroyAllWindows()
