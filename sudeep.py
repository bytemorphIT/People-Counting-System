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
from enum import Enum

# ─────────── CONFIG ───────────
RTSP_URL = "rtsp://admin:123456@192.168.1.77/H264?ch=1&subtype=0"
MODEL = YOLO("best-3.pt")
FRAME_W, FRAME_H = 1280, 720
CONFIDENCE_THRESHOLD = 0.5

# ─────────── ENHANCED LINE LOGIC ───────────
class Direction(Enum):
    ENTRY = "entry"
    EXIT = "exit"
    UNKNOWN = "unknown"

class PersonState(Enum):
    INSIDE = "inside"
    OUTSIDE = "outside" 
    CROSSING = "crossing"
    UNKNOWN = "unknown"

# Line configuration
MIDDLE_LINE_Y = FRAME_H // 2
CROSSING_BUFFER = 30  # Buffer zone around the line
ENTRY_ZONE = (0, MIDDLE_LINE_Y - CROSSING_BUFFER)  # Above line
EXIT_ZONE = (MIDDLE_LINE_Y + CROSSING_BUFFER, FRAME_H)  # Below line
CROSSING_ZONE = (MIDDLE_LINE_Y - CROSSING_BUFFER, MIDDLE_LINE_Y + CROSSING_BUFFER)

# State tracking
person_states = {}
entry_count = 0
exit_count = 0

# Performance tracking
total_detection_conf_sum = 0.0
total_detection_conf_count = 0
total_person_conf_sum = 0.0
total_person_conf_count = 0

# ──────── ENHANCED PERSON TRACKING CLASS ────────
class PersonTracker:
    def __init__(self, track_id, initial_y):
        self.track_id = track_id
        self.positions = deque([initial_y], maxlen=8)  # Increased history for better accuracy
        self.state = PersonState.UNKNOWN
        self.last_count_time = datetime.min
        self.last_seen = datetime.now()
        self.crossing_direction = Direction.UNKNOWN
        self.stable_positions = 0  # Count of consecutive stable positions
        self.min_crossing_distance = 40  # Minimum distance to travel for valid crossing
        
    def update_position(self, y_pos):
        self.positions.append(y_pos)
        self.last_seen = datetime.now()
        
        # Update stability counter
        if len(self.positions) >= 2:
            if abs(self.positions[-1] - self.positions[-2]) < 10:
                self.stable_positions += 1
            else:
                self.stable_positions = 0
    
    def get_movement_direction(self):
        """Enhanced direction detection with noise filtering"""
        if len(self.positions) < 3:
            return Direction.UNKNOWN
            
        # Use multiple points for more stable direction detection
        recent_positions = list(self.positions)[-5:]  # Last 5 positions
        if len(recent_positions) < 3:
            return Direction.UNKNOWN
            
        # Calculate weighted movement (more weight to recent positions)
        weighted_movement = 0
        total_weight = 0
        
        for i in range(1, len(recent_positions)):
            weight = i  # Increasing weight for more recent positions
            movement = recent_positions[i] - recent_positions[i-1]
            weighted_movement += movement * weight
            total_weight += weight
            
        if total_weight == 0:
            return Direction.UNKNOWN
            
        avg_movement = weighted_movement / total_weight
        
        # Threshold for significant movement
        if avg_movement > 5:
            return Direction.EXIT  # Moving down
        elif avg_movement < -5:
            return Direction.ENTRY  # Moving up
        else:
            return Direction.UNKNOWN
    
    def has_crossed_line(self):
        """Enhanced line crossing detection"""
        if len(self.positions) < 2:
            return False, Direction.UNKNOWN
            
        current_y = self.positions[-1]
        
        # Check if person has traveled sufficient distance
        y_range = max(self.positions) - min(self.positions)
        if y_range < self.min_crossing_distance:
            return False, Direction.UNKNOWN
            
        # Enhanced crossing logic with zone-based detection
        direction = self.get_movement_direction()
        
        if direction == Direction.ENTRY:
            # Check if person moved from exit zone to entry zone
            has_exit_history = any(pos > MIDDLE_LINE_Y + CROSSING_BUFFER for pos in self.positions)
            in_entry_zone = current_y < MIDDLE_LINE_Y - CROSSING_BUFFER
            
            if has_exit_history and in_entry_zone:
                return True, Direction.ENTRY
                
        elif direction == Direction.EXIT:
            # Check if person moved from entry zone to exit zone  
            has_entry_history = any(pos < MIDDLE_LINE_Y - CROSSING_BUFFER for pos in self.positions)
            in_exit_zone = current_y > MIDDLE_LINE_Y + CROSSING_BUFFER
            
            if has_entry_history and in_exit_zone:
                return True, Direction.EXIT
                
        return False, Direction.UNKNOWN
    
    def can_count(self, cooldown_period=timedelta(seconds=1)):
        """Check if enough time has passed since last count"""
        return datetime.now() - self.last_count_time > cooldown_period
    
    def is_stable(self, min_stable_frames=3):
        """Check if person has been relatively stable"""
        return self.stable_positions >= min_stable_frames

# ────── OPTIMIZED LINE CROSSING LOGIC ──────
def process_line_crossings(person_trackers):
    global entry_count, exit_count
    
    crossings_detected = []
    
    for tracker in person_trackers.values():
        if not tracker.can_count():
            continue
            
        has_crossed, direction = tracker.has_crossed_line()
        
        if has_crossed and direction != Direction.UNKNOWN:
            if direction == Direction.ENTRY:
                entry_count += 1
                tracker.state = PersonState.INSIDE
                tracker.last_count_time = datetime.now()
                crossings_detected.append(("ENTRY", tracker.track_id))
                
            elif direction == Direction.EXIT:
                exit_count += 1
                tracker.state = PersonState.OUTSIDE  
                tracker.last_count_time = datetime.now()
                crossings_detected.append(("EXIT", tracker.track_id))
    
    return crossings_detected

# ────── ENHANCED DEBUG LOGGING ──────
debug_path = "debug_optimized.txt"
with open(debug_path, "w") as f:
    f.write("OPTIMIZED DEBUG LOG for YOLO Person Counter\n")
    f.write("Enhanced line crossing logic with zone-based detection\n")
    f.write("=" * 70 + "\n")
    f.write(f"[{datetime.now()}] Debugging session started.\n")
    f.write("-" * 70 + "\n")

def log_crossing_event(event_type, track_id, tracker):
    with open(debug_path, "a") as f:
        f.write(f"[{datetime.now()}] {event_type} - ID {track_id}\n")
        f.write(f"    Position history: {list(tracker.positions)}\n")
        f.write(f"    Movement direction: {tracker.get_movement_direction()}\n")
        f.write(f"    Y range traveled: {max(tracker.positions) - min(tracker.positions)}\n")
        f.write(f"    Entry count: {entry_count}, Exit count: {exit_count}\n")
        f.write("-" * 50 + "\n")

# ────── VIDEO STREAM (SAME AS ORIGINAL) ──────
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

# ──────── MAIN LOOP WITH OPTIMIZED LOGIC ────────
stream = VideoStream(RTSP_URL)
cv2.namedWindow("Optimized Person Counter")

STALE_TIME = timedelta(seconds=3)
person_trackers = {}

while True:
    ok, frame = stream.read()
    if not ok or frame is None:
        print("Waiting for video feed...")
        cv2.waitKey(10)
        continue

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    results = MODEL.track(frame, persist=True, conf=CONFIDENCE_THRESHOLD, iou=0.5)[0]

    current_time = datetime.now()
    active_tracks = set()

    # Process detections
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

            active_tracks.add(tid)

            # Update or create tracker
            if tid not in person_trackers:
                person_trackers[tid] = PersonTracker(tid, cy)
            else:
                person_trackers[tid].update_position(cy)

            # Confidence tracking
            total_detection_conf_sum += confidence
            total_detection_conf_count += 1
            total_person_conf_sum += confidence
            total_person_conf_count += 1

            # Draw enhanced visualization
            tracker = person_trackers[tid]
            
            # Color coding based on zone
            if cy < MIDDLE_LINE_Y - CROSSING_BUFFER:
                zone_color = (0, 255, 0)  # Green for entry zone
                zone_text = "ENTRY ZONE"
            elif cy > MIDDLE_LINE_Y + CROSSING_BUFFER:
                zone_color = (0, 0, 255)  # Red for exit zone  
                zone_text = "EXIT ZONE"
            else:
                zone_color = (0, 255, 255)  # Yellow for crossing zone
                zone_text = "CROSSING"

            cv2.rectangle(frame, (x1, y1), (x2, y2), zone_color, 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            
            # Enhanced info display
            cv2.putText(frame, f"ID {tid} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, zone_text, (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, zone_color, 1)
            
            # Draw trajectory with direction arrow
            positions = list(tracker.positions)
            if len(positions) > 1:
                for i in range(1, len(positions)):
                    alpha = i / len(positions)  # Fade older positions
                    color = (int(200 * alpha), int(100 * alpha), int(255 * alpha))
                    cv2.line(frame, (cx, positions[i-1]), (cx, positions[i]), color, 2)

    # Process line crossings
    crossings = process_line_crossings(person_trackers)
    
    # Log crossings
    for event_type, track_id in crossings:
        log_crossing_event(event_type, track_id, person_trackers[track_id])
        print(f"[{event_type}] ID {track_id} | Entry: {entry_count}, Exit: {exit_count}")

    # Remove stale trackers
    stale_ids = [tid for tid, tracker in person_trackers.items() 
                 if current_time - tracker.last_seen > STALE_TIME]
    for tid in stale_ids:
        person_trackers.pop(tid)

    # Calculate average confidences
    avg_detection_conf = (total_detection_conf_sum / total_detection_conf_count * 100) if total_detection_conf_count else 0
    avg_person_conf = (total_person_conf_sum / total_person_conf_count * 100) if total_person_conf_count else 0

    # Enhanced UI with zone visualization
    # Draw zones
    cv2.rectangle(frame, (0, 0), (FRAME_W, MIDDLE_LINE_Y - CROSSING_BUFFER), (0, 100, 0), 2)  # Entry zone
    cv2.rectangle(frame, (0, MIDDLE_LINE_Y + CROSSING_BUFFER), (FRAME_W, FRAME_H), (0, 0, 100), 2)  # Exit zone
    
    # Main crossing line
    cv2.line(frame, (0, MIDDLE_LINE_Y), (FRAME_W, MIDDLE_LINE_Y), (0, 255, 255), 3)
    
    # Buffer zone lines
    cv2.line(frame, (0, MIDDLE_LINE_Y - CROSSING_BUFFER), (FRAME_W, MIDDLE_LINE_Y - CROSSING_BUFFER), (255, 255, 0), 1)
    cv2.line(frame, (0, MIDDLE_LINE_Y + CROSSING_BUFFER), (FRAME_W, MIDDLE_LINE_Y + CROSSING_BUFFER), (255, 255, 0), 1)

    # Labels and counts
    cv2.putText(frame, "ENTRY ↑", (10, MIDDLE_LINE_Y - CROSSING_BUFFER - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "EXIT ↓", (FRAME_W - 130, MIDDLE_LINE_Y + CROSSING_BUFFER + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Enhanced stats display
    cv2.putText(frame, f"Entered: {entry_count}", (10, FRAME_H - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Exited: {exit_count}", (FRAME_W - 200, FRAME_H - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Net Count: {entry_count - exit_count}", (FRAME_W//2 - 100, FRAME_H - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"Active Tracks: {len(person_trackers)}", (10, FRAME_H - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Avg Confidence: {avg_person_conf:.1f}%", (10, FRAME_H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Optimized Person Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ────── CLEANUP & FINAL STATS ──────
def save_count(ent, ext):
    fname = datetime.now().strftime("count_log_%Y-%m-%d.txt")
    with open(fname, "w") as f:
        f.write(f"=== Optimized Person Counter Results ===\n")
        f.write(f"People Entered: {ent}\n")
        f.write(f"People Exited: {ext}\n")
        f.write(f"Net Count: {ent - ext}\n")
        f.write(f"Total Crossings: {ent + ext}\n")

save_count(entry_count, exit_count)

final_avg_detection_conf = (total_detection_conf_sum / total_detection_conf_count * 100) if total_detection_conf_count > 0 else 0
final_avg_person_conf = (total_person_conf_sum / total_person_conf_count * 100) if total_person_conf_count > 0 else 0

with open(debug_path, "a") as f:
    f.write("\n=== FINAL OPTIMIZED STATS ===\n")
    f.write(f"Total Entries: {entry_count}\n")
    f.write(f"Total Exits: {exit_count}\n") 
    f.write(f"Net Count: {entry_count - exit_count}\n")
    f.write(f"Total Crossings: {entry_count + exit_count}\n")
    f.write(f"Average Detection Confidence: {final_avg_detection_conf:.2f}%\n")
    f.write(f"Average Person Detection Confidence: {final_avg_person_conf:.2f}%\n")
    f.write("=" * 70 + "\n")

print(f"\nFinal Results:")
print(f"Entries: {entry_count}, Exits: {exit_count}, Net: {entry_count - exit_count}")
print(f"Average Person Detection Confidence: {final_avg_person_conf:.2f}%")

stream.stop()
cv2.destroyAllWindows()
