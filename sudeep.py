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
from ultralytics import YOLO
from datetime import datetime, timedelta
from collections import deque
import time


class Config:
    """Configuration constants"""
    RTSP_URL = "rtsp://admin:123456@192.168.1.77/H264?ch=1&subtype=1"
    MODEL_PATH = "best-3.pt"
    FRAME_W, FRAME_H = 1280, 720
    CONFIDENCE_THRESHOLD = 0.5
    THRESHOLD = 180  # distance from center
    MAX_POSITION_HISTORY = 6
    COUNT_COOLDOWN = timedelta(milliseconds=500)
    STALE_TIME = timedelta(seconds=2)
    IOU_THRESHOLD = 0.5


class PersonTracker:
    """Handles person tracking and counting logic"""

    def __init__(self):
        self.person_states = {}
        self.entry_count = 0
        self.exit_count = 0

        # Line coordinates
        self.middle_line_y = Config.FRAME_H // 2
        self.upper_line_y = self.middle_line_y - Config.THRESHOLD
        self.lower_line_y = self.middle_line_y + Config.THRESHOLD

        # Pre-computed line points
        self.line_pts = {
            'middle': [(0, self.middle_line_y), (Config.FRAME_W, self.middle_line_y)],
            'upper': [(0, self.upper_line_y), (Config.FRAME_W, self.upper_line_y)],
            'lower': [(0, self.lower_line_y), (Config.FRAME_W, self.lower_line_y)]
        }

        # Debug logging
        self._init_debug_log()

    def _init_debug_log(self):
        """Initialize debug logging"""
        self.debug_path = "debug.txt"
        with open(self.debug_path, "w") as f:
            f.write("DEBUG LOG for YOLO Person Counter\n")
            f.write("Purpose: Track detection and crossing accuracy.\n")
            f.write("=" * 60 + "\n")
            f.write(f"[{datetime.now()}] Debugging session started.\n")
            f.write("-" * 60 + "\n")

    def update_tracking(self, detections, current_time):
        """Update person tracking and count crossings"""
        for detection in detections:
            self._process_detection(detection, current_time)

        # Clean up stale tracking IDs
        self._cleanup_stale_ids(current_time)

    def _process_detection(self, detection, current_time):
        """Process individual detection"""
        tid, cx, cy, confidence = detection

        if tid not in self.person_states:
            self.person_states[tid] = {
                'last_positions': deque([cy], maxlen=Config.MAX_POSITION_HISTORY),
                'last_count_time': datetime.min,
                'last_seen': current_time,
                'state': 'unknown'
            }
            return

        state_info = self.person_states[tid]
        state_info['last_seen'] = current_time
        positions = state_info['last_positions']
        positions.append(cy)

        if len(positions) >= 2:
            self._check_line_crossing(tid, positions, state_info, current_time)

    def _check_line_crossing(self, tid, positions, state_info, current_time):
        """Check if person crossed counting lines"""
        last_cy = positions[-2]
        current_cy = positions[-1]
        can_count = current_time - state_info['last_count_time'] > Config.COUNT_COOLDOWN

        # Log movement
        with open(self.debug_path, "a") as f:
            f.write(
                f"[{current_time}] ID {tid} | last_cy={last_cy}, current_cy={current_cy}, state={state_info['state']}\n")

        if not can_count:
            return

        # Entry detection (upward crossing)
        if last_cy > self.upper_line_y >= current_cy:
            self._count_entry(tid, state_info, current_time)

        # Exit detection (downward crossing)
        elif last_cy < self.lower_line_y <= current_cy:
            self._count_exit(tid, state_info, current_time)

    def _count_entry(self, tid, state_info, current_time):
        """Count entry event"""
        self.entry_count += 1
        state_info['state'] = "inside"
        state_info['last_count_time'] = current_time
        print(f"[ENTRY] ID {tid} | Total Entry: {self.entry_count}")

        with open(self.debug_path, "a") as f:
            f.write(f"--> ENTRY COUNTED!\n")
            f.write(f"    Entry Count: {self.entry_count}, Exit Count: {self.exit_count}\n")
            f.write("-" * 40 + "\n")

    def _count_exit(self, tid, state_info, current_time):
        """Count exit event"""
        self.exit_count += 1
        state_info['state'] = "outside"
        state_info['last_count_time'] = current_time
        print(f"[EXIT] ID {tid} | Total Exit: {self.exit_count}")

        with open(self.debug_path, "a") as f:
            f.write(f"--> EXIT COUNTED!\n")
            f.write(f"    Entry Count: {self.entry_count}, Exit Count: {self.exit_count}\n")
            f.write("-" * 40 + "\n")

    def _cleanup_stale_ids(self, current_time):
        """Remove stale tracking IDs"""
        stale_ids = [
            tid for tid, state in self.person_states.items()
            if current_time - state['last_seen'] > Config.STALE_TIME
        ]
        for tid in stale_ids:
            self.person_states.pop(tid)


class ConfidenceTracker:
    """Tracks detection confidence statistics"""

    def __init__(self):
        self.total_detection_conf_sum = 0.0
        self.total_detection_conf_count = 0
        self.total_person_conf_sum = 0.0
        self.total_person_conf_count = 0

    def update(self, detection_confidences, person_confidences):
        """Update confidence statistics"""
        if detection_confidences:
            self.total_detection_conf_sum += sum(detection_confidences)
            self.total_detection_conf_count += len(detection_confidences)

        if person_confidences:
            self.total_person_conf_sum += sum(person_confidences)
            self.total_person_conf_count += len(person_confidences)

    @property
    def avg_detection_conf(self):
        """Get average detection confidence"""
        if self.total_detection_conf_count == 0:
            return 0
        return (self.total_detection_conf_sum / self.total_detection_conf_count) * 100

    @property
    def avg_person_conf(self):
        """Get average person confidence"""
        if self.total_person_conf_count == 0:
            return 0
        return (self.total_person_conf_sum / self.total_person_conf_count) * 100


class VideoStream:
    """Threaded video stream handler"""

    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise ConnectionError("Unable to open video stream")

        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """Update frame in separate thread"""
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage

    def read(self):
        """Read current frame"""
        with self.lock:
            return self.frame is not None, self.frame

    def stop(self):
        """Stop video stream"""
        self.stopped = True
        self.thread.join()
        self.cap.release()


class Visualizer:
    """Handles frame visualization and UI overlay"""

    def __init__(self, tracker):
        self.tracker = tracker

        # Pre-defined colors
        self.colors = {
            'detection_box': (0, 255, 0),
            'center_point': (255, 0, 0),
            'trajectory': (200, 100, 255),
            'middle_line': (0, 255, 255),
            'entry_line': (0, 255, 0),
            'exit_line': (0, 0, 255),
            'text_white': (255, 255, 255),
            'text_yellow': (255, 255, 0)
        }

    def draw_frame(self, frame, detections, conf_tracker):
        """Draw all visualizations on frame"""
        self._draw_reference_lines(frame)
        self._draw_detections(frame, detections)
        self._draw_statistics(frame, conf_tracker)
        return frame

    def _draw_reference_lines(self, frame):
        """Draw counting reference lines"""
        # Draw lines
        cv2.line(frame, *self.tracker.line_pts['middle'], self.colors['middle_line'], 2)
        cv2.line(frame, *self.tracker.line_pts['upper'], self.colors['entry_line'], 2)
        cv2.line(frame, *self.tracker.line_pts['lower'], self.colors['exit_line'], 2)

        # Draw labels
        cv2.putText(frame, "ENTRY ↑", (10, self.tracker.upper_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['entry_line'], 2)
        cv2.putText(frame, "EXIT ↓", (Config.FRAME_W - 130, self.tracker.lower_line_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['exit_line'], 2)

    def _draw_detections(self, frame, detections):
        """Draw detection boxes and trajectories"""
        for tid, cx, cy, confidence, bbox in detections:
            x1, y1, x2, y2 = bbox

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['detection_box'], 2)

            # Draw center point
            cv2.circle(frame, (cx, cy), 4, self.colors['center_point'], -1)

            # Draw trajectory
            if tid in self.tracker.person_states:
                positions = self.tracker.person_states[tid]['last_positions']
                for i in range(1, len(positions)):
                    cv2.line(frame, (cx, positions[i - 1]), (cx, positions[i]),
                             self.colors['trajectory'], 2)

            # Draw labels
            cv2.putText(frame, f"ID {tid} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text_white'], 2)
            cv2.putText(frame, "person", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['detection_box'], 2)

    def _draw_statistics(self, frame, conf_tracker):
        """Draw counting statistics"""
        # Count display
        cv2.putText(frame, f"Entered: {self.tracker.entry_count}", (10, Config.FRAME_H - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['entry_line'], 2)
        cv2.putText(frame, f"Exited:  {self.tracker.exit_count}", (Config.FRAME_W - 230, Config.FRAME_H - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['exit_line'], 2)

        # Confidence display
        cv2.putText(frame, f"Detection Conf (All): {conf_tracker.avg_detection_conf:.1f}%",
                    (10, Config.FRAME_H - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text_yellow'], 2)
        cv2.putText(frame, f"Person Conf: {conf_tracker.avg_person_conf:.1f}%",
                    (10, Config.FRAME_H - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['entry_line'], 2)


def save_count(entry_count, exit_count):
    """Save count to file"""
    fname = datetime.now().strftime("count_log_%Y-%m-%d.txt")
    with open(fname, "w") as f:
        f.write(f"People Entered: {entry_count}\nPeople Exited: {exit_count}\n")


def extract_detections(results):
    """Extract detection data from YOLO results"""
    detections = []
    detection_confidences = []
    person_confidences = []

    if results.boxes.id is None:
        return detections, detection_confidences, person_confidences

    for box, tid_raw in zip(results.boxes, results.boxes.id):
        cls_id = int(box.cls[0])
        class_name = results.names[cls_id]
        confidence = float(box.conf[0])
        tid = int(tid_raw)

        detection_confidences.append(confidence)

        if class_name != "person" or confidence < Config.CONFIDENCE_THRESHOLD:
            continue

        person_confidences.append(confidence)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        detections.append((tid, cx, cy, confidence, (x1, y1, x2, y2)))

    return detections, detection_confidences, person_confidences


def main():
    """Main execution function"""
    # Initialize components
    model = YOLO(Config.MODEL_PATH)
    tracker = PersonTracker()
    conf_tracker = ConfidenceTracker()
    visualizer = Visualizer(tracker)

    try:
        stream = VideoStream(Config.RTSP_URL)
        cv2.namedWindow("Person Counter", cv2.WINDOW_NORMAL)

        print("Person counter started. Press 'q' to quit.")

        while True:
            ok, frame = stream.read()
            if not ok or frame is None:
                print("Waiting for video feed...")
                cv2.waitKey(10)
                continue

            # Resize frame
            frame = cv2.resize(frame, (Config.FRAME_W, Config.FRAME_H))

            # Run YOLO detection
            results = model.track(frame, persist=True, conf=Config.CONFIDENCE_THRESHOLD,
                                  iou=Config.IOU_THRESHOLD)[0]

            # Extract detection data
            detections, detection_confs, person_confs = extract_detections(results)

            # Update tracking and confidence stats
            current_time = datetime.now()
            tracker.update_tracking([(d[0], d[1], d[2], d[3]) for d in detections], current_time)
            conf_tracker.update(detection_confs, person_confs)

            # Draw visualizations
            frame = visualizer.draw_frame(frame, detections, conf_tracker)

            # Display frame
            cv2.imshow("Person Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup and save results
        save_count(tracker.entry_count, tracker.exit_count)

        # Write final stats to debug log
        with open(tracker.debug_path, "a") as f:
            f.write("\n=== FINAL STATS ===\n")
            f.write(f"Total Crossings: {tracker.entry_count + tracker.exit_count}\n")
            f.write(f"Avg Detection Confidence (all): {conf_tracker.avg_detection_conf:.2f}%\n")
            f.write(f"Avg Person Detection Confidence: {conf_tracker.avg_person_conf:.2f}%\n")
            f.write("=" * 60 + "\n")

        print(f"\nFinal Results:")
        print(f"Entries: {tracker.entry_count}, Exits: {tracker.exit_count}")
        print(f"Average Detection Confidence: {conf_tracker.avg_detection_conf:.2f}%")

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'stream' in locals():
            stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
