
import cv2
from config import LINE_COLOR, ENTRY_COLOR, EXIT_COLOR, MIDDLE_LINE_Y

def draw_frame(frame, results, tracker_states, counter):
    cv2.line(frame, (0, MIDDLE_LINE_Y), (frame.shape[1], MIDDLE_LINE_Y), LINE_COLOR, 2)
    cv2.putText(frame, "ENTRY ↑", (10, MIDDLE_LINE_Y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ENTRY_COLOR, 2)
    cv2.putText(frame, "EXIT ↓", (frame.shape[1]-130, MIDDLE_LINE_Y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, EXIT_COLOR, 2)
    cv2.putText(frame, f"Entered: {counter.entry_count}", (10, frame.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 1, ENTRY_COLOR, 2)
    cv2.putText(frame, f"Exited:  {counter.exit_count}", (frame.shape[1]-230, frame.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 1, EXIT_COLOR, 2)

    # Draw detections
    if results.boxes.id is not None:
        for box, tid_raw in zip(results.boxes, results.boxes.id):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            tid = int(tid_raw)
            cv2.rectangle(frame, (x1, y1), (x2, y2), ENTRY_COLOR, 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {tid} {box.conf[0]:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame
