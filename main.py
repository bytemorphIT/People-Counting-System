import cv2
from config import FRAME_W, FRAME_H, STALE_TIME
from video_stream import VideoStream
from detector import Detector
from tracker_state import TrackerState
from counting import Counter
from vizualization import draw_frame
from logger import log_debug, save_count

stream = VideoStream(src="rtsp://admin:123456@192.168.1.77:554/H264?ch=1&subtype=0", width=FRAME_W, height=FRAME_H)
detector = Detector()
tracker = TrackerState()
counter = Counter()

cv2.namedWindow("Person Counter")

while True:
    ok, frame = stream.read()
    if not ok:
        cv2.waitKey(10)
        continue

    results = detector.track(frame)

    if results.boxes.id is not None:
        for box, tid_raw in zip(results.boxes, results.boxes.id):
            cls_id = int(box.cls[0])
            class_name = detector.model.names[cls_id]
            confidence = float(box.conf[0])
            tid = int(tid_raw)

            if class_name != "person" or confidence < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if tid not in tracker.person_states:
                tracker.init_person(tid, cy)
                continue

            tracker.update_position(tid, cy)
            state_info = tracker.person_states[tid]
            counter.check_crossing(tid, state_info['last_positions'], state_info)
            log_debug(f"ID {tid} | positions={list(state_info['last_positions'])} | state={state_info['state']}")

    tracker.remove_stale(STALE_TIME.total_seconds())
    frame = draw_frame(frame, results, tracker, counter)

    cv2.imshow("Person Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

save_count(counter.entry_count, counter.exit_count)
stream.stop()
cv2.destroyAllWindows()
