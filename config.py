from datetime import timedelta

# RTSP / Video
RTSP_URL = "rtsp://admin:123456@192.168.1.77:554/H264?ch=1&subtype=0"
FRAME_W, FRAME_H = 1280, 720

# YOLO
MODEL_PATH = "best-3.pt"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# Counting
MIDDLE_LINE_Y = FRAME_H // 2
THRESHOLD = 60
MAX_POSITION_HISTORY = 6
COUNT_COOLDOWN = timedelta(milliseconds=500)
STALE_TIME = timedelta(seconds=2)

# Visualization
LINE_COLOR = (0, 255, 255)
ENTRY_COLOR = (0, 255, 0)
EXIT_COLOR = (0, 0, 255)

# Logging
DEBUG_PATH = "debug.txt"
