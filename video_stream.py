import cv2
import threading

class VideoStream:
    def __init__(self, src, width=None, height=None):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open video stream")
        self.frame = None
        self.stopped = False
        self.width = width
        self.height = height
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                if self.width and self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                self.frame = frame

    def read(self):
        return self.frame is not None, self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()
