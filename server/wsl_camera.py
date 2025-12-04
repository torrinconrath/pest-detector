# wsl_camera.py
import cv2
import threading
import time
from shared_config import MOTION_URL, GlobalState

class WSLRemoteCamera:
    """Camera manager for WSL/Desktop which connects to Raspberry Pi MJPEG Motion stream."""

    def __init__(self, buffer_size=1):
        self.buffer_size = buffer_size
        self._frame = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._cap = None
        self._running = False
        self._thread = None
        self._frame_count = 0
        self._fps_history = []

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._cap:
            try:
                self._cap.release()
            except:
                pass

    def _connect(self):
        try:
            cap = cv2.VideoCapture(MOTION_URL)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            except:
                pass
            return cap
        except:
            return None

    def _loop(self):
        reconnect_delay = 1.0
        last_fps_update = time.time()
        frame_times = []

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._connect()
                if not self._cap or not self._cap.isOpened():
                    GlobalState.camera_connected = False
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 5.0)
                    continue
                reconnect_delay = 1.0
                GlobalState.camera_connected = True

            start = time.time()
            ret, frame = self._cap.read()
            frame_time_ms = (time.time() - start) * 1000

            if not ret:
                try:
                    self._cap.release()
                except:
                    pass
                self._cap = None
                GlobalState.camera_connected = False
                time.sleep(0.5)
                continue

            with self._cond:
                self._frame = (time.time(), frame)
                self._frame_count += 1
                frame_times.append(frame_time_ms)

                if time.time() - last_fps_update >= 1.0:
                    if frame_times:
                        avg = sum(frame_times) / len(frame_times)
                        fps = (1000 / avg) if avg > 0 else 0
                        self._fps_history.append(fps)
                        if len(self._fps_history) > 10:
                            self._fps_history.pop(0)
                    frame_times = []
                    last_fps_update = time.time()

                GlobalState.camera_connected = True
                GlobalState.frame_counter = self._frame_count
                GlobalState.performance_metrics = GlobalState.performance_metrics or {}
                GlobalState.performance_metrics['camera_latency_ms'] = frame_time_ms
                GlobalState.performance_metrics['current_fps'] = self._fps_history[-1] if self._fps_history else 0
                GlobalState.performance_metrics['avg_fps'] = (
                    sum(self._fps_history) / len(self._fps_history)
                    if self._fps_history else 0
                )

                self._cond.notify_all()

    def wait_for_frame(self, timeout=None):
        with self._cond:
            if self._frame is not None:
                return self._frame
            self._cond.wait(timeout=timeout)
            return self._frame

    def get_latest(self):
        with self._lock:
            return self._frame
