import cv2
import threading
import time
from typing import Optional, Tuple
from shared_config import MOTION_URL, ENVIRONMENT, GlobalState

# Single place to store the latest frame + metadata with a Condition to notify consumer(s).
class CameraManager:
    def __init__(self, source: str, buffer_size: int = 1):
        self.source = source
        self.buffer_size = buffer_size
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame = None  # stores (timestamp, frame)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap = None

        # basic stats
        self._frame_count = 0
        self._fps_history = []

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        with self._lock:
            if self._cap:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

    def _open_capture(self):
        try:
            cap = cv2.VideoCapture(self.source)
            # suggest smallest possible internal buffer to reduce lag
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            except Exception:
                pass
            return cap
        except Exception:
            return None

    def _capture_loop(self):
        reconnect_delay = 1.0
        last_fps_update = time.time()
        frame_times = []

        while self._running:
            try:
                if self._cap is None or not self._cap.isOpened():
                    self._cap = self._open_capture()
                    if self._cap is None or not self._cap.isOpened():
                        GlobalState.camera_connected = False
                        time.sleep(reconnect_delay)
                        # exponential backoff up to 5s to avoid busy loop on persistent failures
                        reconnect_delay = min(reconnect_delay * 2, 5.0)
                        continue
                    reconnect_delay = 1.0
                    GlobalState.camera_connected = True

                start = time.time()
                ret, frame = self._cap.read()
                frame_time_ms = (time.time() - start) * 1000.0

                if not ret or frame is None:
                    # connection glitch, release and try to reconnect quickly
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                    self._cap = None
                    GlobalState.camera_connected = False
                    time.sleep(0.5)
                    continue

                # update frame & notify any waiting inference consumer
                with self._cond:
                    self._frame = (time.time(), frame)
                    self._frame_count += 1
                    # maintain fps estimate
                    frame_times.append(frame_time_ms)
                    if time.time() - last_fps_update >= 1.0:
                        if frame_times:
                            avg_frame_time = sum(frame_times) / len(frame_times)
                            current_fps = (1000.0 / avg_frame_time) if avg_frame_time > 0 else 0
                            self._fps_history.append(current_fps)
                            if len(self._fps_history) > 10:
                                self._fps_history.pop(0)
                        frame_times = []
                        last_fps_update = time.time()

                    # Update GlobalState camera metrics (non-blocking)
                    GlobalState.frame_counter = self._frame_count
                    GlobalState.performance_metrics = GlobalState.performance_metrics or {}
                    GlobalState.performance_metrics['camera_latency_ms'] = frame_time_ms
                    GlobalState.performance_metrics['current_fps'] = self._fps_history[-1] if self._fps_history else 0
                    GlobalState.performance_metrics['avg_fps'] = sum(self._fps_history) / len(self._fps_history) if self._fps_history else 0

                    # wake inference thread(s)
                    self._cond.notify_all()

            except Exception as e:
                # keep running; try to reconnect
                print(f"Camera capture loop error: {e}")
                try:
                    if self._cap:
                        self._cap.release()
                except Exception:
                    pass
                self._cap = None
                GlobalState.camera_connected = False
                time.sleep(1.0)

    def get_latest(self) -> Optional[Tuple[float, any]]:
        """Return the most recent (timestamp, frame) or None"""
        with self._lock:
            return self._frame

    def wait_for_frame(self, timeout: Optional[float] = None) -> Optional[Tuple[float, any]]:
        """Wait until a new frame is available (relative to the moment caller last read),
           returns the latest frame (timestamp, frame) or None when timeout."""
        with self._cond:
            # If there's already a frame, return it immediately
            if self._frame is not None:
                return self._frame
            # Otherwise wait (this avoids busy-wait sleeps)
            self._cond.wait(timeout=timeout)
            return self._frame

# Convenience module-level manager
_camera_manager: Optional[CameraManager] = None

def get_camera_manager() -> CameraManager:
    global _camera_manager
    if _camera_manager is None:
        _camera_manager = CameraManager(MOTION_URL, buffer_size=1)
    return _camera_manager

if __name__ == "__main__":
    if ENVIRONMENT != 'raspberry_pi':
        print("This script is intended to run on Raspberry Pi directly.")
        print("Use server.py on WSL/Desktop to connect remotely.")
        exit(1)

    print("Starting camera manager (local Pi camera)...")
    cam = get_camera_manager()
    cam.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        cam.stop()
        print("👋 Camera stopped")
