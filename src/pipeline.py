"""
P5: Integration - Real-time webcam drowsiness detection pipeline
P1 (Segmentation) -> P2 (HOG) -> P3 (dlib EAR, optional) -> P4 (Temporal Analysis)

When dlib/P3 is unavailable the HOG output from P2 is used to synthesize
EAR values and blink counts so P4 still works correctly.

Controls:
    q / ESC  - Quit
    r        - Reset temporal analyzer
    s        - Save screenshot
"""

import cv2
import numpy as np
import time
import threading
from collections import deque
from datetime import datetime

from segmentation import FaceSegmentor
from eye_hog import EyeStateClassifier
from analysis import TemporalAnalyzer

try:
    import winsound
    _WINSOUND = True
except ImportError:
    _WINSOUND = False

_DLIB_OK = False
try:
    from dlib_ear import EARDetector
    _DLIB_OK = True
except Exception as _e:
    print(f"[P5] P3/dlib unavailable ({_e}). Using HOG-based blink synthesis.")


# ── colour palette ────────────────────────────────────────────────────────────
_STATE_COLOR = {
    "alert":    (30, 200, 30),
    "drowsy":   (0, 200, 255),
    "critical": (0, 0, 255),
}
_PANEL_BG   = (20, 20, 20)
_TEXT_COLOR = (220, 220, 220)


def _draw_hbar(img, x, y, w, h, fill_ratio, bar_color, label=""):
    fill_ratio = float(np.clip(fill_ratio, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), (55, 55, 55), -1)
    fill_px = int(w * fill_ratio)
    if fill_px > 0:
        cv2.rectangle(img, (x, y), (x + fill_px, y + h), bar_color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (110, 110, 110), 1)
    if label:
        cv2.putText(img, label, (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1)


# ── HOG-based blink + EAR synthesizer (used when P3 is unavailable) ──────────

class _HOGBlinkTracker:
    """
    Converts HOG eye-state ("open"/"closed") into a P4-compatible ear_data dict.
    - Synthesizes EAR: closed → 0.15 (below 0.2 threshold), open → 0.32
    - Detects blinks by tracking open→closed transitions
    """

    _EAR_CLOSED = 0.15
    _EAR_OPEN   = 0.32

    def __init__(self):
        self._prev_state  = "open"
        self._blink_count = 0

    def update(self, hog_result: dict) -> dict:
        if not hog_result.get("success"):
            return {"success": False}

        state = hog_result.get("eye_state", "open")
        ear   = self._EAR_CLOSED if state == "closed" else self._EAR_OPEN

        # count blink on open→closed edge
        blink_detected = False
        if self._prev_state == "open" and state == "closed":
            self._blink_count += 1
            blink_detected = True
        self._prev_state = state

        return {
            "success":       True,
            "ear":           ear,
            "blink_detected": blink_detected,
            "blink_count":   self._blink_count,
            "head_pose":     (0.0, 0.0),
            "left_eye_pts":  None,
            "right_eye_pts": None,
        }


# ── main pipeline ─────────────────────────────────────────────────────────────

class DrowsinessDetectionPipeline:

    # alert frequencies & cooldowns per state
    _ALERT_CFG = {
        "drowsy":   {"freq": 800,  "duration": 250, "cooldown": 3.5},
        "critical": {"freq": 1300, "duration": 500, "cooldown": 1.2},
    }

    def __init__(self, fps: int = 30):
        self.target_fps  = fps
        self.frame_count = 0
        self.frame_times = deque(maxlen=30)
        self._last_alert = {"drowsy": 0.0, "critical": 0.0}

        print("[P5] Loading P1 (Segmentation)…")
        self.segmentor = FaceSegmentor()

        print("[P5] Loading P2 (HOG classifier)…")
        self.hog = EyeStateClassifier()

        self.ear_detector = None
        self._hog_blink   = None
        if _DLIB_OK:
            print("[P5] Loading P3 (dlib EAR)…")
            try:
                self.ear_detector = EARDetector()
                print("[P5] P3 loaded — using real EAR.")
            except Exception as e:
                print(f"[P5] P3 init failed: {e}. Falling back to HOG synthesis.")
        if self.ear_detector is None:
            self._hog_blink = _HOGBlinkTracker()
            print("[P5] HOG blink synthesizer active.")

        print("[P5] Loading P4 (Temporal Analyzer)…")
        self.analyzer = TemporalAnalyzer(fps=fps)
        print("[P5] Pipeline ready.")

    # ── alert ─────────────────────────────────────────────────────────────────

    def _trigger_alert(self, state: str):
        cfg = self._ALERT_CFG.get(state)
        if cfg is None:
            return
        now = time.time()
        if now - self._last_alert[state] < cfg["cooldown"]:
            return
        self._last_alert[state] = now
        if _WINSOUND:
            freq, dur = cfg["freq"], cfg["duration"]
            threading.Thread(
                target=lambda: winsound.Beep(freq, dur), daemon=True
            ).start()

    # ── per-frame processing ──────────────────────────────────────────────────

    def process_frame(self, frame):
        seg = self.segmentor.process(frame)
        if not seg["success"]:
            return None, seg

        face_crop = seg["face_crop"]
        bbox      = seg["bbox"]

        hog_result = self.hog.process(face_crop)

        if self.ear_detector is not None:
            ear_result = self.ear_detector.process(face_crop, bbox)
            if not ear_result.get("success"):
                # real P3 failed on this frame — synthesize from HOG instead
                ear_result = self._hog_blink.update(hog_result) if self._hog_blink else {"success": False}
        else:
            ear_result = self._hog_blink.update(hog_result)

        analysis = self.analyzer.process(ear_result, hog_result)
        return analysis, seg

    # ── visualisation ─────────────────────────────────────────────────────────

    def draw_overlay(self, frame, analysis, seg):
        h, w = frame.shape[:2]
        out  = frame.copy()

        success = bool(analysis and analysis.get("success"))
        state   = analysis.get("state", "alert") if success else "alert"
        color   = _STATE_COLOR.get(state, _STATE_COLOR["alert"])

        # flashing red border for critical
        if state == "critical":
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 14)
        # orange border for drowsy
        elif state == "drowsy":
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 180, 255), 6)

        # face bbox
        if seg.get("bbox"):
            x, y, bw, bh = seg["bbox"]
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(out, "FACE", (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # semi-transparent top panel
        panel_h = 108
        panel   = out.copy()
        cv2.rectangle(panel, (0, 0), (w, panel_h), _PANEL_BG, -1)
        cv2.addWeighted(panel, 0.78, out, 0.22, 0, out)

        if success:
            score    = analysis["drowsiness_score"]
            ear_val  = analysis["ear_mean"]
            blink_r  = analysis["blink_rate"]
            closure  = analysis["eye_closure_duration"]
            conf     = analysis["confidence"]
            blinks_t = analysis.get("metadata", {}).get("blink_count_total", 0)

            # state badge
            cv2.rectangle(out, (8, 6), (178, 80), color, -1)
            cv2.putText(out, state.upper(), (14, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(out, f"score {score:.2f}", (14, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

            # metrics
            mx = 192
            cv2.putText(out, f"EAR      {ear_val:.3f}", (mx, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _TEXT_COLOR, 1)
            cv2.putText(out, f"Blinks   {blink_r:.0f}/min",  (mx, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _TEXT_COLOR, 1)
            cv2.putText(out, f"Closure  {closure:.2f}s",     (mx, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _TEXT_COLOR, 1)
            cv2.putText(out, f"Conf     {conf:.2f}",         (mx, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 140, 140), 1)

            # drowsiness gauge
            _draw_hbar(out, 8, panel_h + 10, 260, 20, score,
                       color, label="Drowsiness")

            # EAR gauge (0–0.4 → 0–1)
            ear_color = (30, 200, 30) if ear_val >= 0.2 else (0, 100, 255)
            _draw_hbar(out, 8, panel_h + 48, 260, 16, ear_val / 0.4,
                       ear_color, label="EAR")

            # total blink count bottom-left
            cv2.putText(out, f"Total blinks: {blinks_t}", (8, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

        else:
            cv2.putText(out, "NO FACE DETECTED", (10, 58),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (100, 100, 100), 2)

        # FPS
        if len(self.frame_times) > 1:
            dt  = self.frame_times[-1] - self.frame_times[0]
            fps = (len(self.frame_times) - 1) / dt if dt > 0 else 0.0
            cv2.putText(out, f"FPS {fps:.1f}", (w - 115, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 200, 30), 1)

        cv2.putText(out, f"Frame {self.frame_count}", (w - 130, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)

        return out

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self, camera_index: int = 0):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("[P5] ERROR: Cannot open camera.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        print("[P5] Running — q/ESC quit | r reset | s screenshot")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[P5] Camera read failed.")
                break

            self.frame_count += 1
            self.frame_times.append(time.time())

            analysis, seg = self.process_frame(frame)

            if analysis and analysis.get("success"):
                state = analysis.get("state", "alert")
                if state in ("drowsy", "critical"):
                    self._trigger_alert(state)

            out = self.draw_overlay(frame, analysis, seg)
            cv2.imshow("Driver Drowsiness Detection [P5]", out)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                self.analyzer.reset()
                if self._hog_blink:
                    self._hog_blink = _HOGBlinkTracker()
                print("[P5] Analyzer reset.")
            elif key == ord('s'):
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"screenshot_{ts}.png"
                cv2.imwrite(name, out)
                print(f"[P5] Screenshot saved: {name}")

        cap.release()
        cv2.destroyAllWindows()
        print("[P5] Finished.")
