# src/temporal_analysis.py
"""
P4: Temporal Analysis & Drowsiness Classification
===================================================

Integrates outputs from P1 (segmentation), P2 (HOG eye state), and P3 (dlib EAR).
Performs temporal analysis to detect drowsiness patterns and output a drowsiness score.

Key Features:
- Sustained eye closure detection (0.5s+ windows)
- Blink rate analysis (normal: 15-30 blinks/min, drowsy: <10 blinks/min)
- Eye Aspect Ratio (EAR) trend analysis
- Head pose assessment (nodding/yawing detection)
- Multi-signal fusion into drowsiness_score (0-1)
- State machine: alert → drowsy → critical

Input: Dict from P3 (EAR, blink_count, head_pose) + Dict from P2 (eye_state, confidence)
Output: Dict with drowsiness_score, state, confidence, debug info
"""

import numpy as np
from collections import deque
from enum import Enum
from typing import Dict, List, Tuple
import time


class DrowsinessState(Enum):
    """State machine for drowsiness detection"""
    ALERT = 0
    DROWSY = 1
    CRITICAL = 2


class TemporalAnalyzer:
    
    def __init__(
        self,
        fps: int = 30,
        ear_threshold: float = 0.22,
        blink_window_sec: float = 5.0,
        sustained_closure_sec: float = 0.5,
        critical_closure_sec: float = 1.5,
        blink_rate_normal_min: float = 15,
        blink_rate_drowsy_max: float = 8,
        head_pose_threshold: float = 15.0,
    ):
        self.fps = fps
        self.ear_threshold = ear_threshold
        self.blink_window_frames = int(blink_window_sec * fps)
        self.sustained_closure_frames = int(sustained_closure_sec * fps)
        self.critical_closure_frames = int(critical_closure_sec * fps)
        self.blink_rate_normal_min = blink_rate_normal_min
        self.blink_rate_drowsy_max = blink_rate_drowsy_max
        self.head_pose_threshold = head_pose_threshold
        self.critical_closure_sec = critical_closure_sec
        self.sustained_closure_sec = sustained_closure_sec
        
        # History buffers
        self.ear_history = deque(maxlen=300)  # 10 sec @ 30fps
        self.blink_history = deque(maxlen=300)
        self.eye_state_history = deque(maxlen=300)  # "open" or "closed"
        self.hog_confidence_history = deque(maxlen=300)
        self.head_pose_history = deque(maxlen=300)  # (yaw, pitch)
        
        # State tracking
        self.current_state = DrowsinessState.ALERT
        self.frame_count = 0
        self.last_state_change_frame = 0
        self.closure_start_frame = None  # When did the eye last close?
        self.total_blink_count = 0
        
        # Metrics for debugging
        self.last_blink_rate = 0.0
        self.last_ear_mean = 0.0
        self.last_head_pose = (0.0, 0.0)
        self.closure_duration_sec = 0.0
    
    def process(
        self,
        ear_data: Dict,
        hog_data: Dict,
    ) -> Dict:
        
        self.frame_count += 1
        
        result = {
            "success": False,
            "drowsiness_score": 0.0,
            "state": "alert",
            "state_enum": DrowsinessState.ALERT,
            "confidence": 0.0,
            "blink_rate": 0.0,
            "ear_mean": 0.0,
            "eye_closure_duration": 0.0,
            "head_pose": (0.0, 0.0),
            "signals": {},
            "metadata": {},
        }
        
        # Validation
        if not ear_data.get("success") or not hog_data.get("success"):
            return result
        
        # ===== STEP 1: Extract and buffer signals =====
        ear = ear_data.get("ear", 0.0)
        blink_detected = ear_data.get("blink_detected", False)
        blink_count = ear_data.get("blink_count", 0)
        yaw, pitch = ear_data.get("head_pose", (0.0, 0.0))
        
        eye_state = hog_data.get("eye_state", "open")
        hog_confidence = hog_data.get("confidence", 0.0)
        
        self.ear_history.append(ear)
        self.blink_history.append(1 if blink_detected else 0)
        self.eye_state_history.append(eye_state)
        self.hog_confidence_history.append(hog_confidence)
        self.head_pose_history.append((yaw, pitch))
        
        # ===== STEP 2: Calculate individual signal scores =====
        
        # Signal 1: EAR score (lower EAR = higher drowsiness)
        ear_score = self._calculate_ear_score(ear)
        
        # Signal 2: Blink rate score
        blink_rate = self._calculate_blink_rate()
        blink_score = self._calculate_blink_score(blink_rate)
        
        # Signal 3: Eye state agreement (HOG classifier)
        eye_state_score = self._calculate_eye_state_score(eye_state)
        
        # Signal 4: Sustained closure detection
        closure_duration = self._update_closure_tracking(ear)
        closure_score = self._calculate_closure_score(closure_duration)
        
        # Signal 5: Head pose analysis
        head_pose_score = self._calculate_head_pose_score(yaw, pitch)
        
        # ===== STEP 3: Fuse signals into drowsiness score =====
        signals = {
            "ear_score": ear_score,
            "blink_score": blink_score,
            "eye_state_score": eye_state_score,
            "closure_score": closure_score,
            "head_pose_score": head_pose_score,
        }
        
        # Weighted fusion (these weights can be tuned)
        drowsiness_score = (
            0.25 * ear_score +           # EAR is a strong indicator
            0.25 * blink_score +         # Blink rate is reliable over time
            0.20 * eye_state_score +     # HOG classification adds robustness
            0.20 * closure_score +       # Sustained closure is critical
            0.10 * head_pose_score       # Head pose is optional but informative
        )
        
        drowsiness_score = np.clip(drowsiness_score, 0.0, 1.0)
        
        # ===== STEP 4: State machine logic =====
        new_state = self._update_state(drowsiness_score, closure_duration)
        confidence = self._calculate_confidence(signals)
        
        # ===== STEP 5: Populate result dict =====
        result["success"] = True
        result["drowsiness_score"] = float(drowsiness_score)
        result["state"] = new_state.name.lower()
        result["state_enum"] = new_state
        result["confidence"] = float(confidence)
        result["blink_rate"] = float(blink_rate)
        result["ear_mean"] = float(np.mean(self.ear_history)) if self.ear_history else 0.0
        result["eye_closure_duration"] = float(self.closure_duration_sec)
        result["head_pose"] = (float(yaw), float(pitch))
        result["signals"] = {k: float(v) for k, v in signals.items()}
        result["metadata"] = {
            "frame_count": self.frame_count,
            "time_in_state_sec": (self.frame_count - self.last_state_change_frame) / self.fps,
            "buffer_size": len(self.ear_history),
            "blink_count_total": blink_count,
        }
        
        self.current_state = new_state
        return result
    
    # ===== Signal Calculation Functions =====
    
    def _calculate_ear_score(self, ear: float) -> float:
        """
        EAR score: 0 (fully open) to 1 (fully closed).
        Normal EAR is typically 0.3-0.4, threshold is 0.2.
        """
        if ear >= 0.3:
            return 0.0  # Fully open
        elif ear <= self.ear_threshold:
            return 1.0  # Fully closed
        else:
            # Linear interpolation between threshold and normal
            return (0.3 - ear) / (0.3 - self.ear_threshold)
    
    def _calculate_blink_rate(self) -> float:
        """
        Calculate blinks per minute over the last window.
        Normal: 15-30 blinks/min
        Drowsy: <10 blinks/min
        """
        if len(self.blink_history) < self.blink_window_frames:
            return 0.0
        
        recent_blinks = sum(self.blink_history)
        time_window_sec = len(self.blink_history) / self.fps
        blinks_per_min = (recent_blinks / time_window_sec) * 60
        
        self.last_blink_rate = blinks_per_min
        return blinks_per_min
    
    def _calculate_blink_score(self, blink_rate: float) -> float:
        """
        Blink score: 0 (normal rate) to 1 (very low rate).
        Normal: >= 15 blinks/min → score 0
        Drowsy: 8-15 blinks/min → score 0.3-0.7
        Critical: < 8 blinks/min → score 0.8-1.0
        """
        if blink_rate >= self.blink_rate_normal_min:
            return 0.0
        elif blink_rate <= self.blink_rate_drowsy_max:
            return 1.0
        else:
            # Linear interpolation
            ratio = (self.blink_rate_normal_min - blink_rate) / (
                self.blink_rate_normal_min - self.blink_rate_drowsy_max
            )
            return np.clip(ratio, 0.0, 1.0)
    
    def _calculate_eye_state_score(self, eye_state: str) -> float:
        """
        Eye state score from HOG classifier.
        "open" → 0.0, "closed" → 1.0
        """
        return 1.0 if eye_state == "closed" else 0.0
    
    def _update_closure_tracking(self, ear: float) -> float:
        """
        Track sustained eye closure. Returns duration in seconds.
        """
        is_closed = ear < self.ear_threshold
        
        if is_closed:
            if self.closure_start_frame is None:
                self.closure_start_frame = self.frame_count
            
            closure_frames = self.frame_count - self.closure_start_frame
            self.closure_duration_sec = closure_frames / self.fps
        else:
            self.closure_start_frame = None
            self.closure_duration_sec = 0.0
        
        return self.closure_duration_sec
    
    def _calculate_closure_score(self, closure_duration: float) -> float:
        """
        Closure score based on sustained closure duration.
        0-0.5s: score 0 (normal blink)
        0.5-2.0s: score 0.3-0.8 (concerning)
        >2.0s: score 1.0 (critical)
        """
        if closure_duration <= 0.3:
            return 0.0
        elif closure_duration >= self.critical_closure_sec:
            return 1.0
        elif closure_duration >= self.sustained_closure_sec:
            # Linear interpolation between sustained and critical
            ratio = (closure_duration - self.sustained_closure_sec) / (
                self.critical_closure_sec - self.sustained_closure_sec
            )
            return np.clip(0.3 + 0.7 * ratio, 0.0, 1.0)
        else:
            # Below sustained threshold but closed
            return 0.1
    
    def _calculate_head_pose_score(self, yaw: float, pitch: float) -> float:
        """
        Head pose score: detects nodding (pitch) and yawing (yaw).
        Higher magnitude = higher drowsiness indicator.
        """
        head_magnitude = np.sqrt(yaw**2 + pitch**2)
        
        if head_magnitude <= self.head_pose_threshold:
            return 0.0
        elif head_magnitude >= 40.0:
            return 1.0
        else:
            # Linear interpolation
            ratio = (head_magnitude - self.head_pose_threshold) / (40.0 - self.head_pose_threshold)
            return np.clip(ratio, 0.0, 1.0)
    
    def _update_state(self, drowsiness_score: float, closure_duration: float) -> DrowsinessState:
        """
        State machine: alert → drowsy → critical
        """
        # Check closure duration first (it's a strong indicator)
        if closure_duration >= self.critical_closure_sec:
            new_state = DrowsinessState.CRITICAL
        elif drowsiness_score >= 0.7 or closure_duration >= self.sustained_closure_sec:
            new_state = DrowsinessState.DROWSY
        else:
            new_state = DrowsinessState.ALERT
        
        # Hysteresis: don't flip-flop between states rapidly
        if new_state != self.current_state:
            # Require change to persist for at least 5 frames before transitioning
            if self.frame_count - self.last_state_change_frame >= 5:
                self.last_state_change_frame = self.frame_count
                return new_state
            else:
                return self.current_state
        
        return new_state
    
    def _calculate_confidence(self, signals: Dict) -> float:
        """
        Overall confidence in the drowsiness prediction.
        Based on signal agreement and buffer fullness.
        """
        buffer_fill_ratio = len(self.ear_history) / self.blink_window_frames
        buffer_confidence = min(buffer_fill_ratio, 1.0)
        
        # Signal variance (lower variance = higher confidence)
        signal_values = list(signals.values())
        if len(signal_values) > 1:
            signal_std = np.std(signal_values)
            signal_agreement = 1.0 - min(signal_std, 1.0)
        else:
            signal_agreement = 0.5
        
        # Combined confidence
        confidence = 0.7 * buffer_confidence + 0.3 * signal_agreement
        return np.clip(confidence, 0.0, 1.0)
    
    def reset(self):
        """Reset all buffers and state. Call if starting new session."""
        self.ear_history.clear()
        self.blink_history.clear()
        self.eye_state_history.clear()
        self.hog_confidence_history.clear()
        self.head_pose_history.clear()
        self.current_state = DrowsinessState.ALERT
        self.frame_count = 0
        self.closure_start_frame = None
        self.closure_duration_sec = 0.0
        self.total_blink_count = 0


# ===== Helper Function for Integration =====

def create_temporal_analyzer(
    fps: int = 30,
    config: Dict = None,
) -> TemporalAnalyzer:
    """Factory function to create TemporalAnalyzer with optional config overrides"""
    analyzer = TemporalAnalyzer(fps=fps)
    
    if config:
        for key, value in config.items():
            if hasattr(analyzer, key):
                setattr(analyzer, key, value)
    
    return analyzer