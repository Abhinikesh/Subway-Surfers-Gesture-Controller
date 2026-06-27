import time
from collections import deque
from typing import Optional
from modules.pose_detector import PoseDetector, PoseResult, Landmark
import config

class GestureEngine:
    def __init__(self, detector: PoseDetector):
        self.detector = detector
        self.buffer = deque(maxlen=config.GESTURE_BUFFER_SIZE)
        self.last_gesture_time = {}
        self.current_gesture = "IDLE"

    def _get_lm(self, result: PoseResult, index: int) -> Optional[Landmark]:
        return self.detector.get_landmark(result, index)

    def _detect_raw(self, result: PoseResult) -> str:
        d = self.detector

        l_wrist    = self._get_lm(result, d.LEFT_WRIST)
        r_wrist    = self._get_lm(result, d.RIGHT_WRIST)
        l_shoulder = self._get_lm(result, d.LEFT_SHOULDER)
        r_shoulder = self._get_lm(result, d.RIGHT_SHOULDER)
        l_hip      = self._get_lm(result, d.LEFT_HIP)
        r_hip      = self._get_lm(result, d.RIGHT_HIP)
        l_knee     = self._get_lm(result, d.LEFT_KNEE)
        r_knee     = self._get_lm(result, d.RIGHT_KNEE)

        if not all([l_wrist, r_wrist, l_shoulder, r_shoulder]):
            return "IDLE"

        shoulder_w = abs(r_shoulder.x - l_shoulder.x)
        shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
        body_cx    = (l_shoulder.x + r_shoulder.x) / 2

        l_wrist_rel_y = l_wrist.y - l_shoulder.y
        r_wrist_rel_y = r_wrist.y - r_shoulder.y

        if (l_wrist_rel_y < -config.JUMP_THRESHOLD and
            r_wrist_rel_y < -config.JUMP_THRESHOLD):
            return "JUMP"

        wrist_dist_x     = abs(l_wrist.x - r_wrist.x)
        wrist_dist_y     = abs(l_wrist.y - r_wrist.y)
        both_low         = (l_wrist.y > shoulder_y + 0.08 and
                            r_wrist.y > shoulder_y + 0.08)
        if (wrist_dist_x < shoulder_w * 0.55 and
            wrist_dist_y < 0.15 and
            both_low):
            return "SLIDE"

        l_out = l_shoulder.x - l_wrist.x
        r_out = r_wrist.x - r_shoulder.x

        threshold = shoulder_w * 0.5

        l_mid = shoulder_y - 0.2 < l_wrist.y < shoulder_y + 0.35
        r_mid = shoulder_y - 0.2 < r_wrist.y < shoulder_y + 0.35

        r_relaxed = r_out < threshold * 0.5
        l_relaxed = l_out < threshold * 0.5

        if (l_out > threshold and l_mid and r_relaxed):
            return "LEFT"

        if (r_out > threshold and r_mid and l_relaxed):
            return "RIGHT"

        chest_top = shoulder_y - 0.05
        chest_bot = shoulder_y + 0.25
        l_in_chest = chest_top < l_wrist.y < chest_bot
        r_in_chest = chest_top < r_wrist.y < chest_bot
        if (l_wrist.x > body_cx + 0.06 and
            r_wrist.x < body_cx - 0.06 and
            l_in_chest and r_in_chest):
            return "SHIELD"

        return "IDLE"

    def update(self, result: PoseResult) -> str:
        raw = self._detect_raw(result)
        self.buffer.append(raw)

        if len(self.buffer) < config.GESTURE_BUFFER_SIZE:
            return "IDLE"

        counts = {}
        for g in self.buffer:
            counts[g] = counts.get(g, 0) + 1

        best = max(counts, key=counts.get)

        if counts[best] < config.GESTURE_CONFIRM_COUNT:
            self.current_gesture = "IDLE"
            return "IDLE"

        if best == "IDLE":
            self.current_gesture = "IDLE"
            return "IDLE"

        now  = time.time()
        last = self.last_gesture_time.get(best, 0)
        if now - last < config.COOLDOWN_SECONDS:
            return "IDLE"

        self.last_gesture_time[best] = now
        self.current_gesture = best
        return best

    def confidence(self) -> float:
        if not self.buffer or self.current_gesture == "IDLE":
            return 0.0
        count = sum(1 for x in self.buffer if x == self.current_gesture)
        return count / len(self.buffer)

class GestureDetector:
    pass
class GestureSmoother:
    pass
