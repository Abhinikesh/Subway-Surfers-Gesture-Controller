import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from dataclasses import dataclass
from typing import Optional, List
import math

@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float

@dataclass
class PoseResult:
    landmarks: List[Landmark]
    image: np.ndarray

class PoseDetector:
    NOSE           = 0
    LEFT_SHOULDER  = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW     = 13
    RIGHT_ELBOW    = 14
    LEFT_WRIST     = 15
    RIGHT_WRIST    = 16
    LEFT_HIP       = 23
    RIGHT_HIP      = 24
    LEFT_KNEE      = 25
    RIGHT_KNEE     = 26
    LEFT_ANKLE     = 27
    RIGHT_ANKLE    = 28

    def __init__(self, model_path: str = "pose_landmarker.task"):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray) -> Optional[PoseResult]:
        # Resize to 320x240 for speed before sending to MediaPipe
        small = cv2.resize(frame, (320, 240))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        landmarks = []
        for lm in result.pose_landmarks[0]:
            landmarks.append(Landmark(
                x=lm.x, y=lm.y, z=lm.z,
                visibility=lm.visibility or 0.0
            ))

        return PoseResult(landmarks=landmarks, image=frame)

    def draw_skeleton(self, frame: np.ndarray, result: PoseResult) -> np.ndarray:
        h, w = frame.shape[:2]
        lms  = result.landmarks

        CONNECTIONS = [
            (11,12),(11,13),(13,15),(12,14),(14,16),
            (11,23),(12,24),(23,24),(23,25),(24,26),
            (25,27),(26,28)
        ]

        overlay = frame.copy()
        for a, b in CONNECTIONS:
            if lms[a].visibility > 0.3 and lms[b].visibility > 0.3:
                p1 = (int(lms[a].x * w), int(lms[a].y * h))
                p2 = (int(lms[b].x * w), int(lms[b].y * h))
                cv2.line(overlay, p1, p2, (0,245,255), 6)
                cv2.line(frame,   p1, p2, (0,245,255), 2)

        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        for lm in lms:
            if lm.visibility > 0.3:
                px, py = int(lm.x*w), int(lm.y*h)
                cv2.circle(frame, (px,py), 5, (179,0,255), -1)
                cv2.circle(frame, (px,py), 7, (255,255,255), 1)

        return frame

    def get_landmark(self, result: PoseResult, index: int) -> Optional[Landmark]:
        if index < len(result.landmarks):
            lm = result.landmarks[index]
            if lm.visibility > 0.3:
                return lm
        return None

    @staticmethod
    def angle_between(a: Landmark, b: Landmark, c: Landmark) -> float:
        v1 = np.array([a.x-b.x, a.y-b.y])
        v2 = np.array([c.x-b.x, c.y-b.y])
        cos_a = np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
        return math.degrees(math.acos(np.clip(cos_a,-1,1)))

    @staticmethod
    def shoulder_tilt_angle(l: Landmark, r: Landmark) -> float:
        return math.degrees(math.atan2(r.y-l.y, r.x-l.x))
