import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetector:

    def __init__(self):
        base_options = python.BaseOptions(
            model_asset_path="hand_landmarker.task"
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        results = self.detector.detect(mp_image)

        if results.hand_landmarks:
            return results.hand_landmarks[0]

        return None