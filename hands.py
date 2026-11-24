"""Hand tracking and features, shared by collect.py, train.py and app.py.

The one function to understand is `to_features`: it turns MediaPipe's 21 hand
landmarks into the 63 numbers the model learns from.
"""
import os
import urllib.request

import cv2
import numpy as np

# ---------------------------------------------------------------- features
FINGERS = {  # landmark indices for each finger (0 is the wrist)
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}
LANDMARK_COLS = [f"{axis}{i}" for i in range(21) for axis in "xyz"]


def to_features(world, handedness):
    """21 world landmarks (x, y, z in metres) -> 63 numbers.

    1. Move the wrist to the origin, so where your hand is in the frame doesn't matter.
    2. Mirror left hands, so a left-handed signer looks like a right-handed one.
    3. Scale so wrist -> middle knuckle = 1, so hand size and distance don't matter.
    Rotation is kept on purpose: in sign language, palm direction can change the meaning.
    """
    pts = np.asarray(world, dtype=np.float32).reshape(21, 3).copy()
    pts -= pts[0]
    if handedness == "Left":
        pts[:, 0] *= -1
    scale = float(np.linalg.norm(pts[9])) or 1.0
    return (pts / scale).flatten()


# ---------------------------------------------------------------- tracking
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")
MODEL_URLS = [  # official location first, older one as a fallback
    "https://huggingface.co/litert-community/MediaPipe-HandLandmarker-Full/resolve/main/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
]


def ensure_model(path=MODEL_PATH):
    """Download MediaPipe's hand model (about 8 MB) the first time."""
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for url in MODEL_URLS:
        try:
            print(f"Downloading the hand model from {url} ...")
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)
            return path
        except Exception as e:
            print(f"  failed: {e}")
    raise SystemExit(f"Couldn't download the hand model. Download it from {MODEL_URLS[0]} and save it as {path}")


class Hand:
    def __init__(self, world, image, handedness):
        self.world = world            # 21 x (x, y, z) in metres, for features
        self.image = image            # 21 x (x, y) in 0-1 image coordinates, for drawing
        self.handedness = handedness  # "Left" or "Right"

    def features(self):
        return to_features(self.world, self.handedness)

    def in_frame(self, margin=0.02):
        return all(margin < x < 1 - margin and margin < y < 1 - margin for x, y in self.image)


class HandTracker:
    """MediaPipe HandLandmarker. Feed it normal (not mirrored) BGR frames from OpenCV."""

    def __init__(self, video=True, max_hands=1):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        self.mp, self.video = mp, video
        self.landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_hands=max_hands))

    def detect(self, frame_bgr, timestamp_ms=0):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        r = self.landmarker.detect_for_video(image, int(timestamp_ms)) if self.video else self.landmarker.detect(image)
        hands = []
        for world, img, hd in zip(r.hand_world_landmarks, r.hand_landmarks, r.handedness):
            hands.append(Hand([(p.x, p.y, p.z) for p in world], [(p.x, p.y) for p in img], hd[0].category_name))
        return hands

    def close(self):
        self.landmarker.close()


# ---------------------------------------------------------------- drawing
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def finger_of(i):
    return next((name for name, ids in FINGERS.items() if i in ids), "palm")


def draw_hand(view, hand, colors=None, default=(255, 255, 255)):
    """Draw on a mirrored view. `colors` maps finger name -> BGR colour."""
    h, w = view.shape[:2]
    pts = [(int((1 - x) * w), int(y * h)) for x, y in hand.image]
    for a, b in BONES:
        c = (colors or {}).get(finger_of(b), default)
        cv2.line(view, pts[a], pts[b], c, 4, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(view, p, 5, (colors or {}).get(finger_of(i), default), -1, cv2.LINE_AA)


def put_text(view, text, org, scale=0.8, color=(255, 255, 255), thick=2, box=True):
    """Text on a dark box so it stays readable over any background."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    if box:
        pad = max(4, int(6 * scale))
        overlay = view.copy()
        cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, view, 0.45, 0, dst=view)
    cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def text_width(text, scale=0.8, thick=2):
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]
