import csv
import math
import os
from collections import deque, Counter

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import RandomForestClassifier

from hand_detector import HandDetector

DEBUG = False  # True prints finger states so you can tune the ASL thresholds

# ---------------------------------------------------------------- Tamil setup
TAMIL = [
    # 12 vowels + aytham
    ("a", "அ"), ("aa", "ஆ"), ("i", "இ"), ("ii", "ஈ"), ("u", "உ"), ("uu", "ஊ"),
    ("e", "எ"), ("ee", "ஏ"), ("ai", "ஐ"), ("o", "ஒ"), ("oo", "ஓ"), ("au", "ஔ"),
    ("aytham", "ஃ"),
    # 18 consonants
    ("k", "க்"), ("ng", "ங்"), ("c", "ச்"), ("ny", "ஞ்"), ("tt", "ட்"),
    ("nn", "ண்"), ("th", "த்"), ("n", "ந்"), ("p", "ப்"), ("m", "ம்"),
    ("y", "ய்"), ("r", "ர்"), ("l", "ல்"), ("v", "வ்"), ("zh", "ழ்"),
    ("ll", "ள்"), ("rr", "ற்"), ("nnn", "ன்"),
]
TAMIL_MAP = dict(TAMIL)
DATA_FILE = "tamil_data.csv"
MODEL_FILE = "tamil_model.pkl"
CONF = 0.5  # minimum probability to accept a Tamil prediction

# OpenCV can't draw Tamil script, so text goes through PIL with a Tamil font
FONT_PATHS = [
    "Nirmala.ttf",
    "C:/Windows/Fonts/Nirmala.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf",
    "/usr/share/fonts/truetype/lohit-tamil/Lohit-Tamil.ttf",
]


def load_font(size):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return None


FONT = load_font(46)


def draw(frame, text, y, color=(0, 255, 0)):
    """Draw text; known Tamil sign names are shown as 'அ (a)'."""
    if text in TAMIL_MAP and FONT is not None:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(img).text((20, y - 46), f"{TAMIL_MAP[text]}  ({text})",
                                 font=FONT, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return frame


def extract_features(hand):
    """21 landmarks -> 63 numbers (wrist at origin, scaled by hand size)."""
    pts = np.array([[p.x, p.y, getattr(p, "z", 0.0)] for p in hand],
                   dtype=np.float32)
    pts -= pts[0]
    pts /= max(np.linalg.norm(pts[9, :2]), 1e-6)
    return pts.flatten()


def mirror(X):
    """Flip x so right-hand samples also teach the model the left hand."""
    M = X.reshape(-1, 21, 3).copy()
    M[:, :, 0] *= -1
    return M.reshape(len(X), -1)


def load_counts():
    counts = {r: 0 for r, _ in TAMIL}
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            for row in csv.reader(f):
                if row:
                    counts[row[0]] = counts.get(row[0], 0) + 1
    return counts


def train_tamil():
    if not os.path.exists(DATA_FILE):
        print("No Tamil data yet.")
        return None
    labels, rows = [], []
    with open(DATA_FILE) as f:
        for r in csv.reader(f):
            if r:
                labels.append(r[0])
                rows.append(r[1:])
    if len(set(labels)) < 2:
        print("Record at least 2 different signs first.")
        return None
    X = np.array(rows, dtype=np.float32)
    y = np.array(labels)
    X = np.vstack([X, mirror(X)])
    y = np.concatenate([y, y])
    model = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    print(f"Trained on {len(labels)} samples, {len(set(labels))} signs.")
    return model


# ------------------------------------------------------------- ASL (rules)
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def thumb_out(hand):
    return dist(hand[4], hand[17]) > dist(hand[3], hand[17])


def finger_state(hand, mcp, tip):
    """2 = straight, 1 = hooked, 0 = curled (works at any rotation)."""
    r = dist(hand[tip], hand[0]) / dist(hand[mcp], hand[0])
    if r > 1.5:
        return 2
    if r > 1.15:
        return 1
    return 0


def orientation(hand):
    dx = hand[9].x - hand[0].x
    dy = hand[9].y - hand[0].y
    if abs(dx) > abs(dy):
        return "side"
    return "up" if dy < 0 else "down"


def thumb_position(hand):
    """Thumb tip along the knuckle line: 0 = index side, 1 = pinky side."""
    ax, ay = hand[5].x, hand[5].y
    vx, vy = hand[17].x - ax, hand[17].y - ay
    return ((hand[4].x - ax) * vx + (hand[4].y - ay) * vy) / (vx * vx + vy * vy)


def recognize_asl(hand):
    size = dist(hand[0], hand[9])

    def d(a, b):  # distance between landmarks, relative to hand size
        return dist(hand[a], hand[b]) / size

    idx = finger_state(hand, 5, 8)
    mid = finger_state(hand, 9, 12)
    rng = finger_state(hand, 13, 16)
    pnk = finger_state(hand, 17, 20)
    thumb = thumb_out(hand)
    pose = orientation(hand)

    if DEBUG:
        print((idx, mid, rng, pnk), thumb, pose, round(thumb_position(hand), 2))

    # ---- Pointing sideways / down: G, H, P, Q ----
    if pose == "side":
        if idx == 2 and mid == 2 and rng < 2 and pnk < 2:
            return "H"
        if idx == 2 and mid < 2 and rng < 2 and pnk < 2:
            return "G"
    if pose == "down":
        if idx == 2 and mid == 2 and rng < 2 and pnk < 2:
            return "P"
        if idx >= 1 and mid < 2 and rng < 2 and pnk < 2 and thumb:
            return "Q"

    # ---- All four straight: B (thumb tucked) ----
    if idx == 2 and mid == 2 and rng == 2 and pnk == 2:
        return "5 - OPEN HAND" if thumb else "B"

    # ---- F: thumb touches index, other three straight ----
    if mid == 2 and rng == 2 and pnk == 2 and d(4, 8) < 0.3:
        return "F"

    # ---- W: three straight, pinky folded ----
    if idx == 2 and mid == 2 and rng == 2 and pnk < 2:
        return "W"

    # ---- Index + middle straight: R, U, V, K ----
    if idx == 2 and mid == 2 and rng < 2 and pnk < 2:
        gap = d(8, 12)
        if gap < 0.15:          # crossed fingers
            return "R"
        if gap < 0.3:           # together
            return "U"
        return "K" if d(4, 10) < d(4, 14) else "V"

    # ---- Index + pinky straight: I LOVE YOU / ROCK ----
    if idx == 2 and pnk == 2 and mid < 2 and rng < 2:
        return "I LOVE YOU" if thumb else "ROCK"

    # ---- Single index straight: D, L ----
    if idx == 2 and mid < 2 and rng < 2 and pnk < 2:
        if d(4, 12) < 0.35:     # thumb touching middle fingertip
            return "D"
        if thumb:
            return "L"
        return "1 - INDEX"

    # ---- Pinky straight: I, Y ----
    if pnk == 2 and idx < 2 and mid < 2 and rng < 2:
        return "Y" if thumb else "I"

    # ---- Hooked index only: X ----
    if idx == 1 and mid == 0 and rng == 0 and pnk == 0:
        return "X"

    # ---- Curved hand: C, O ----
    if idx == 1 and mid == 1 and rng == 1 and pnk == 1:
        return "O" if d(4, 8) < 0.3 else "C"

    # ---- Closed hand: A, E, S, T, N, M ----
    if idx < 2 and mid < 2 and rng < 2 and pnk < 2:
        if thumb and d(4, 10) > 0.55:
            return "A"
        if d(4, 8) < 0.35 and d(4, 12) < 0.35 and d(4, 16) < 0.35:
            return "E"
        t = thumb_position(hand)
        if t < 0.3:
            return "T"
        if t < 0.5:
            return "S"
        if t < 0.75:
            return "N"
        return "M"

    return "UNKNOWN"


# --------------------------------------------------------------------- main
def main():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # drop CAP_DSHOW on Linux/macOS
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    detector = HandDetector()
    history = deque(maxlen=7)  # smoothing window

    mode = "ASL"
    tamil_model = joblib.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else None
    counts = load_counts()
    current, recording = 0, False

    data_file = open(DATA_FILE, "a", newline="")
    writer = csv.writer(data_file)

    while True:
        success, frame = camera.read()
        if not success:
            print("ERROR: Could not read camera frame.")
            break

        frame = cv2.flip(frame, 1)
        hand = detector.detect(frame)
        roman = TAMIL[current][0]

        if hand:
            if mode == "ASL":
                history.append(recognize_asl(hand))
                text = Counter(history).most_common(1)[0][0]
            else:
                feats = extract_features(hand)
                if recording:
                    writer.writerow([roman, *feats])
                    counts[roman] += 1
                if tamil_model is not None:
                    probs = tamil_model.predict_proba(feats.reshape(1, -1))[0]
                    i = probs.argmax()
                    history.append(tamil_model.classes_[i]
                                   if probs[i] >= CONF else "?")
                    text = Counter(history).most_common(1)[0][0]
                else:
                    text = "NO MODEL YET"
        else:
            history.clear()
            text = "SHOW YOUR HAND"

        cv2.putText(frame, f"MODE: {mode}   (M = switch, Q = quit)", (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        frame = draw(frame, text, 85)

        if mode == "TAMIL":
            status = "REC" if recording else "PAUSED"
            frame = draw(frame, roman, 150, (0, 165, 255))
            cv2.putText(frame,
                        f"{status} n={counts[roman]}  [ ] sign  SPACE rec  T train",
                        (20, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255) if recording else (255, 255, 255), 2)

        cv2.imshow("Signify AI - Sign Language", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            mode = "TAMIL" if mode == "ASL" else "ASL"
            history.clear()
            recording = False
        elif mode == "TAMIL":
            if key == ord("]"):
                current, recording = (current + 1) % len(TAMIL), False
            elif key == ord("["):
                current, recording = (current - 1) % len(TAMIL), False
            elif key == 32:
                recording = not recording
            elif key == ord("t"):
                recording = False
                data_file.flush()
                tamil_model = train_tamil() or tamil_model
                history.clear()

    data_file.close()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()