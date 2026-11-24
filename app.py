"""Live sign recognition and practice.

    python app.py

Keys:  p  switch between Free mode and Practice mode
       n  skip to the next sign (Practice)
       q  quit

Free mode shows what the model thinks you're signing.
Practice mode gives you a sign, colours each finger, and points out the finger
that's furthest from how the sign looked in your training data.
"""
import collections
import os
import random
import time

import cv2
import joblib
import numpy as np

from hands import FINGERS, HandTracker, draw_hand, put_text, text_width

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "sign_model.joblib")
SMOOTH = 8          # average the model's output over this many frames to stop flicker
CONFIDENT = 0.70    # how sure the model must be before we accept a sign
HOLD = 12           # frames the right sign must be held in Practice mode
FINGER_OFF = 0.35   # palm units a finger can drift from the usual shape before we flag it

GREEN, RED, WHITE, ORANGE, GREY = (110, 220, 60), (90, 70, 255), (255, 255, 255), (0, 170, 255), (160, 160, 160)


def worst_finger(features, mean):
    """The finger furthest from this sign's average shape in the training data."""
    a, b = features.reshape(21, 3), mean.reshape(21, 3)
    errors = {name: float(np.linalg.norm(a[ids] - b[ids], axis=1).mean()) for name, ids in FINGERS.items()}
    name = max(errors, key=errors.get)
    return name, errors[name]


def practice_feedback(label, conf, target, finger, err):
    """What to tell the learner this frame. Returns (correct, message)."""
    part = "thumb" if finger == "thumb" else f"{finger} finger"
    if label == target and conf >= CONFIDENT:
        return True, "Hold it..."
    if conf >= CONFIDENT:
        return False, f"That looks like {label}. Check your {part}."
    if err > FINGER_OFF:
        return False, f"Check your {part}"
    return False, "Almost there"


def bar(view, x, y, w, h, frac, color):
    cv2.rectangle(view, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(view, (x, y), (x + int(w * max(0.0, min(1.0, frac))), y + h), color, -1)


def main():
    if not os.path.exists(MODEL):
        raise SystemExit("No trained model yet. Run collect.py, then train.py.")
    bundle = joblib.load(MODEL)
    model, labels, means = bundle["model"], bundle["labels"], bundle["class_means"]
    print(f"Loaded {bundle['model_name']} with {len(labels)} signs ({bundle['test_accuracy']:.0%} on unseen {bundle['tested_on']}s)")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Couldn't open the camera.")
    tracker = HandTracker(video=True)
    history = collections.deque(maxlen=SMOOTH)
    mode, target, held, score, flash = "free", random.choice(labels), 0, 0, 0.0
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hands = tracker.detect(frame, (time.time() - start) * 1000)
        view = cv2.flip(frame, 1)
        h, w = view.shape[:2]
        hand = hands[0] if hands else None
        cv2.rectangle(view, (0, 0), (w, 70), (30, 20, 10), -1)
        put_text(view, "FREE" if mode == "free" else f"PRACTICE   score {score}", (20, 45), 0.9, ORANGE, 2, box=False)
        help_text = "p: mode   n: next   q: quit"
        put_text(view, help_text, (w - 20 - text_width(help_text, 0.55, 1), 45), 0.55, GREY, 1, box=False)

        if hand is None or not hand.in_frame():
            history.clear()
            held = 0
            if hand:
                draw_hand(view, hand, default=GREY)
            put_text(view, "Show one hand, fully inside the frame", (20, h - 30), 0.9, WHITE)
        else:
            x = hand.features()
            history.append(model.predict_proba([x])[0])
            probs = np.mean(history, axis=0)
            top = int(np.argmax(probs))
            label, conf = labels[top], float(probs[top])

            if mode == "free":
                draw_hand(view, hand, default=GREEN if conf >= CONFIDENT else WHITE)
                put_text(view, label if conf >= CONFIDENT else "?", (20, 170), 3.0, GREEN if conf >= CONFIDENT else WHITE, 6)
                bar(view, 20, 190, 260, 16, conf, GREEN if conf >= CONFIDENT else ORANGE)
                for k, j in enumerate(np.argsort(probs)[::-1][:3]):
                    put_text(view, f"{labels[j]:<10} {probs[j]:.0%}", (20, 240 + 30 * k), 0.7, WHITE, 1)
            else:
                finger, err = worst_finger(x, means[target])
                correct, message = practice_feedback(label, conf, target, finger, err)
                colors = {n: GREEN for n in FINGERS}
                if not correct and err > FINGER_OFF:
                    colors[finger] = RED
                draw_hand(view, hand, colors)
                held = held + 1 if correct else 0
                put_text(view, f"Sign: {target}", (20, 150), 2.0, ORANGE, 5)
                bar(view, 20, 175, 260, 16, held / HOLD, GREEN)
                put_text(view, message, (20, h - 30), 0.9, GREEN if correct else WHITE)
                if held >= HOLD:
                    score += 1
                    flash = time.time() + 0.8
                    target = random.choice([l for l in labels if l != target] or labels)
                    held = 0
                    history.clear()

        if time.time() < flash:
            put_text(view, "Nice! +1", (w // 2 - 120, h // 2), 2.0, GREEN, 5)
        cv2.imshow("sign tutor", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            mode, held = ("practice" if mode == "free" else "free"), 0
        if key == ord("n") and mode == "practice":
            target, held = random.choice([l for l in labels if l != target] or labels), 0

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
