"""Record signs from your webcam into data/signs.csv.

    python collect.py --signs A,B,L,V,Y --signer priya

Keys:  SPACE record a take   ] next sign   [ previous sign   q quit

Each take saves 40 frames of the sign you're holding. Aim for 5+ takes per sign,
moving your hand a little between takes (closer, further, slightly tilted).
Ask friends to record too: train.py can then test on people the model never saw.
"""
import argparse
import csv
import os
import time

import cv2

from hands import LANDMARK_COLS, HandTracker, draw_hand, put_text

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "signs.csv")
HEADER = ["label", "signer", "take", "handedness"] + LANDMARK_COLS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signs", required=True, help="comma-separated sign names, e.g. A,B,HELLO")
    ap.add_argument("--signer", required=True, help="your name (used to test on unseen people)")
    ap.add_argument("--frames", type=int, default=40, help="frames per take (default 40)")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    signs = [s.strip().upper() for s in args.signs.split(",") if s.strip()]
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    new_file = not os.path.exists(DATA)
    f = open(DATA, "a", newline="")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(HEADER)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Couldn't open the camera. Close other apps using it, or try --camera 1.")
    tracker = HandTracker(video=True)
    i, state, t0, saved, take = 0, "ready", 0.0, 0, ""
    takes = {s: 0 for s in signs}
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hands = tracker.detect(frame, (time.time() - start) * 1000)
        hand = hands[0] if hands else None
        good = hand is not None and hand.in_frame()
        view = cv2.flip(frame, 1)  # mirror so it feels natural; detection used the unmirrored frame
        if hand:
            draw_hand(view, hand, default=(0, 220, 120) if good else (80, 80, 255))

        if state == "countdown" and time.time() - t0 > 3:
            state, saved = "recording", 0
        if state == "recording" and good:
            writer.writerow([signs[i], args.signer, take, hand.handedness] + [f"{v:.5f}" for p in hand.world for v in p])
            saved += 1
            if saved >= args.frames:
                takes[signs[i]] += 1
                state = "ready"

        put_text(view, f"Sign: {signs[i]}   ({takes[signs[i]]} takes)", (20, 45), 1.1, (0, 200, 255), 3)
        if state == "ready":
            put_text(view, "SPACE record   ] next   [ previous   q quit", (20, view.shape[0] - 25), 0.7)
        elif state == "countdown":
            put_text(view, f"Get ready... {3 - int(time.time() - t0)}", (20, view.shape[0] - 25), 1.0, (0, 200, 255))
        else:
            put_text(view, f"Recording {saved}/{args.frames}" + ("" if good else "  (show your whole hand)"),
                     (20, view.shape[0] - 25), 1.0, (80, 80, 255))
        cv2.imshow("collect", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if state == "ready" and key == ord(" "):
            state, t0, take = "countdown", time.time(), f"{args.signer}-{signs[i]}-{int(time.time())}"
        elif state == "ready" and key == ord("]"):
            i = (i + 1) % len(signs)
        elif state == "ready" and key == ord("["):
            i = (i - 1) % len(signs)

    f.close()
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Takes recorded:", ", ".join(f"{s} {n}" for s, n in takes.items()))


if __name__ == "__main__":
    main()
