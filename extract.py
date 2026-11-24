"""Optional: add a dataset of sign images or videos to data/signs.csv.

    dataset/
      HELLO/   img1.jpg  clip1.mp4 ...
      THANKS/  ...

    python extract.py path/to/dataset --signer include_dataset

Videos: a few frames from the middle of each clip are used, which suits signs
that hold a handshape. If your dataset has one folder per person inside each
sign folder (dataset/HELLO/person1/...), add --signer-from-folders.
"""
import argparse
import csv
import os

import cv2

from collect import DATA, HEADER
from hands import HandTracker

IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEOS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def frames_of(path, per_video):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGES:
        img = cv2.imread(path)
        if img is not None:
            yield img
        return
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    for k in range(per_video):  # spread over the middle 40% of the clip
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (0.3 + 0.4 * k / max(1, per_video - 1))))
        ok, img = cap.read()
        if ok:
            yield img
    cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="folder with one subfolder per sign")
    ap.add_argument("--signer", default="dataset", help="signer name to store (default: dataset)")
    ap.add_argument("--signer-from-folders", action="store_true", help="use dataset/<sign>/<person>/ folder names as signers")
    ap.add_argument("--frames", type=int, default=6, help="frames taken from each video (default 6)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    new_file = not os.path.exists(DATA)
    with open(DATA, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(HEADER)
        tracker = HandTracker(video=False)
        for sign in sorted(os.listdir(args.dataset)):
            folder = os.path.join(args.dataset, sign)
            if not os.path.isdir(folder):
                continue
            rows = files = 0
            for root, _, names in os.walk(folder):
                for name in sorted(names):
                    if os.path.splitext(name)[1].lower() not in IMAGES | VIDEOS:
                        continue
                    path = os.path.join(root, name)
                    rel = os.path.relpath(path, folder)
                    signer = rel.split(os.sep)[0] if args.signer_from_folders and os.sep in rel else args.signer
                    files += 1
                    for img in frames_of(path, args.frames):
                        hands = tracker.detect(img)
                        if hands and hands[0].in_frame():
                            hand = hands[0]
                            writer.writerow([sign.upper(), signer, f"{sign}/{rel}", hand.handedness]
                                            + [f"{v:.5f}" for p in hand.world for v in p])
                            rows += 1
            print(f"{sign}: {files} files -> {rows} frames with a hand")
        tracker.close()
    print(f"Added to {DATA}. Now run train.py.")


if __name__ == "__main__":
    main()
