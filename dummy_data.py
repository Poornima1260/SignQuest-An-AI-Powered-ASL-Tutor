import os
import numpy as np
import pandas as pd
from hands import LANDMARK_COLS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "data", "signs.csv")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

rng = np.random.default_rng(42)

# Base 21x3 hand templates (in metres) for a few example signs
def make_pose(sign_idx):
    pts = np.zeros((21, 3), dtype=np.float32)
    # Wrist at origin (0, 0, 0), palm extending along -y
    # Finger bases (MCPs): thumb=1, index=5, middle=9, ring=13, pinky=17
    bases = {
        1: np.array([-0.03, -0.03, 0.01]),
        5: np.array([-0.035, -0.08, 0.0]),
        9: np.array([0.0, -0.085, 0.0]),
        13: np.array([0.03, -0.08, 0.0]),
        17: np.array([0.055, -0.07, 0.0]),
    }
    # Different curl/spread patterns per sign class
    curl_scale = [0.025, 0.012, 0.020, 0.006][sign_idx % 4]
    spread_x = [-0.012, 0.015, 0.0, 0.022][sign_idx % 4]
    z_tilt = [-0.015, 0.015, 0.0, -0.01][sign_idx % 4]

    for base_idx, base_pos in bases.items():
        pts[base_idx] = base_pos
        for j in range(1, 4):
            pts[base_idx + j] = base_pos + np.array([
                spread_x * j * (0.4 if base_idx == 1 else 1.0),
                -curl_scale * j,
                z_tilt * j,
            ])
    return pts

signs = ["hello", "thanks", "yes", "no"]
signers = ["signer_1", "signer_2", "signer_3"]
takes_per_signer = 2
frames_per_take = 25

rows = []
for s_idx, label in enumerate(signs):
    base_pts = make_pose(s_idx)
    for signer in signers:
        for t in range(1, takes_per_signer + 1):
            take = f"{signer}_take_{t}"
            handedness = "Right" if signer != "signer_2" else "Left"
            # Slight per-take shift so frames within a take stay coherent
            take_offset = rng.normal(0, 0.004, size=(21, 3)).astype(np.float32)
            for _ in range(frames_per_take):
                jitter = rng.normal(0, 0.002, size=(21, 3)).astype(np.float32)
                frame_pts = base_pts + take_offset + jitter
                row = {
                    "label": label,
                    "signer": signer,
                    "take": take,
                    "handedness": handedness,
                }
                for col, val in zip(LANDMARK_COLS, frame_pts.flatten()):
                    row[col] = float(val)
                rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False)
print(f"Wrote {len(df)} rows to {OUT_PATH}")