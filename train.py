"""Train a sign classifier on data/signs.csv.

    python train.py                  # MLP (default)
    python train.py --model all      # compare MLP, random forest, SVM and k-NN, keep the best
    python train.py --model svm --augment 3

Saves models/sign_model.joblib and reports/confusion_matrix.png.
"""
import argparse
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from hands import LANDMARK_COLS, to_features  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "mlp": lambda: MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=800, early_stopping=True, random_state=0),
    "rf": lambda: RandomForestClassifier(n_estimators=300, random_state=0),
    "svm": lambda: CalibratedClassifierCV(SVC(C=10, gamma="scale"), ensemble=False),  # calibrated so it gives probabilities
    "knn": lambda: KNeighborsClassifier(n_neighbors=7),
}


def load(path):
    df = pd.read_csv(path)
    df["label"] = df["label"].astype(str)
    world = df[LANDMARK_COLS].to_numpy(dtype=np.float32)
    X = np.stack([to_features(w, h) for w, h in zip(world, df["handedness"])])
    return df, X, df["label"].to_numpy()


def split(df, test_size, seed):
    """Hold out whole people (3+ signers) or whole takes, never single frames.

    Frames from one take are near-duplicates. A random frame split puts copies of
    the same moment in train and test, and reports a fake ~100% accuracy.
    """
    by = "signer" if df["signer"].nunique() >= 3 else "take"
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df[by]))
    return train_idx, test_idx, by


def augment(X, y, copies, noise=0.02, seed=0):
    """Add jittered copies of the training data (noise in palm units)."""
    if copies <= 0:
        return X, y
    rng = np.random.default_rng(seed)
    extra = [X + rng.normal(0, noise, X.shape).astype(X.dtype) for _ in range(copies)]
    return np.vstack([X] + extra), np.concatenate([y] * (copies + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "signs.csv"))
    ap.add_argument("--model", default="mlp", choices=list(MODELS) + ["all"])
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--augment", type=int, default=0, help="jittered copies of the training data to add (try 2-5)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df, X, y = load(args.data)
    counts = df.groupby("label")["take"].nunique()
    print(f"{len(df)} frames, {len(counts)} signs, {df['signer'].nunique()} signer(s), {df['take'].nunique()} takes")
    if len(counts) < 2:
        raise SystemExit("Record at least two different signs first.")
    few = counts[counts < 2].index.tolist()
    if few:
        print(f"Warning: these signs have only one take, so they can't be tested fairly: {', '.join(few)}")

    train_idx, test_idx, by = split(df, args.test_size, args.seed)
    seen = set(y[train_idx])
    test_idx = np.array([i for i in test_idx if y[i] in seen])
    print(f"Testing on held-out {by}s: {len(train_idx)} training frames, {len(test_idx)} test frames\n")
    X_train, y_train = augment(X[train_idx], y[train_idx], args.augment, seed=args.seed)

    names = list(MODELS) if args.model == "all" else [args.model]
    results = {}
    for name in names:
        clf = make_pipeline(StandardScaler(), MODELS[name]())
        clf.fit(X_train, y_train)
        pred = clf.predict(X[test_idx])
        results[name] = (accuracy_score(y[test_idx], pred), pred)
        print(f"{name:>4}: {results[name][0]:.1%} accuracy on unseen {by}s")
    best = max(results, key=lambda n: results[n][0])
    acc, pred = results[best]
    if len(names) > 1:
        print(f"\nBest: {best}")

    y_test = y[test_idx]
    print("\n" + classification_report(y_test, pred, zero_division=0))
    labels = sorted(set(y))
    cm = confusion_matrix(y_test, pred, labels=labels)
    pairs = [(cm[i, j], labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels)) if i != j and cm[i, j]]
    for n, a, b in sorted(pairs, reverse=True)[:3]:
        print(f"Most confused: {a} predicted as {b} ({n} frames)")

    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), max(5, len(labels) * 0.55)))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=labels, normalize="true", values_format=".2f",
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"{best}: {acc:.1%} on unseen {by}s")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "reports", "confusion_matrix.png"), dpi=130)

    # Final model: same settings, trained on all the data.
    X_all, y_all = augment(X, y, args.augment, seed=args.seed)
    final = make_pipeline(StandardScaler(), MODELS[best]())
    final.fit(X_all, y_all)
    classes = list(final.classes_)
    bundle = {
        "model": final,
        "labels": classes,
        "class_means": {c: X[y == c].mean(axis=0) for c in classes},  # app.py compares fingers to these
        "test_accuracy": acc,
        "tested_on": by,
        "model_name": best,
    }
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    joblib.dump(bundle, os.path.join(HERE, "models", "sign_model.joblib"))
    print(f"\nSaved models/sign_model.joblib ({best}, {len(classes)} signs) and reports/confusion_matrix.png")


if __name__ == "__main__":
    main()
