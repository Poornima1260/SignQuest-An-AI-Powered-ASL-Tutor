# Sign tutor (Python)

Record hand signs with your webcam, train a scikit-learn classifier on them, and practise with live feedback.

```
collect.py   webcam -> data/signs.csv          (you record the signs)
train.py     data/signs.csv -> models/...       (trains, tests, saves the model)
app.py       live recognition + practice mode   (uses the saved model)
extract.py   optional: add an image/video dataset to data/signs.csv
hands.py     shared code: MediaPipe tracking, features, drawing
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads MediaPipe's hand model (about 8 MB) into `models/`.
Tested with Python 3.13, mediapipe 0.10.32 and scikit-learn 1.9. If pip can't
find mediapipe, your Python version may be newer than MediaPipe supports; try a
slightly older Python.

## 1. Record

```bash
python collect.py --signs A,B,L,V,Y --signer priya
```

SPACE records a take (3-second countdown, then 40 frames). `]` and `[` change sign, `q` quits.
Aim for **5+ takes per sign** and move your hand a little between takes. Ask
**2 or more friends** to record too (`--signer their_name`), because that's what
lets `train.py` measure accuracy on people the model has never seen.

## 2. Train

```bash
python train.py                 # MLP neural network
python train.py --model all     # compare MLP, random forest, SVM, k-NN; keeps the best
python train.py --augment 3     # add noisy copies of the training data
```

It prints accuracy, a per-sign report and the most-confused pairs, then saves
`models/sign_model.joblib` and `reports/confusion_matrix.png`.

## 3. Practise

```bash
python app.py
```

- **Free mode** shows what the model thinks you're signing, with the top 3 guesses.
- **Practice mode** (press `p`) gives you a sign to make. The finger furthest from
  how that sign looked in your training data turns red, with one correction like
  "Check your index finger". Hold the right sign to score.

## How it works

**Features** (`hands.to_features`): MediaPipe gives 21 3D hand landmarks. We move
the wrist to the origin, mirror left hands so they match right hands, and scale
so wrist-to-middle-knuckle = 1. That's 63 numbers per frame. Rotation is kept on
purpose, because palm direction can change a sign's meaning.

**Model**: `StandardScaler` + `MLPClassifier(128, 64)` by default. Predictions are
averaged over 8 frames in the app to stop flicker.

**Honest evaluation**: frames within one take are near-duplicates, so a random
train/test split leaks and reports ~100%. `train.py` holds out **whole signers**
(when there are 3 or more) or **whole takes**, and says which it used.

**Feedback**: `train.py` stores each sign's average features. `app.py` compares
your hand to that average finger by finger and flags the furthest one.

## Adding more signs

- New signs: run `collect.py` again with new `--signs`, then retrain. Everything
  appends to `data/signs.csv`.
- Datasets: put files in one folder per sign (`dataset/HELLO/*.jpg|*.mp4`) and run
  `python extract.py dataset --signer dataset_name`. Video datasets of words
  (like INCLUDE for Indian Sign Language) mostly contain moving signs; this
  takes frames from the middle of each clip, so treat it as a rough start.

## Ideas to try next

- Tune hyperparameters with `GridSearchCV` + `GroupKFold` (group by signer).
- Report accuracy per signer, not just overall.
- Two-handed signs (most of the ISL alphabet): set `max_hands=2` in `HandTracker`
  and concatenate both hands' features (126 numbers) plus their relative position.
- Moving signs (J, Z, most words): record 20-30 frames per sample and train a
  sequence model, e.g. flatten the frames into one vector for the MLP, or move
  to a 1D-CNN/LSTM in PyTorch.
