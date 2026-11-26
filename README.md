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

