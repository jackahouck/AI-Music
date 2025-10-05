import numpy as np, joblib, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X    = np.load("data/processed/X_pd.npy")
y_p  = np.load("data/processed/y_pitch.npy")
y_d  = np.load("data/processed/y_dclass.npy")

y = np.stack([y_p, y_d], axis=1)  # shape (N, 2)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

base = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf  = MultiOutputClassifier(base)
clf.fit(Xtr, ytr)

yp = clf.predict(Xte)
pitch_acc = accuracy_score(yte[:,0], yp[:,0])
dur_acc   = accuracy_score(yte[:,1], yp[:,1])
print(f"Pitch acc: {pitch_acc:.3f}  |  Duration acc: {dur_acc:.3f}")

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/baseline_pd.pkl")
print("Saved → models/baseline_pd.pkl")
