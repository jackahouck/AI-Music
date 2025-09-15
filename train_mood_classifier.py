import os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

CSV = "data/mood/features.csv"
OUT = "models/mood_classifier.pkl"

df = pd.read_csv(CSV)

FEATS = ["tempo_bpm","note_density","avg_velocity","pitch_range","ioi_var","mode_major_prob"]
X = df[FEATS].values
y = df["mood"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

clf = RandomForestClassifier(random_state=1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))

os.makedirs("models", exist_ok=True)
joblib.dump(clf, OUT)
print(f"Saved classifer {OUT}")

disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.tight_layout()
plt.show()