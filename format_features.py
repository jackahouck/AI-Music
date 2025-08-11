import numpy as np
import pandas as pd
import os

SEQ_LEN = 3

df = pd.read_csv("data/processed/notes.csv")

pitches = df['pitch'].astype(int).to_numpy()
print(pitches[:10])

#Separating into X (Training) and y (Prediction)
X, y = [], []
for i in range(len(pitches) - SEQ_LEN):
    X.append(pitches[i:i+SEQ_LEN])
    y.append(pitches[i+SEQ_LEN])

X = np.array(X, dtype=np.int16)
y = np.array(y, dtype=np.int16)

os.makedirs("data/processed", exist_ok=True)
np.save("data/processed/X.npy", X)
np.save("data/processed/y.npy", y)
print("saved X.npy and y.npy")