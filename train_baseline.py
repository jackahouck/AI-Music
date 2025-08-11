import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

print('X:', X.shape, 'y:', y.shape, 'min/max pitch:', y.min(), y.max())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= 0.2, random_state=42, 
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Accuracy', accuracy_score(y_test, y_pred))