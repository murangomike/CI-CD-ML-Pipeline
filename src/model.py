from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

from src.data_loader import load_data

def train_model():
    X_train, X_test, y_train, y_test = load_data()
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join("models", "model.joblib"))

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc}")
    return acc
