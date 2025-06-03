import joblib
import numpy as np

def predict(X):
    model = joblib.load("models/model.joblib")
    return model.predict(X)
