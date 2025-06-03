from src.predict import predict

def test_model_training():
    acc = train_model()
    assert acc > 0.7  # Basic accuracy threshold
