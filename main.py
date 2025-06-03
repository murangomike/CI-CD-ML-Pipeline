from src.model import train_model

if __name__ == "__main__":
    accuracy = train_model()
    print(f"Model trained with accuracy: {accuracy:.4f}")