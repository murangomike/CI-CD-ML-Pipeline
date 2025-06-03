from src.model import train_model

if __name__ == "__main__":
    acc = train_model()
    print(f"Model trained with accuracy: {acc:.2f}")
