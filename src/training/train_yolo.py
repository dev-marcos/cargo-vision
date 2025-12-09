import os
import yaml
import shutil
from ultralytics import YOLO

# Moving 3 levels up gives: project_root/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def load_training_config(config_path: str) -> dict:
    """Loads training configuration parameters from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_yolo_model() -> None:
    """Trains a YOLOv8 model based on configuration settings."""
    config = load_training_config("configs/train_config.yaml")

    print("Initializing YOLO model...")
    model = YOLO(config["base_model"])

    # YOLO default folder (we force an absolute path)
    runs_path = os.path.join(ROOT_DIR, "runs", "detect")
    print(f"Training results will be stored in: {runs_path}")

    # Run training
    model.train(
        data=config["data_yaml"],
        epochs=config["epochs"],
        batch=config["batch_size"],
        imgsz=config["img_size"],
        workers=config["workers"],
        device=config["device"],
        patience=config["patience"],
        amp=config["amp"],
        project=runs_path,
        name="train",
        exist_ok=True,
    )

    print("Training finished.")

     # ============================================================
    # Locate YOLO output and copy best.pt to models/best.pt
    # ============================================================
    weights_dir = os.path.join(runs_path, "train", "weights")
    best_src = os.path.join(weights_dir, "best.pt")
    last_src = os.path.join(weights_dir, "last.pt")

    # Ensure models/ directory exists
    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Copy BEST model
    best_dest = os.path.join(models_dir, "best.pt")
    if os.path.exists(best_src):
        shutil.copy(best_src, best_dest)
        print(f"Copied best.pt to: {best_dest}")
    else:
        print("Warning: best.pt not found after training!")

    # Optional: also copy LAST model
    last_dest = os.path.join(models_dir, "last.pt")
    if os.path.exists(last_src):
        shutil.copy(last_src, last_dest)
        print(f"Copied last.pt to: {last_dest}")

    print("\nAll training artifacts processed successfully.")



if __name__ == "__main__":
    train_yolo_model()
