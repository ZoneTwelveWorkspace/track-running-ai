import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
import cv2
from pynput.keyboard import Controller

from model import get_model
from capture_fast import grab_window_bgr, select_window, WIDTH, HEIGHT

DATA_DIR = Path("dataset")
META_FILE = DATA_DIR / "data.jsonl"

KEYS = ["w", "a", "s", "d"]

# =========================
# Dataset for training
# =========================
class WASDDataset(Dataset):
    def __init__(self):
        self.data = []

        if not META_FILE.exists():
            raise FileNotFoundError(f"{META_FILE} not found")

        with open(META_FILE) as f:
            for line in f:
                meta = json.loads(line)
                if meta.get("good"):
                    npz_path = DATA_DIR / meta["file"]
                    if not npz_path.exists():
                        continue
                    npz = np.load(npz_path)
                    frames, labels = npz["frames"], npz["labels"]
                    for x, y in zip(frames, labels):
                        self.data.append((x, y))

        if not self.data:
            raise ValueError("No good samples found for training")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # 1xHxW
        y_tensor = torch.tensor(y, dtype=torch.float32)  # 4-dim multi-label
        return x_tensor, y_tensor


# =========================
# Train model on MPS
# =========================
def train_model(epochs=5, batch_size=32, lr=1e-3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    dataset = WASDDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss={total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "wasd_cnn.pth")
    print("Training finished. Model saved as wasd_cnn.pth")
    return model

# =========================
# Main
# =========================
if __name__ == "__main__":
    train_model(epochs=8)
