import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
import cv2

from model import get_model
from capture_fast import WIDTH, HEIGHT

# Configuration
DATA_DIR = Path("dataset")
META_FILE = DATA_DIR / "data.jsonl"
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 20
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class RaceDataset(Dataset):
    def __init__(self):
        self.samples = []
        if not META_FILE.exists():
            print("No metadata file found. Record some data first!")
            return

        with open(META_FILE, "r") as f:
            for line in f:
                meta = json.loads(line)
                # Only train on data marked as "good"
                if meta.get("good", False):
                    file_path = DATA_DIR / meta["file"]
                    if file_path.exists():
                        data = np.load(file_path)
                        frames = data["frames"]
                        labels = data["labels"]
                        for f_idx in range(len(frames)):
                            self.samples.append((frames[f_idx], labels[f_idx]))
        
        print(f"Loaded {len(self.samples)} total frames for training.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame, label = self.samples[idx]
        # Normalize and add channel dimension (1, H, W)
        frame_t = torch.tensor(frame / 255.0, dtype=torch.float32).unsqueeze(0)
        label_t = torch.tensor(label, dtype=torch.float32)
        return frame_t, label_t

def train():
    dataset = RaceDataset()
    if len(dataset) == 0:
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = get_model().to(DEVICE)
    
    # Load existing weights if they exist to continue training
    model_path = Path("wasd_cnn.pth")
    if model_path.exists():
        print("Loading existing model weights...")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss() # Binary Cross Entropy for multi-label (W,A,S,D)

    print(f"Training on {DEVICE}...")
    model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0
        for frames, labels in loader:
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

    # Save the updated model
    torch.save(model.state_dict(), "wasd_cnn.pth")
    print("Model saved as wasd_cnn.pth")

if __name__ == "__main__":
    train()