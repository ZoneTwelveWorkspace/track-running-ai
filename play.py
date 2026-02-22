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
# Real-time play
# =========================
def play(model, threshold=0.5, patience=30):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    window_id = select_window("Roblox")
    kb = Controller()
    model.eval()

    low_prob_counter = 0  # count consecutive low-prob frames

    with torch.no_grad():
        while True:
            bgr = grab_window_bgr(window_id)
            if bgr is None:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (WIDTH, HEIGHT))
            norm = torch.tensor(resized / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            out = model(norm).squeeze().cpu().numpy()  # 4-dim array
            print("Logits", [f"{n:.2f}" for n in out], low_prob_counter)

            # Check if all probabilities are below threshold
            if np.all(out < threshold):
                low_prob_counter += 1
            else:
                low_prob_counter = 0

            # Decide which keys to press
            if low_prob_counter >= patience:
                # Pick the highest logit
                max_idx = np.argmax(out)
                for i, k in enumerate(KEYS):
                    if i == max_idx:
                        print("trigger soft move", k)
                        kb.press(k)
                    else:
                        kb.release(k)
            else:
                # Normal threshold-based press
                for i, k in enumerate(KEYS):
                    if out[i] > threshold:
                        print("Trigger", k)
                        kb.press(k)
                    else:
                        kb.release(k)

            # Preview
            preview = cv2.resize(resized, (640, 360))
            cv2.putText(
                preview,
                f"pred={['1' if o>0.5 else '0' for o in out]}, low_count={low_prob_counter}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            cv2.imshow("Play Preview", preview)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cv2.destroyAllWindows()

# =========================
# Main
# =========================
if __name__ == "__main__":
    model = get_model()
    model.load_state_dict(torch.load("wasd_cnn.pth"))
    play(model)
