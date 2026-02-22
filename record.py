import time
import numpy as np
import cv2
import csv
import json
from pathlib import Path
from pynput import keyboard

from capture_fast import grab_window_bgr, select_window, WIDTH, HEIGHT

DATA_DIR = Path("dataset")
DATA_DIR.mkdir(exist_ok=True)
META_FILE = DATA_DIR / "data.jsonl"

FPS_TARGET = 30
RECORD_SECONDS = int(input("Record duration (seconds)? "))
KEYS = ["w", "a", "s", "d"]


# =========================
# Key tracking
# =========================
class KeyTracker:
    def __init__(self):
        self.state = {k: 0 for k in KEYS}
        self.events = []

    def on_press(self, key):
        try:
            ch = key.char.lower()
        except:
            return
        if ch in self.state and self.state[ch] == 0:
            self.state[ch] = 1
            t = time.time()
            self.events.append((t, ch, "down"))
            print(f"[KEY] {ch.upper()} DOWN")

    def on_release(self, key):
        try:
            ch = key.char.lower()
        except:
            return
        if ch in self.state and self.state[ch] == 1:
            self.state[ch] = 0
            t = time.time()
            self.events.append((t, ch, "up"))
            print(f"[KEY] {ch.upper()} UP")

    def label(self):
        return np.array([self.state[k] for k in KEYS], dtype=np.int8)


def main():
    window_id = select_window("Roblox")
    tracker = KeyTracker()
    listener = keyboard.Listener(
        on_press=tracker.on_press,
        on_release=tracker.on_release
    )
    listener.start()

    frames = []
    labels = []
    timestamps = []

    print(f"\nRecording for {RECORD_SECONDS}s...")

    start_time = time.time()
    next_frame_time = start_time
    dt = 1.0 / FPS_TARGET

    try:
        while True:
            now = time.time()
            if now - start_time > RECORD_SECONDS:
                break
            if now < next_frame_time:
                time.sleep(next_frame_time - now)
            next_frame_time += dt

            bgr = grab_window_bgr(window_id)
            if bgr is None:
                continue  # window minimized or hidden

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            x = resized.astype(np.float32) / 255.0

            frames.append(x)
            labels.append(tracker.label())
            timestamps.append(time.time())

            # Preview
            preview = cv2.resize(resized, (640, 360))
            cv2.putText(preview, f"action={tracker.label()}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Record Preview", preview)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            mean = float(x.mean())
            std = float(x.std())
            print(f"mean={mean:.3f} std={std:.3f} action={tracker.label()}")

    finally:
        listener.stop()
        cv2.destroyAllWindows()

    # Save dataset
    index = int(time.time())
    dataset_file = DATA_DIR / f"{index}.npz"
    frames = np.asarray(frames, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int8)
    timestamps = np.asarray(timestamps, dtype=np.float64)

    np.savez_compressed(dataset_file,
                        frames=frames,
                        labels=labels,
                        timestamps=timestamps)
    print("Saved dataset:", dataset_file, frames.shape)

    # Save events CSV
    events_file = DATA_DIR / f"{index}_events.csv"
    with open(events_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "key", "event"])
        for row in tracker.events:
            writer.writerow(row)
    print("Saved events:", events_file, len(tracker.events))

    # =========================
    # Label good/bad
    # =========================
    while True:
        label_input = input("Is this sample good? (y/n): ").strip().lower()
        if label_input in ("y", "yes", "n", "no"):
            break
    good = label_input in ("y", "yes")

    meta = {
        "file": f"{index}.npz",
        "events": f"{index}_events.csv",
        "good": good
    }

    with open(META_FILE, "a") as f:
        f.write(json.dumps(meta) + "\n")
    print("Saved metadata:", meta)


if __name__ == "__main__":
    main()

