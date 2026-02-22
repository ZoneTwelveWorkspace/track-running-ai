import cv2
import numpy as np
import json
import time
from pathlib import Path
from pynput import keyboard
from capture_fast import grab_window_bgr, select_window, WIDTH, HEIGHT

# Configuration
DATA_DIR = Path("dataset")
DATA_DIR.mkdir(exist_ok=True)
META_FILE = DATA_DIR / "data.jsonl"

KEYS_TO_TRACK = ["w", "a", "s", "d"]
current_keys = {k: 0 for k in KEYS_TO_TRACK}
recording = False  # Global recording state

def save_data(frames, labels):
    if not frames: return
    timestamp = int(time.time())
    filename = f"drive_{timestamp}.npz"
    filepath = DATA_DIR / filename
    np.savez_compressed(filepath, frames=np.array(frames), labels=np.array(labels))
    meta = {"file": filename, "good": True, "count": len(frames)}
    with open(META_FILE, "a") as f:
        f.write(json.dumps(meta) + "\n")
    print(f"\n[SAVED] {filename} with {len(frames)} frames.")

# --- Improved Global Listener ---
def on_press(key):
    global recording
    try:
        # Use 'r' to toggle recording globally
        if hasattr(key, 'char') and key.char == 'r':
            recording = not recording
            state = "STARTED" if recording else "STOPPED"
            print(f"\n[RECORDING {state}]")
            return 

        # Track driving keys
        k = key.char.lower()
        if k in current_keys:
            current_keys[k] = 1
    except AttributeError:
        pass

def on_release(key):
    try:
        k = key.char.lower()
        if k in current_keys:
            current_keys[k] = 0
    except AttributeError:
        pass
    if key == keyboard.Key.esc:
        return False

def record_session():
    global recording
    window_id = select_window("Roblox")
    
    # Start the global listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    frames = []
    labels = []
    
    print("--- RECORDER ACTIVE ---")
    print("1. Click into Roblox to drive.")
    print("2. Press 'R' ANYWHERE to start/stop recording.")
    print("3. Press 'ESC' to exit script.")

    try:
        while listener.running:
            bgr = grab_window_bgr(window_id)
            if bgr is None: continue

            # Process frame for preview
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (WIDTH, HEIGHT))
            
            # Logic for data collection
            if recording:
                frames.append(resized)
                labels.append([current_keys[k] for k in KEYS_TO_TRACK])
            elif not recording and len(frames) > 0:
                # We just stopped recording, save the buffer
                save_data(frames, labels)
                frames, labels = [], []

            # Visualization
            viz = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            if recording:
                # Draw a recording indicator
                cv2.circle(viz, (WIDTH-15, 15), 5, (0, 0, 255), -1)
                cv2.putText(viz, "REC", (WIDTH-45, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Overlay active keys for debugging
            active_keys = [k for k, v in current_keys.items() if v == 1]
            cv2.putText(viz, f"Keys: {active_keys}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            cv2.imshow("Data Recorder Monitor", cv2.resize(viz, (640, 360)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        listener.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    record_session()