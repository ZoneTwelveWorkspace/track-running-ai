import cv2
import numpy as np
from capture_fast import grab_window_bgr, select_window, WIDTH, HEIGHT


# =========================
# Track Detection
# =========================
def detect_track_mask(bgr):

    b = bgr[:,:,0].astype(np.float32)
    g = bgr[:,:,1].astype(np.float32)
    r = bgr[:,:,2].astype(np.float32)

    mean = (r + g + b) / 3.0

    rms = np.sqrt(((r-mean)**2 + (g-mean)**2 + (b-mean)**2) / 3.0)

    # gray pixels have small rms
    gray_mask = (rms < 18).astype(np.uint8) * 255

    # also require medium brightness
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bright_mask = cv2.inRange(gray, 110, 200)

    track_mask = cv2.bitwise_and(gray_mask, bright_mask)

    # clean noise
    kernel = np.ones((5,5), np.uint8)
    track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_CLOSE, kernel)
    track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_OPEN, kernel)

    # keep region connected to center
    h, w = track_mask.shape
    cx = w // 2
    cy = h // 2

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(track_mask)

    center_label = labels[cy, cx]

    final_mask = np.zeros_like(track_mask)
    final_mask[labels == center_label] = 255

    return final_mask

# =========================
# Reward Function
# =========================
def compute_reward(track_mask):

    h, w = track_mask.shape
    cx = w // 2
    cy = h // 2

    patch = track_mask[cy-3:cy+3, cx-3:cx+3]

    on_track_ratio = np.mean(patch) / 255.0

    if on_track_ratio > 0.6:
        reward = 1.0
        on_track = True
    else:
        reward = -1.0
        on_track = False

    return reward, (cx, cy), on_track, on_track_ratio


# =========================
# Visualization
# =========================
def draw_overlay(gray, track_mask, reward, center, on_track, ratio):

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # color detected track
    overlay[track_mask > 0] = (0, 200, 0)

    cx, cy = center

    color = (0, 255, 0) if on_track else (0, 0, 255)
    cv2.circle(overlay, (cx, cy), 6, color, -1)

    cv2.putText(
        overlay,
        f"Reward: {reward:.2f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        overlay,
        f"Track ratio: {ratio:.2f}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    return overlay


# =========================
# Monitor Loop
# =========================
def monitor():

    window_id = select_window("Roblox")

    while True:

        bgr = grab_window_bgr(window_id)

        if bgr is None:
            continue

        # resize once
        bgr = cv2.resize(bgr, (WIDTH, HEIGHT))

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        track_mask = detect_track_mask(bgr)

        reward, center, on_track, ratio = compute_reward(track_mask)

        overlay = draw_overlay(gray, track_mask, reward, center, on_track, ratio)

        preview = cv2.resize(overlay, (640, 360))

        cv2.imshow("Race Track Monitor", preview)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cv2.destroyAllWindows()


# =========================
# Main
# =========================
if __name__ == "__main__":
    monitor()