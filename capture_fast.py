import time
import numpy as np
import cv2

from Quartz import (
    CGWindowListCopyWindowInfo,
    CGWindowListCreateImage,
    CGRectMake,
    CGRectNull,
    kCGWindowListOptionOnScreenOnly,
    kCGNullWindowID,
    kCGWindowImageDefault,
    kCGWindowImageBoundsIgnoreFraming,
    kCGWindowImageNominalResolution,
    kCGWindowListOptionIncludingWindow,
    CGDataProviderCopyData,
    CGImageGetDataProvider,
    CGImageGetWidth,
    CGImageGetHeight,
    CGImageGetBytesPerRow,
)

WIDTH, HEIGHT = 160, 120
FPS = 30  # try higher now

def list_windows():
    windows = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    valid = []
    for w in windows:
        owner = w.get("kCGWindowOwnerName", "")
        name = w.get("kCGWindowName", "")
        bounds = w.get("kCGWindowBounds", {})
        wid = w.get("kCGWindowNumber", None)
        if owner and bounds and wid is not None:
            valid.append({"owner": owner, "name": name, "bounds": bounds, "wid": wid})

    print("\n=== Visible Windows ===")
    for i, w in enumerate(valid):
        print(f"[{i}] id={w['wid']} | {w['owner']} | {w['name']}")
    return valid

def select_window(keyword = None):
    windows = list_windows()
    target = next((item for item in windows if item['name'] == keyword), None)
    if keyword != None and target != None:
        return target['wid']
    else:
        idx = int(input("\nSelect window index: "))
        
    w = windows[idx]
    print(f"\nTracking window: id={w['wid']} | {w['owner']} | {w['name']}")
    return w["wid"]

def cgimage_to_bgr(cgimg):
    # Quartz usually gives BGRA (little-endian). We’ll reshape and drop alpha.
    w = CGImageGetWidth(cgimg)
    h = CGImageGetHeight(cgimg)
    bpr = CGImageGetBytesPerRow(cgimg)
    data = CGDataProviderCopyData(CGImageGetDataProvider(cgimg))
    buf = np.frombuffer(data, dtype=np.uint8)
    bgra = buf.reshape((h, bpr // 4, 4))[:, :w, :]   # trim row padding
    bgr = bgra[:, :, :3]
    return bgr

def grab_window_bgr(window_id):
    # Include only that window; ignore framing for speed/consistency.
    cgimg = CGWindowListCreateImage(
        CGRectNull,
        kCGWindowListOptionIncludingWindow,
        window_id,
        kCGWindowImageBoundsIgnoreFraming | kCGWindowImageNominalResolution
    )
    if cgimg is None:
        return None
    return cgimage_to_bgr(cgimg)

def main():
    window_id = select_window()

    print("\nFast window capture started. Press Q to quit.\n")
    next_t = time.perf_counter()
    dt = 1.0 / FPS

    while True:
        # schedule-based throttle (more stable than sleep(1/FPS - elapsed))
        now = time.perf_counter()
        if now < next_t:
            time.sleep(next_t - now)
        next_t += dt

        bgr = grab_window_bgr(window_id)
        if bgr is None:
            # window minimized/hidden etc.
            continue

        # Fast grayscale + resize
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

        # Keep mean/std cheap
        x_norm = resized.astype(np.float16) * (1.0 / 255.0)
        mean = float(x_norm.mean())
        std = float(x_norm.std())

        # Preview (avoid extra convert+resize if you don't need it)
        preview = cv2.resize(resized, (640, 360), interpolation=cv2.INTER_NEAREST)
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

        cv2.putText(preview, f"mean={mean:.3f} std={std:.3f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Fast Window Capture", preview)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
