import cv2
import numpy as np
import pyrealsense2 as rs

# --- Параметри ---
DEPTH_THRESH_M = 0.10   # поріг ближче за стіну для удару
CALIBRATION_POINTS = []
ZONE_POLY = None
TRACKING_MODE = False
CONTACT_POINTS = []
POINT_LIFETIME = 30  # кадри

# --- Ініціалізація камери ---
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(cfg)

align = rs.align(rs.stream.color)

# --- Глобальні змінні ---
depth_wall = None
frame_counter = 0

# --- Клік мишкою для калібрування ---
def mouse_callback(event, x, y, flags, param):
    global CALIBRATION_POINTS, ZONE_POLY
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(CALIBRATION_POINTS) < 4:
            CALIBRATION_POINTS.append((x, y))
        if len(CALIBRATION_POINTS) == 4:
            ZONE_POLY = np.array(CALIBRATION_POINTS, np.int32)

cv2.namedWindow("Depth View")
cv2.setMouseCallback("Depth View", mouse_callback)

print("Натисни C для калібрування, T для трекінгу, Q для виходу.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Перетворення
        depth_img = np.asanyarray(depth_frame.get_data())
        color_img = np.asanyarray(color_frame.get_data())

        # Depth map -> колірна карта
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Калібрування
        if ZONE_POLY is not None and not TRACKING_MODE:
            cv2.polylines(depth_colormap, [ZONE_POLY], True, (0, 255, 0), 2)
            cv2.polylines(color_img, [ZONE_POLY], True, (0, 255, 0), 2)

        # Обробка в зоні
        if TRACKING_MODE and ZONE_POLY is not None and depth_wall is not None:
            mask = np.zeros_like(depth_img, dtype=np.uint8)
            cv2.fillPoly(mask, [ZONE_POLY], 255)

            wall_dist = depth_wall.copy()
            cur_depth = depth_img.copy()

            # Удар = ближче за стіну на DEPTH_THRESH_M
            diff = (wall_dist - cur_depth) > (DEPTH_THRESH_M / 0.001)
            diff = diff & (mask > 0)

            # Контури ударів
            contours, _ = cv2.findContours(
                diff.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                if cv2.contourArea(cnt) > 20:
                    (cx, cy), _ = cv2.minEnclosingCircle(cnt)
                    CONTACT_POINTS.append({
                        "pos": (int(cx), int(cy)),
                        "ttl": POINT_LIFETIME
                    })

        # Малюємо точки попадань
        for pt in CONTACT_POINTS:
            cv2.circle(depth_colormap, pt["pos"], 8, (0, 0, 255), -1)
            cv2.circle(color_img, pt["pos"], 8, (0, 0, 255), -1)
        CONTACT_POINTS = [p for p in CONTACT_POINTS if p.update({"ttl": p["ttl"] - 1}) or p["ttl"] > 0]

        # Підписи
        cv2.putText(depth_colormap, f"Mode: {'TRACK' if TRACKING_MODE else 'CALIB'} Thr={DEPTH_THRESH_M:.2f}m",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(color_img, f"Mode: {'TRACK' if TRACKING_MODE else 'CALIB'} Thr={DEPTH_THRESH_M:.2f}m",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Відображення
        cv2.imshow("Depth View", depth_colormap)
        cv2.imshow("RGB View", color_img)

        # Клавіші
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            CALIBRATION_POINTS.clear()
            ZONE_POLY = None
            TRACKING_MODE = False
            print("Вибери 4 точки зони на вікні Depth.")
        elif key == ord('t'):
            if ZONE_POLY is not None:
                depth_wall = depth_img.copy()
                TRACKING_MODE = not TRACKING_MODE
                print("Трекінг:", TRACKING_MODE)
        elif key == ord('+'):
            DEPTH_THRESH_M += 0.01
        elif key == ord('-'):
            DEPTH_THRESH_M = max(0.01, DEPTH_THRESH_M - 0.01)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()