import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
from collections import deque

# ---------- Параметри ----------
W, H, FPS = 640, 480, 30
LEFT_EDGE, RIGHT_EDGE = 0.33, 0.67   # пороги зон
EMA_ALPHA = 0.2                      # згладжування X (0..1)
ABSENCE_FRAMES = 10                  # скільки кадрів підряд вважаємо "нема людини"
MAX_VALID_DISTANCE_M = 3.0           # ігноруємо "виявлення" далі ніж 3м

# ---------- РеалСенс ----------
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
profile = pipeline.start(cfg)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()  # метри/од
align = rs.align(rs.stream.color)

# ---------- MediaPipe Pose ----------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------- Хелпери ----------
def decide_zone(x_norm, last_zone):
    # Гістерезис, щоб не скакало на межах
    left_th  = LEFT_EDGE  - 0.03 if last_zone == "LEFT"  else LEFT_EDGE
    right_th = RIGHT_EDGE + 0.03 if last_zone == "RIGHT" else RIGHT_EDGE
    if x_norm < left_th:
        return "LEFT"
    if x_norm > right_th:
        return "RIGHT"
    return "CENTER"

ema_x = None
last_zone = "CENTER"
no_person_ctr = 0
trail = deque(maxlen=20)  # маленький хвостик траєкторії

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            continue

        color_img = np.asanyarray(color.get_data())
        depth_img  = np.asanyarray(depth.get_data())

        # MediaPipe очікує RGB
        rgb_for_mp = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb_for_mp)

        person_present = False
        x_norm = None
        depth_ok = False

        if res.pose_landmarks:
            # Візьмемо центр тіла: середина стегон (між LEFT_HIP і RIGHT_HIP)
            lh = res.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            rh = res.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            x_norm = (lh.x + rh.x) / 2.0
            y_norm = (lh.y + rh.y) / 2.0

            # Перевіримо дистанцію в цій точці
            x_px = int(np.clip(x_norm * W, 0, W - 1))
            y_px = int(np.clip(y_norm * H, 0, H - 1))
            dist_m = depth.get_distance(x_px, y_px)

            if 0.1 < dist_m < MAX_VALID_DISTANCE_M:
                depth_ok = True
                person_present = True

                # Візуалізація
                cv2.circle(color_img, (x_px, y_px), 6, (0, 255, 255), -1)
                cv2.putText(color_img, f"{dist_m:.2f} m", (x_px+8, y_px-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)

        # Логіка зон
        if person_present and depth_ok:
            no_person_ctr = 0
            if ema_x is None:
                ema_x = x_norm
            else:
                ema_x = EMA_ALPHA * x_norm + (1 - EMA_ALPHA) * ema_x

            zone = decide_zone(ema_x, last_zone)
            if zone != last_zone:
                # тут можна виводити подію/надсилати в Unity
                print(f"EVENT: {zone}")
                last_zone = zone

            # Малюємо зони
            L = int(LEFT_EDGE * W)
            R = int(RIGHT_EDGE * W)
            cv2.line(color_img, (L, 0), (L, H), (80, 80, 80), 1)
            cv2.line(color_img, (R, 0), (R, H), (80, 80, 80), 1)

            cx = int(ema_x * W)
            trail.append((cx, int(y_norm * H)))
            for i in range(1, len(trail)):
                cv2.line(color_img, trail[i-1], trail[i], (0, 255, 0), 2)

            cv2.putText(color_img, f"ZONE: {last_zone}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            no_person_ctr += 1
            if no_person_ctr > ABSENCE_FRAMES and last_zone != "NO_PERSON":
                print("EVENT: NO_PERSON")
                last_zone = "NO_PERSON"
                ema_x = None
                trail.clear()

            cv2.putText(color_img, "NO PERSON", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        # Показ кольору + «фарбованої» глибини поруч (зручно дебажити)
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=0.03),
            cv2.COLORMAP_JET
        )
        view = np.hstack([color_img, depth_vis])
        cv2.imshow("Person direction (color | depth)", view)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()