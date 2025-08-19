import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
from collections import deque

# ---------- Налаштування ----------
W, H, FPS = 640, 480, 30

# Перевірка присутності/глибини
MAX_VALID_DISTANCE_M = 3.0
ABSENCE_FRAMES = 12

# Згладжування та детекція руху
EMA_ALPHA = 0.25               # згладжування X (0..1)
VEL_ALPHA = 0.4                # згладжування «швидкості» (дельта X на кадр)
VEL_DEADBAND = 0.003           # ігнорувати дуже малі коливання швидкості
MIN_SPEED = 0.012              # мін. швидкість для події (норм. / кадр)
WINDOW = 15                    # скільки останніх кадрів дивимося на «чистий зсув»
MIN_DELTA = 0.12               # мін. чистий зсув за вікно для події
COOLDOWN_FRAMES = 18           # антиспам: після події стільки кадрів ігноруємо нові

# Відображення
DRAW_TRAIL = True
TRAIL_LEN = 25

# ---------- RealSense ----------
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
profile = pipeline.start(cfg)
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

# ---------- Стан ----------
ema_x = None
ema_v = 0.0
xs = deque(maxlen=WINDOW)      # історія згладжених X
cooldown = 0
no_person_ctr = 0
trail = deque(maxlen=TRAIL_LEN)

def emit(event: str):
    # тут зручно замінити на відправку в Unity/мережу
    print(f"EVENT: {event}")

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

        # ---- MediaPipe ----
        rgb_for_mp = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb_for_mp)

        person_present = False
        x_norm = None
        y_norm = None

        if res.pose_landmarks:
            lh = res.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            rh = res.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            x_norm = float((lh.x + rh.x) / 2.0)
            y_norm = float((lh.y + rh.y) / 2.0)

            x_px = int(np.clip(x_norm * W, 0, W - 1))
            y_px = int(np.clip(y_norm * H, 0, H - 1))
            dist_m = depth.get_distance(x_px, y_px)

            if 0.1 < dist_m < MAX_VALID_DISTANCE_M:
                person_present = True
                cv2.circle(color_img, (x_px, y_px), 6, (0, 255, 255), -1)
                cv2.putText(color_img, f"{dist_m:.2f} m", (x_px+8, y_px-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)

        if person_present:
            no_person_ctr = 0

            # ----- згладжений X та швидкість -----
            if ema_x is None:
                ema_x = x_norm
                ema_v = 0.0
                xs.clear()
            else:
                prev_x = ema_x
                ema_x = EMA_ALPHA * x_norm + (1 - EMA_ALPHA) * ema_x
                inst_v = ema_x - prev_x
                ema_v = VEL_ALPHA * inst_v + (1 - VEL_ALPHA) * ema_v

            xs.append(ema_x)

            # хвостик-траєкторія
            if DRAW_TRAIL and y_norm is not None:
                trail.append((int(ema_x * W), int(y_norm * H)))
                for i in range(1, len(trail)):
                    cv2.line(color_img, trail[i-1], trail[i], (0, 255, 0), 2)

            # ----- детекція напрямку -----
            if cooldown > 0:
                cooldown -= 1
            else:
                # чистий зсув за вікно кадрів
                if len(xs) >= 2:
                    delta = xs[-1] - xs[0]  # >0 вправо, <0 вліво
                    speed = ema_v

                    # прибираємо дрібне «тремтіння»
                    if abs(speed) < VEL_DEADBAND:
                        speed = 0.0

                    moved_right = (delta > MIN_DELTA) and (speed > MIN_SPEED)
                    moved_left  = (delta < -MIN_DELTA) and (speed < -MIN_SPEED)

                    if moved_right:
                        emit("MOVE_RIGHT")
                        cooldown = COOLDOWN_FRAMES
                        xs.clear()   # починаємо заново вікно
                    elif moved_left:
                        emit("MOVE_LEFT")
                        cooldown = COOLDOWN_FRAMES
                        xs.clear()

            cv2.putText(color_img, f"x={ema_x:.3f} v={ema_v:+.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2, cv2.LINE_AA)
        else:
            no_person_ctr += 1
            if no_person_ctr == ABSENCE_FRAMES:
                emit("NO_PERSON")
                ema_x = None
                ema_v = 0.0
                xs.clear()
                trail.clear()

            cv2.putText(color_img, "NO PERSON",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        # ---- Відображення ----
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=0.03),
            cv2.COLORMAP_JET
        )
        view = np.hstack([color_img, depth_vis])
        cv2.imshow("Direction only (color | depth)", view)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()