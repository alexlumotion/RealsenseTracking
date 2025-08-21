# tpose_depth_debug_overlay.py
# RealSense (color+depth) + MediaPipe Pose
# Дзеркальний показ, оверлей кісток і числових дельт, стабільна T‑поза (зап’ястя+лікті)

import cv2, numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import time

# ------------------ Параметри стріму ------------------
W, H, FPS = 640, 480, 30
MAX_VALID_DISTANCE_M = 4.0         # гранична відстань (за бажанням)
USE_CLAHE = False                   # якщо темно/контраст низький → True

# ------------------ Пороги T‑позі ---------------------
TPOSE_MIN_EXTEND   = 0.25          # мінімальне розведення рук по X (нормоване)
TPOSE_Y_TOL        = 0.12          # допуск по Y для зап’ясть
TPOSE_ELBOW_Y_TOL  = 0.15          # ДОПУСК по Y для ЛІКТІВ (НОВЕ)
TPOSE_MIN_FRAMES   = 4             # скільки послідовних кадрів підтверджуємо
tpose_streak       = 0

# ------------------ MediaPipe -------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------ RealSense init --------------------
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
profile = pipeline.start(cfg)
align = rs.align(rs.stream.color)

def draw_landmark_point(img, lm, color, radius=5):
    x = int(lm.x * W); y = int(lm.y * H)
    cv2.circle(img, (x, y), radius, color, -1)

def draw_skeleton(img, lms):
    # кілька простих ребер для візуала
    def pt(i): return (int(lms[i].x*W), int(lms[i].y*H))
    Y = mp_pose.PoseLandmark
    pairs = [
        (Y.LEFT_SHOULDER, Y.RIGHT_SHOULDER),
        (Y.LEFT_SHOULDER, Y.LEFT_ELBOW),   (Y.LEFT_ELBOW, Y.LEFT_WRIST),
        (Y.RIGHT_SHOULDER, Y.RIGHT_ELBOW), (Y.RIGHT_ELBOW, Y.RIGHT_WRIST),
        (Y.LEFT_SHOULDER, Y.LEFT_HIP),     (Y.RIGHT_SHOULDER, Y.RIGHT_HIP),
        (Y.LEFT_HIP, Y.RIGHT_HIP)
    ]
    for a, b in pairs:
        cv2.line(img, pt(a), pt(b), (0, 255, 255), 2)

def put_text(img, text, org, color=(255,255,255), scale=0.7, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

try:
    print("[INFO] Running... Esc=exit. Hotkeys: u/i tolY ±, j/k wristsY ±, n/m elbowsY ±")
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            continue

        color_img = np.asanyarray(color.get_data())
        depth_img  = np.asanyarray(depth.get_data())

        # Дзеркальний фліп (як у дзеркалі)
        color_img = cv2.flip(color_img, 1)
        depth_img = cv2.flip(depth_img, 1)

        # Опційно трохи підняти контраст
        src_tag = "RGB"
        if USE_CLAHE:
            lab = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l2 = clahe.apply(l)
            lab2 = cv2.merge([l2, a, b])
            color_img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
            src_tag = "CLAHE"

        # MediaPipe → RGB
        res = pose.process(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))

        # Візуал глибини
        depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)

        is_tpose = False
        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            # Основні точки
            Y = mp_pose.PoseLandmark
            ls = lms[Y.LEFT_SHOULDER];  rs_ = lms[Y.RIGHT_SHOULDER]
            le = lms[Y.LEFT_ELBOW];     re = lms[Y.RIGHT_ELBOW]
            lw = lms[Y.LEFT_WRIST];     rw = lms[Y.RIGHT_WRIST]

            # Малюємо скелет і точки
            draw_skeleton(color_img, lms)
            for lm in (ls, rs_, le, re, lw, rw):
                draw_landmark_point(color_img, lm, (0, 255, 0), 5)

            # Середній Y плечей
            shoulder_y = 0.5 * (ls.y + rs_.y)

            # Діагностика по зап’ястям/ліктям (дельти по Y від плечей)
            dwL = lw.y - shoulder_y; dwR = rw.y - shoulder_y
            deL = le.y - shoulder_y; deR = re.y - shoulder_y

            # Візуальні підписи біля точок
            put_text(color_img, f"dwL={dwL:+.3f}", (int(lw.x*W)-60, int(lw.y*H)-10), (0,255,0), 0.5, 2)
            put_text(color_img, f"dwR={dwR:+.3f}", (int(rw.x*W)+10, int(rw.y*H)-10), (0,255,0), 0.5, 2)
            put_text(color_img, f"deL={deL:+.3f}", (int(le.x*W)-60, int(le.y*H)-10), (0,200,255), 0.5, 2)
            put_text(color_img, f"deR={deR:+.3f}", (int(re.x*W)+10, int(re.y*H)-10), (0,200,255), 0.5, 2)

            # Умови T‑позі
            wrists_y_ok = (abs(dwL) <= TPOSE_Y_TOL) and (abs(dwR) <= TPOSE_Y_TOL)
            elbows_y_ok = (abs(deL) <= TPOSE_ELBOW_Y_TOL) and (abs(deR) <= TPOSE_ELBOW_Y_TOL)

            # Розведення рук по X: ліве зап’ястя лівіше від лівого плеча, праве — правіше правого плеча
            left_extend  = max(0.0, ls.x - lw.x)       # якщо зап’ястя лівіше плеча → позитивне
            right_extend = max(0.0, rw.x - rs_.x)      # якщо зап’ястя правіше плеча → позитивне
            total_extend = left_extend + right_extend

            # Вивід розведення
            put_text(color_img, f"extend={total_extend:.2f}", (20, 56), (50,220,50), 0.7, 2)

            # Перевірка кадру
            is_tpose_frame = wrists_y_ok and elbows_y_ok and (total_extend >= TPOSE_MIN_EXTEND)

            # Накопичуємо кадри для стабільності
            if is_tpose_frame:
                tpose_streak += 1
            else:
                tpose_streak = 0

            is_tpose = tpose_streak >= TPOSE_MIN_FRAMES

        # Заголовок/статус
        hdr = f"TPOSE  tolY={TPOSE_Y_TOL:.2f}  elbows={TPOSE_ELBOW_Y_TOL:.2f}  extend>={TPOSE_MIN_EXTEND:.2f}  need={TPOSE_MIN_FRAMES}  src={src_tag}"
        put_text(color_img, hdr, (18, 28), (0,255,255), 0.8, 2)
        put_text(color_img, "TPOSE ✅" if is_tpose else "TPOSE —",
                 (18, 86), (0,255,180) if is_tpose else (200,200,200), 0.8, 2)

        view = np.hstack([color_img, depth_vis])
        cv2.imshow("Pose view (left) | Depth (right)", view)

        # --- хоткеї для тюнінгу ---
        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # Esc
            break
        elif key == ord('u'):  TPOSE_MIN_EXTEND = max(0.05, TPOSE_MIN_EXTEND - 0.02)
        elif key == ord('i'):  TPOSE_MIN_EXTEND += 0.02
        elif key == ord('j'):  TPOSE_Y_TOL      = max(0.02, TPOSE_Y_TOL - 0.01)
        elif key == ord('k'):  TPOSE_Y_TOL      += 0.01
        elif key == ord('n'):  TPOSE_ELBOW_Y_TOL = max(0.02, TPOSE_ELBOW_Y_TOL - 0.01)
        elif key == ord('m'):  TPOSE_ELBOW_Y_TOL += 0.01

finally:
    pipeline.stop()
    cv2.destroyAllWindows()