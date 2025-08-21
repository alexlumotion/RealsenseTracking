# tpose_depth_debug_overlay.py
# РеальнаSense D455/D435 + MediaPipe Pose
# Дзеркальне відображення, скелет, метрики та детекція T‑pose (проста умова).
# Hotkeys: q/w — поріг EXTEND, a/s — поріг по Y; Esc — вихід

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp

# ---------- ПАРАМЕТРИ ----------
W, H, FPS = 640, 480, 30

# Пороги для Т‑пози (можна змінювати гарячими клавішами)
TPOSE_EXTEND_THR = 0.25   # мінімальний поперечний розліт рук (нормалізований: wristR.x - wristL.x)
TPOSE_Y_THR      = 0.15   # наскільки зап’ястя можуть відхилятись по Y від лінії плечей

# Відмалювати підказки/координати
DRAW_NUMBERS = True

# ---------- INIT RealSense ----------
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
profile = pipeline.start(cfg)
align = rs.align(rs.stream.color)

# ---------- INIT MediaPipe ----------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def norm_to_px(x, y):
    """Перетворити нормалізовані координати MediaPipe (0..1) у пікселі."""
    return int(np.clip(x * W, 0, W - 1)), int(np.clip(y * H, 0, H - 1))

def draw_point(img, x, y, color, r=6):
    cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)

def put_text(img, text, org, color=(0, 255, 255), scale=0.6, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

try:
    print("[INFO] Running...  Esc = exit.  Hotkeys: q/w EXTEND, a/s Y")
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            continue

        color_img = np.asanyarray(color.get_data())
        depth_img  = np.asanyarray(depth.get_data())

        # Дзеркало для обох потоків
        color_img = cv2.flip(color_img, 1)
        depth_img = cv2.flip(depth_img, 1)

        # ---- Pose (на RGB) ----
        rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        is_tpose = False
        extend_val = 0.0
        dwL = dwR = 0.0

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            # Важливі точки
            l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_wr = lm[mp_pose.PoseLandmark.LEFT_WRIST]
            r_wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            l_el = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
            r_el = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]

            # Нормалізовані координати (0..1)
            l_sh_xy = (float(l_sh.x), float(l_sh.y))
            r_sh_xy = (float(r_sh.x), float(r_sh.y))
            l_wr_xy = (float(l_wr.x), float(l_wr.y))
            r_wr_xy = (float(r_wr.x), float(r_wr.y))
            l_el_xy = (float(l_el.x), float(l_el.y))
            r_el_xy = (float(r_el.x), float(r_el.y))

            # В пікселі для малювання
            LSH = norm_to_px(*l_sh_xy)
            RSH = norm_to_px(*r_sh_xy)
            LWR = norm_to_px(*l_wr_xy)
            RWR = norm_to_px(*r_wr_xy)
            LEL = norm_to_px(*l_el_xy)
            REL = norm_to_px(*r_el_xy)

            # Середня лінія плечей (y)
            shoulder_y = (l_sh_xy[1] + r_sh_xy[1]) / 2.0

            # Наскільки зап’ястя “вище/нижче” плечей (плюс = нижче, мінус = вище у координатах з верхнім нулем)
            dwL = l_wr_xy[1] - shoulder_y
            dwR = r_wr_xy[1] - shoulder_y

            # Розліт рук: відстань між зап’ястями по X (нормалізовано до ширини кадру)
            extend_val = r_wr_xy[0] - l_wr_xy[0]

            # --- Проста умова T‑pose ---
            if extend_val > TPOSE_EXTEND_THR and abs(dwL) < TPOSE_Y_THR and abs(dwR) < TPOSE_Y_THR:
                is_tpose = True

            # Відмалювати кістки рук
            cv2.line(color_img, LSH, LEL, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.line(color_img, LEL, LWR, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.line(color_img, RSH, REL, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.line(color_img, REL, RWR, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.line(color_img, LSH, RSH, (0, 255, 255), 2, cv2.LINE_AA)

            # Точки
            for p in (LSH, RSH, LEL, REL, LWR, RWR):
                draw_point(color_img, p[0], p[1], (0, 200, 0), r=6)

            # Числа dw/dx біля зап’ясть
            if DRAW_NUMBERS:
                put_text(color_img, f"dwL={dwL:+.3f}", (LWR[0]-60, LWR[1]-10), (0,255,0))
                put_text(color_img, f"dwR={dwR:+.3f}", (RWR[0]-60, RWR[1]-10), (0,255,0))

        # Заголовок з порогами
        header = f"TPOSE  toY={TPOSE_Y_THR:.2f}   extend>={TPOSE_EXTEND_THR:.2f}"
        put_text(color_img, header, (18, 28), (0,255,255), 0.9, 2)

        # Показник T‑pose
        put_text(
            color_img,
            "TPOSE  \u2705" if is_tpose else "TPOSE  \u274C",
            (18, 68),
            (0,255,0) if is_tpose else (0,0,255),
            1.2, 3
        )

        # Показати поточні значення extend/dw
        put_text(color_img, f"extend={extend_val:+.2f}   dwL={dwL:+.3f}  dwR={dwR:+.3f}",
                 (18, 100), (255,255,0), 0.7, 2)

        # Праворуч — depth heatmap
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET
        )
        view = np.hstack([color_img, depth_vis])
        cv2.imshow("Pose view (left) | Depth (right)", view)

        # --- Hotkeys ---
        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # Esc
            break
        elif key == ord('q'):
            TPOSE_EXTEND_THR = max(0.05, TPOSE_EXTEND_THR - 0.01)
        elif key == ord('w'):
            TPOSE_EXTEND_THR = min(0.9, TPOSE_EXTEND_THR + 0.01)
        elif key == ord('a'):
            TPOSE_Y_THR = max(0.02, TPOSE_Y_THR - 0.01)
        elif key == ord('s'):
            TPOSE_Y_THR = min(0.5, TPOSE_Y_THR + 0.01)

finally:
    try: pipeline.stop()
    except: pass
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")