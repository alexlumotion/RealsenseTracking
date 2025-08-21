# tpose_depth_calibrated.py
# RealSense + MediaPipe Pose | Дзеркальний перегляд, детекція Т‑пози з калібровкою

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
from collections import deque

# ---------- Відеопотік ----------
W, H, FPS = 640, 480, 30

# ---------- Пороги детекції за замовчуванням (якщо без калібровки) ----------
DEFAULT_EXTEND_THR = 0.25     # мін. розліт рук (wristR.x - wristL.x), 0..1
DEFAULT_Y_THR      = 0.15     # макс. |y(wrist) - y(лінії плечей)|, 0..1

# ---------- Калібровка ----------
CAL_FRAMES = 20               # скільки стабільних кадрів зібрати в калібровці
EXT_TOL = 0.07                # допуск по розльоту рук від еталону
Y_TOL   = 0.06                # допуск по вертикалі від еталону

# Стан машини
STATE_IDLE = "IDLE"
STATE_CAL  = "CALIBRATING"
STATE_READY= "READY"

state = STATE_IDLE
cal_queue_extend = deque(maxlen=CAL_FRAMES)
cal_queue_dwL    = deque(maxlen=CAL_FRAMES)
cal_queue_dwR    = deque(maxlen=CAL_FRAMES)

extend_ref = None
dwL_ref = None
dwR_ref = None

# ---------- Ініціалізація RealSense ----------
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
profile = pipeline.start(cfg)
align = rs.align(rs.stream.color)

# ---------- Ініціалізація MediaPipe ----------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def norm_to_px(x, y):
    return int(np.clip(x * W, 0, W-1)), int(np.clip(y * H, 0, H-1))

def put_text(img, text, org, color=(0,255,255), scale=0.7, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_point(img, pt, color, r=6):
    cv2.circle(img, pt, r, color, -1, cv2.LINE_AA)

def draw_bone(img, p1, p2, color, w=3):
    cv2.line(img, p1, p2, color, w, cv2.LINE_AA)

def stable_enough(values, eps=0.02):
    """Оцінка стабільності (розкид у черзі невеликий)."""
    if len(values) < values.maxlen:
        return False
    v = np.array(values)
    return (v.max() - v.min()) < eps

try:
    print("[INFO] Running...  Esc=exit | C=start calibration | R=reset | U/J ±EXT tol | I/K ±Y tol")
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            continue

        color_img = np.asanyarray(color.get_data())
        depth_img  = np.asanyarray(depth.get_data())

        # Дзеркало
        color_img = cv2.flip(color_img, 1)
        depth_img = cv2.flip(depth_img, 1)

        # Pose
        res = pose.process(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))

        is_tpose = False
        extend_val = 0.0
        dwL = dwR = 0.0

        have_pose = res.pose_landmarks is not None
        if have_pose:
            lm = res.pose_landmarks.landmark

            l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_el = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
            r_el = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
            l_wr = lm[mp_pose.PoseLandmark.LEFT_WRIST]
            r_wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

            l_sh_xy = (float(l_sh.x), float(l_sh.y))
            r_sh_xy = (float(r_sh.x), float(r_sh.y))
            l_el_xy = (float(l_el.x), float(l_el.y))
            r_el_xy = (float(r_el.x), float(r_el.y))
            l_wr_xy = (float(l_wr.x), float(l_wr.y))
            r_wr_xy = (float(r_wr.x), float(r_wr.y))

            LSH = norm_to_px(*l_sh_xy); RSH = norm_to_px(*r_sh_xy)
            LEL = norm_to_px(*l_el_xy); REL = norm_to_px(*r_el_xy)
            LWR = norm_to_px(*l_wr_xy); RWR = norm_to_px(*r_wr_xy)

            # Лінія плечей (y)
            y_sh = (l_sh_xy[1] + r_sh_xy[1]) / 2.0
            dwL = l_wr_xy[1] - y_sh
            dwR = r_wr_xy[1] - y_sh

            # Розліт рук
            extend_val = r_wr_xy[0] - l_wr_xy[0]

            # Малювання
            draw_bone(color_img, LSH, LEL, (0,255,255), 3)
            draw_bone(color_img, LEL, LWR, (0,255,255), 3)
            draw_bone(color_img, RSH, REL, (0,255,255), 3)
            draw_bone(color_img, REL, RWR, (0,255,255), 3)
            draw_bone(color_img, LSH, RSH, (0,255,255), 2)
            for p in (LSH, RSH, LEL, REL, LWR, RWR):
                draw_point(color_img, p, (0,200,0), 6)

            put_text(color_img, f"extend={extend_val:+.3f}", (20, 130), (255,255,0))
            put_text(color_img, f"dwL={dwL:+.3f}  dwR={dwR:+.3f}", (20, 155), (255,255,0))

            # --------- ЛОГІКА СТАНІВ ---------
            if state == STATE_CAL:
                # Збираємо стабільні кадри
                cal_queue_extend.append(extend_val)
                cal_queue_dwL.append(dwL)
                cal_queue_dwR.append(dwR)

                progress = int(100 * len(cal_queue_extend)/CAL_FRAMES)
                put_text(color_img, f"CALIBRATING... {progress}%", (20, 80), (0,255,255), 0.9, 2)

                if stable_enough(cal_queue_extend, eps=0.02) and \
                   stable_enough(cal_queue_dwL,    eps=0.02) and \
                   stable_enough(cal_queue_dwR,    eps=0.02) and \
                   len(cal_queue_extend) == CAL_FRAMES:
                    # фіксуємо еталон як медіану
                    extend_ref = float(np.median(cal_queue_extend))
                    dwL_ref    = float(np.median(cal_queue_dwL))
                    dwR_ref    = float(np.median(cal_queue_dwR))
                    state = STATE_READY
                    cal_queue_extend.clear(); cal_queue_dwL.clear(); cal_queue_dwR.clear()
                    print(f"[CALIBRATED] extend_ref={extend_ref:.3f}, dwL_ref={dwL_ref:.3f}, dwR_ref={dwR_ref:.3f}")

            # Детекція
            if state == STATE_READY and extend_ref is not None:
                if (abs(extend_val - extend_ref) <= EXT_TOL and
                    abs(dwL - dwL_ref)         <= Y_TOL and
                    abs(dwR - dwR_ref)         <= Y_TOL):
                    is_tpose = True
            elif state in (STATE_IDLE, STATE_CAL):
                # Фолбек без калібровки — базові пороги
                if extend_val > DEFAULT_EXTEND_THR and abs(dwL) < DEFAULT_Y_THR and abs(dwR) < DEFAULT_Y_THR:
                    is_tpose = True

        # ---------- Оверлеї ----------
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET
        )
        view = np.hstack([color_img, depth_vis])

        # Шапка
        hdr = f"STATE={state}   tolEXT={EXT_TOL:.2f}  tolY={Y_TOL:.2f}"
        if extend_ref is not None:
            hdr += f"   ref: ext={extend_ref:.2f}  dwL={dwL_ref:.2f}  dwR={dwR_ref:.2f}"
        put_text(view, hdr, (18, 28), (0,255,255), 0.9, 2)

        put_text(
            view,
            "T-POSE  \u2705" if is_tpose else "T-POSE  \u274C",
            (18, 68),
            (0,255,0) if is_tpose else (0,0,255),
            1.2, 3
        )

        # Підказки
        put_text(view, "C=calibrate  R=reset  U/J=EXT tol  I/K=Y tol  Esc=exit", (18, H-16), (200,200,200), 0.6, 1)

        cv2.imshow("T-pose calibrated | Left: RGB (mirror)  Right: Depth", view)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc
            break
        elif key == ord('c'):
            print("[STATE] -> CALIBRATING")
            state = STATE_CAL
            cal_queue_extend.clear(); cal_queue_dwL.clear(); cal_queue_dwR.clear()
        elif key == ord('r'):
            print("[STATE] -> IDLE (reset refs)")
            state = STATE_IDLE
            extend_ref = dwL_ref = dwR_ref = None
            cal_queue_extend.clear(); cal_queue_dwL.clear(); cal_queue_dwR.clear()
        elif key == ord('u'):
            EXT_TOL = min(0.5, EXT_TOL + 0.01)
        elif key == ord('j'):
            EXT_TOL = max(0.0, EXT_TOL - 0.01)
        elif key == ord('i'):
            Y_TOL = min(0.5, Y_TOL + 0.01)
        elif key == ord('k'):
            Y_TOL = max(0.0, Y_TOL - 0.01)

finally:
    try: pipeline.stop()
    except: pass
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")