# tpose_depth_fixed_refs.py
# RealSense + MediaPipe Pose | Дзеркальне відображення, оверлеї, Т‑поза за зафіксованими еталонами.

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp

# --------------------- Камера ---------------------
W, H, FPS = 640, 480, 30

# --------------------- Еталони (з твоєї калібровки) ---------------------
EXTEND_REF = -0.499   # зверни увагу: беремо МОДУЛЬ при порівнянні (див. нижче)
DWL_REF    =  0.070
DWR_REF    =  0.016

# --------------------- Допуски/стабілізація ---------------------
EXT_TOL        = 0.08   # допуск по розльоту рук (| |extend| - |EXTEND_REF| | <= EXT_TOL)
Y_TOL          = 0.06   # допуск по вертикалі зап’ясть відносно еталонів
MIN_FRAMES_OK  = 4      # скільки кадрів поспіль потрібно, щоб зарахувати Т‑позу
DECAY_ON_FAIL  = True   # плавне падіння лічильника при провалі

# --------------------- Ініціалізація ---------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
profile = pipeline.start(cfg)
align = rs.align(rs.stream.color)

def norm_to_px(x, y):
    return int(np.clip(x * W, 0, W-1)), int(np.clip(y * H, 0, H-1))

def put(img, text, org, color=(0,255,255), scale=0.7, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def dot(img, pt, color, r=6):
    cv2.circle(img, pt, r, color, -1, cv2.LINE_AA)

def bone(img, p1, p2, color, w=3):
    cv2.line(img, p1, p2, color, w, cv2.LINE_AA)

streak = 0
last_state = None

try:
    print("[INFO] Running... Esc=exit | U/J ±EXT_TOL | I/K ±Y_TOL | Z/X ±MIN_FRAMES")
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

        # Pose
        res = pose.process(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))

        is_tpose_frame = False
        extend_val = 0.0
        dwL = dwR = 0.0

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            LSH = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            RSH = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            LEL = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
            REL = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
            LWR = lm[mp_pose.PoseLandmark.LEFT_WRIST]
            RWR = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

            lsh = (float(LSH.x), float(LSH.y))
            rsh = (float(RSH.x), float(RSH.y))
            lel = (float(LEL.x), float(LEL.y))
            rel = (float(REL.x), float(REL.y))
            lwr = (float(LWR.x), float(LWR.y))
            rwr = (float(RWR.x), float(RWR.y))

            # Пікселі для оверлею
            LSHp, RSHp = norm_to_px(*lsh), norm_to_px(*rsh)
            LELp, RELp = norm_to_px(*lel), norm_to_px(*rel)
            LWRp, RWRp = norm_to_px(*lwr), norm_to_px(*rwr)

            # Скелет
            bone(color_img, LSHp, LELp, (0,255,255), 3)
            bone(color_img, LELp, LWRp, (0,255,255), 3)
            bone(color_img, RSHp, RELp, (0,255,255), 3)
            bone(color_img, RELp, RWRp, (0,255,255), 3)
            bone(color_img, LSHp, RSHp, (0,255,255), 2)
            for p in (LSHp, RSHp, LELp, RELp, LWRp, RWRp):
                dot(color_img, p, (0,200,0), 6)

            # Лінія плечей (y)
            y_sh = (lsh[1] + rsh[1]) / 2.0

            # Вертикальні відхилення зап’ясть від лінії плечей
            dwL = lwr[1] - y_sh
            dwR = rwr[1] - y_sh

            # Розліт рук по X (нормалізований)
            extend_val = rwr[0] - lwr[0]

            # Порівнюємо МОДУЛІ розльоту (щоб не мучитися зі знаком при різних дзеркалах)
            ext_ok = abs(abs(extend_val) - abs(EXTEND_REF)) <= EXT_TOL
            y_ok   = (abs(dwL - DWL_REF) <= Y_TOL) and (abs(dwR - DWR_REF) <= Y_TOL)

            is_tpose_frame = ext_ok and y_ok

        # Антифлікер
        if is_tpose_frame:
            streak += 1
        else:
            streak = max(0, streak-1) if DECAY_ON_FAIL else 0

        is_tpose = streak >= MIN_FRAMES_OK
        state = "T-POSE" if is_tpose else ("MAYBE" if is_tpose_frame else "NO")

        if state != last_state:
            print(f"[STATE] {state}  streak={streak}/{MIN_FRAMES_OK}  "
                  f"ext={extend_val:+.3f} (ref {EXTEND_REF:+.3f} ±{EXT_TOL:.2f})  "
                  f"dwL={dwL:+.3f} (ref {DWL_REF:+.3f} ±{Y_TOL:.2f})  "
                  f"dwR={dwR:+.3f} (ref {DWR_REF:+.3f} ±{Y_TOL:.2f})")
            last_state = state

        # Оверлей
        put(color_img, f"REF: |ext|={abs(EXTEND_REF):.3f}  dwL={DWL_REF:+.3f}  dwR={DWR_REF:+.3f}",
            (18, 28), (0,255,255), 0.8, 2)
        put(color_img, f"NOW:  |ext|={abs(extend_val):.3f}  dwL={dwL:+.3f}  dwR={dwR:+.3f}",
            (18, 56), (255,255,0), 0.8, 2)
        put(color_img, f"TOL:  ext±{EXT_TOL:.2f}  y±{Y_TOL:.2f}   streak={streak}/{MIN_FRAMES_OK}",
            (18, 84), (200,220,200), 0.75, 2)

        put(color_img, "T-POSE  \u2705" if is_tpose else ("T-POSE  …" if is_tpose_frame else "T-POSE  \u274C"),
            (18, 114), (0,255,0) if is_tpose else ((0,200,255) if is_tpose_frame else (0,0,255)), 1.1, 3)

        # Depth heatmap
        depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
        view = np.hstack([color_img, depth_vis])
        put(view, "U/J: EXT tol  |  I/K: Y tol  |  Z/X: frames  |  Esc: exit", (18, H-16), (220,220,220), 0.6, 1)

        cv2.imshow("T-pose | Left: RGB (mirror)  Right: Depth", view)

        # Хоткеї
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc
            break
        elif key == ord('u'):
            EXT_TOL = min(0.5, EXT_TOL + 0.01)
        elif key == ord('j'):
            EXT_TOL = max(0.0, EXT_TOL - 0.01)
        elif key == ord('i'):
            Y_TOL = min(0.5, Y_TOL + 0.01)
        elif key == ord('k'):
            Y_TOL = max(0.0, Y_TOL - 0.01)
        elif key == ord('z'):
            MIN_FRAMES_OK = max(1, MIN_FRAMES_OK - 1)
        elif key == ord('x'):
            MIN_FRAMES_OK += 1

finally:
    try: pipeline.stop()
    except: pass
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")