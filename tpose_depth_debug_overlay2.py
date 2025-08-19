# tpose_depth_debug_overlay.py
# Реальний час: RealSense (D4xx), MediaPipe Pose, детекція Т‑пози (руки горизонтально)
# Відзеркалено по горизонталі (як дзеркало). Є оверлеї з кутами/дельтами та гарячі клавіші.

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import math
import time

# -------------------- ПАРАМЕТРИ --------------------
W, H, FPS = 640, 480, 30

# Пороги Т‑пози (крути під свою сцену/відстань/зріст)
TPOSE_Y_TOL        = 0.06   # допустима різниця y між зап’ястям і рівнем плечей (у частках висоти кадру)
TPOSE_MIN_EXTEND   = 0.80   # зап’ястя має бути далі від центру, ніж лікоть (коеф. від половини міжплечової)
TPOSE_MAX_ANGLE_DEG= 15.0   # рука ≈ горизонтальна: кут (плече->зап’ястя) до горизонталі <= цього значення
TPOSE_MIN_FRAMES   = 8      # скільки кадрів поспіль повинна триматися Т‑поза
VIS_THRESH         = 0.5    # мінімальна visibility для ключових точок

# Сегментація (лише для оверлея; не впливає на детекцію)
DRAW_LANDMARKS     = True
DRAW_NUMBERS       = True

# -------------------- INIT --------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,              # 2 точніше (повільніше), 1/0 швидше
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
profile = pipeline.start(cfg)

align_to_color = rs.align(rs.stream.color)

# -------------------- УТИЛІТИ --------------------
def ang_deg(vx, vy):
    # кут до горизонталі (0° = горизонт), беремо |кут|
    return abs(math.degrees(math.atan2(vy, vx)))

def draw_tag(img, text, xy, color=(0,255,255)):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def draw_dot(img, x, y, color=(0,255,255), r=5):
    cv2.circle(img, (int(x), int(y)), r, color, -1)

def draw_line(img, p1, p2, color=(0,200,0), t=2):
    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, t, cv2.LINE_AA)

# Головна перевірка Т‑пози
def is_t_pose(lm):
    """Перевіряємо Т‑позу:
       - зап’ястя ≈ на висоті плечей (TPOSE_Y_TOL)
       - зап’ястя далі від центру, ніж лікоть (extend) на деякий мін. запас
       - рука ≈ горизонтальна: кут (плече->зап’ястя) <= TPOSE_MAX_ANGLE_DEG
       - ключові точки мають достатню visibility
    """
    LSH = mp_pose.PoseLandmark.LEFT_SHOULDER
    RSH = mp_pose.PoseLandmark.RIGHT_SHOULDER
    LE  = mp_pose.PoseLandmark.LEFT_ELBOW
    RE  = mp_pose.PoseLandmark.RIGHT_ELBOW
    LW  = mp_pose.PoseLandmark.LEFT_WRIST
    RW  = mp_pose.PoseLandmark.RIGHT_WRIST

    ls, rs_, le, re, lw, rw = lm[LSH], lm[RSH], lm[LE], lm[RE], lm[LW], lm[RW]

    # 0) Перевірка visibility
    keypts = [ls, rs_, le, re, lw, rw]
    if any(getattr(p, "visibility", 1.0) < VIS_THRESH for p in keypts):
        return False, {"reason":"low_visibility"}, (ls, rs_, le, re, lw, rw)

    # 1) Центр і ширина плечей
    c_x = (ls.x + rs_.x) / 2.0
    c_y = (ls.y + rs_.y) / 2.0
    shoulder_w = abs(ls.x - rs_.x) + 1e-6
    min_off = (shoulder_w * 0.5) * TPOSE_MIN_EXTEND

    # 2) Висота зап’ясть ≈ рівню плечей
    dyL = lw.y - c_y
    dyR = rw.y - c_y
    y_ok_L = abs(dyL) <= TPOSE_Y_TOL
    y_ok_R = abs(dyR) <= TPOSE_Y_TOL

    # 3) Зап’ястя далі від центру, ніж лікоть (і достатньо далеко)
    dxL_w = c_x - lw.x
    dxL_e = c_x - le.x
    dxR_w = rw.x - c_x
    dxR_e = re.x - c_x
    extend_ok_L = (dxL_w >= max(min_off, dxL_e + 0.02))
    extend_ok_R = (dxR_w >= max(min_off, dxR_e + 0.02))

    # 4) Кути до горизонталі (плече->зап’ястя)
    angL = ang_deg(lw.x - ls.x, lw.y - ls.y)
    angR = ang_deg(rw.x - rs_.x, rw.y - rs_.y)
    ang_ok_L = angL <= TPOSE_MAX_ANGLE_DEG
    ang_ok_R = angR <= TPOSE_MAX_ANGLE_DEG

    ok = y_ok_L and y_ok_R and extend_ok_L and extend_ok_R and ang_ok_L and ang_ok_R
    dbg = {
        "shoulder_w": round(shoulder_w,3),
        "min_off": round(min_off,3),
        "dyL": round(dyL,3), "dyR": round(dyR,3),
        "dxL_w": round(dxL_w,3), "dxL_e": round(dxL_e,3),
        "dxR_w": round(dxR_w,3), "dxR_e": round(dxR_e,3),
        "angL": round(angL,1), "angR": round(angR,1),
        "okY": (y_ok_L, y_ok_R), "okExtend": (extend_ok_L, extend_ok_R),
        "okAng": (ang_ok_L, ang_ok_R)
    }
    return ok, dbg, (ls, rs_, le, re, lw, rw)

# -------------------- ГОЛОВНИЙ ЦИКЛ --------------------
tpose_frames = 0
last_state = "NO_PERSON"

print("[INFO] Running...  Esc=exit. Hotkeys: q/w y_tol, a/s extend, e/r angle, z/x frames")
try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align_to_color.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            continue

        # Отримуємо numpy
        color_img = np.asanyarray(color.get_data())
        depth_img = np.asanyarray(depth.get_data())

        # Дзеркало
        color_img = cv2.flip(color_img, 1)
        depth_img = cv2.flip(depth_img, 1)

        # MediaPipe
        rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        person = False
        tpose_ok = False
        dbg = {}
        pts = None

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            # Перевірка Т‑пози
            tpose_ok, dbg, pts = is_t_pose(lm)

            # Перевірка, що бодай щось валідне по глибині по центру плечей
            LSH = mp_pose.PoseLandmark.LEFT_SHOULDER
            RSH = mp_pose.PoseLandmark.RIGHT_SHOULDER
            c_x = int(((lm[LSH].x + lm[RSH].x)/2.0) * W)
            c_y = int(((lm[LSH].y + lm[RSH].y)/2.0) * H)
            dist = depth.get_distance(np.clip(c_x,0,W-1), np.clip(c_y,0,H-1))
            person = (dist > 0.1 and dist < 4.5)

            # Оверлей ключових точок і чисел
            if DRAW_LANDMARKS and pts is not None:
                ls, rs_, le, re, lw, rw = pts
                px = lambda p: (int(p.x*W), int(p.y*H))
                # точки
                for p, name, col in [(ls,"LS",(0,255,255)), (rs_,"RS",(0,255,255)),
                                     (le,"LE",(0,200,255)), (re,"RE",(0,200,255)),
                                     (lw,"LW",(0,150,255)), (rw,"RW",(0,150,255))]:
                    x,y = px(p)
                    draw_dot(color_img, x, y, col, 6)
                    if DRAW_NUMBERS:
                        draw_tag(color_img, name, (x+6,y-6), col)
                # сегменти
                draw_line(color_img, px(ls), px(le), (0,200,0), 3)
                draw_line(color_img, px(le), px(lw), (0,200,0), 3)
                draw_line(color_img, px(rs_), px(re), (0,200,0), 3)
                draw_line(color_img, px(re), px(rw), (0,200,0), 3)

                # текст з кутами/дельтами
                draw_tag(color_img, f"angL={dbg.get('angL',0)}  angR={dbg.get('angR',0)}", (10, 28))
                draw_tag(color_img, f"dyL={dbg.get('dyL',0)} dyR={dbg.get('dyR',0)}", (10, 52))
                draw_tag(color_img, f"dxLw={dbg.get('dxL_w',0)} dxLe={dbg.get('dxL_e',0)}", (10, 76))
                draw_tag(color_img, f"dxRw={dbg.get('dxR_w',0)} dxRe={dbg.get('dxR_e',0)}", (10,100))

        # Стан/рахівник кадрів
        if person and tpose_ok:
            tpose_frames += 1
        else:
            tpose_frames = max(0, tpose_frames-1)  # м'який спад

        state = "T_POSE" if tpose_frames >= TPOSE_MIN_FRAMES else ("PERSON" if person else "NO_PERSON")
        if state != last_state:
            print(f"[EVENT] {state}  frames={tpose_frames}  dbg={dbg}")
            last_state = state

        # Оверлей стану та порогів
        draw_tag(color_img, f"STATE: {state}  frames={tpose_frames}/{TPOSE_MIN_FRAMES}", (10, H-60),
                 (0,200,255) if state=="T_POSE" else ((50,220,50) if state=="PERSON" else (60,60,220)))
        draw_tag(color_img, f"TOLy={TPOSE_Y_TOL:.3f}  EXT={TPOSE_MIN_EXTEND:.2f}  ANG<={TPOSE_MAX_ANGLE_DEG:.1f}  VIS>={VIS_THRESH:.2f}",
                 (10, H-30), (200,200,60))

        # Вікно depth для наочності
        depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
        view = np.hstack([color_img, depth_vis])
        cv2.imshow("T-POSE | RealSense + MediaPipe (mirror)", view)

        # Клавіші тюнінгу
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc
            break
        elif key == ord('q'):
            TPOSE_Y_TOL = max(0.0, TPOSE_Y_TOL - 0.005)
        elif key == ord('w'):
            TPOSE_Y_TOL += 0.005
        elif key == ord('a'):
            TPOSE_MIN_EXTEND = max(0.2, TPOSE_MIN_EXTEND - 0.02)
        elif key == ord('s'):
            TPOSE_MIN_EXTEND = min(1.2, TPOSE_MIN_EXTEND + 0.02)
        elif key == ord('e'):
            TPOSE_MAX_ANGLE_DEG = min(45.0, TPOSE_MAX_ANGLE_DEG + 1.0)
        elif key == ord('r'):
            TPOSE_MAX_ANGLE_DEG = max(5.0, TPOSE_MAX_ANGLE_DEG - 1.0)
        elif key == ord('z'):
            TPOSE_MIN_FRAMES = max(1, TPOSE_MIN_FRAMES - 1)
        elif key == ord('x'):
            TPOSE_MIN_FRAMES += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")