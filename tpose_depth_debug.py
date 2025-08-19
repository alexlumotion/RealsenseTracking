import cv2, numpy as np, pyrealsense2 as rs, mediapipe as mp
import time

# ------------------ НАЛАШТУВАННЯ ------------------
W, H, FPS = 640, 480, 30

# Дзеркалити картинку (як у дзеркалі)
MIRROR = True

# Параметри T-пози
TPOSE_Y_TOL = 0.08       # допуск по вертикалі (нормалізовані координати 0..1)
TPOSE_MIN_EXTEND = 1.4   # наскільки зап’ястя далі від центру плечей за півширину плечей
TPOSE_MIN_FRAMES = 4     # скільки кадрів підряд має триматися умова
TPOSE_COOLDOWN = 20      # антиспам після спрацювання

# Візуалізація depth
DEPTH_ALPHA = 0.03       # масштаб у convertScaleAbs
# ---------------------------------------------------

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

def draw_landmarks(image_bgr, results):
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            image_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_style.get_default_pose_landmarks_style()
        )

def is_t_pose(lm, mp_pose):
    """
    lm: список landmark-ів MediaPipe (results.pose_landmarks.landmark)
    Повертає (bool, debug_dict)
    """
    LSH = mp_pose.PoseLandmark.LEFT_SHOULDER
    RSH = mp_pose.PoseLandmark.RIGHT_SHOULDER
    LE  = mp_pose.PoseLandmark.LEFT_ELBOW
    RE  = mp_pose.PoseLandmark.RIGHT_ELBOW
    LW  = mp_pose.PoseLandmark.LEFT_WRIST
    RW  = mp_pose.PoseLandmark.RIGHT_WRIST

    ls = lm[LSH]; rs = lm[RSH]
    le = lm[LE];  re = lm[RE]
    lw = lm[LW];  rw = lm[RW]

    shoulder_cx = (ls.x + rs.x) / 2.0
    shoulder_cy = (ls.y + rs.y) / 2.0
    shoulder_w  = abs(ls.x - rs.x) + 1e-6

    # По Y — приблизно на лінії плечей
    left_y_ok  = abs(lw.y - shoulder_cy) <= TPOSE_Y_TOL
    right_y_ok = abs(rw.y - shoulder_cy) <= TPOSE_Y_TOL
    left_elbow_y_ok  = abs(le.y - shoulder_cy) <= (TPOSE_Y_TOL + 0.03)
    right_elbow_y_ok = abs(re.y - shoulder_cy) <= (TPOSE_Y_TOL + 0.03)

    # По X — достатньо витягнуті
    min_off = (shoulder_w * 0.5) * TPOSE_MIN_EXTEND
    left_x_ok  = (shoulder_cx - lw.x) >= min_off
    right_x_ok = (rw.x - shoulder_cx) >= min_off

    ok = left_y_ok and right_y_ok and left_x_ok and right_x_ok and left_elbow_y_ok and right_elbow_y_ok

    dbg = {
        "shoulder_w": round(shoulder_w,3),
        "min_off": round(min_off,3),
        "dyL": round(lw.y - shoulder_cy,3),
        "dyR": round(rw.y - shoulder_cy,3),
        "dxL": round(shoulder_cx - lw.x,3),
        "dxR": round(rw.x - shoulder_cx,3),
        "okY": (left_y_ok, right_y_ok),
        "okX": (left_x_ok, right_x_ok)
    }
    return ok, dbg

def main():
    # RealSense: depth + infrared, і вирівнюємо depth до IR
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth,     W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.infrared,  W, H, rs.format.y8,  FPS)  # IR (грейскейл)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.infrared)   # приводимо depth до системи координат IR

    # MediaPipe Pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    tpose_frames = 0
    tpose_cooldown = 0
    last_event_time = 0

    print("[INFO] Running... Esc to quit.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth = frames.get_depth_frame()
            ir    = frames.get_infrared_frame()
            if not depth or not ir:
                continue

            # IR -> numpy
            ir_img = np.asanyarray(ir.get_data())        # uint8, shape (H, W), Y8
            depth_img = np.asanyarray(depth.get_data())  # uint16, shape (H, W)

            # Дзеркало (і IR, і depth), щоб координати співпадали з візуалізацією
            if MIRROR:
                ir_img    = cv2.flip(ir_img, 1)
                depth_img = cv2.flip(depth_img, 1)

            # Готуємо картинку для MediaPipe: з грейскейлу робимо 3‑канальний RGB
            rgb_for_mp = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2RGB)

            # Pose
            res = pose.process(rgb_for_mp)

            # Малюємо landmark-и на візуальній копії IR (перетворимо в BGR для OpenCV)
            vis = cv2.cvtColor(rgb_for_mp, cv2.COLOR_RGB2BGR)
            draw_landmarks(vis, res)

            # Візуалізація depth
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=DEPTH_ALPHA),
                cv2.COLORMAP_JET
            )

            # Перевірка T-пози
            person_present = res.pose_landmarks is not None
            if person_present:
                ok_t, dbg = is_t_pose(res.pose_landmarks.landmark, mp_pose)

                if tpose_cooldown > 0:
                    tpose_cooldown -= 1

                if ok_t and tpose_cooldown == 0:
                    tpose_frames += 1
                    cv2.putText(vis, f"T-Pose? frames={tpose_frames}",
                                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2, cv2.LINE_AA)
                    if tpose_frames >= TPOSE_MIN_FRAMES:
                        print(f"[EVENT] T_POSE  dbg={dbg}  t={round(time.time()-last_event_time,2)}s")
                        last_event_time = time.time()
                        tpose_frames = 0
                        tpose_cooldown = TPOSE_COOLDOWN
                else:
                    tpose_frames = 0

                # Дебаг-підписи
                cv2.putText(vis, f"TPOSE tolY={TPOSE_Y_TOL:.2f}  ext={TPOSE_MIN_EXTEND:.2f}  need={TPOSE_MIN_FRAMES} frm  cd={tpose_cooldown}",
                            (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,50), 2, cv2.LINE_AA)

            else:
                tpose_frames = 0
                cv2.putText(vis, "NO PERSON", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

            # Поруч IR+landmarks та depth
            stacked = np.hstack([vis, depth_vis])
            cv2.imshow("IR landmarks (left) | Depth (right)", stacked)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc
                break
            # Хоткеї для тюнінгу на льоту
            elif key == ord('q'): TPOSE_Y_TOL = max(0.0, TPOSE_Y_TOL - 0.01)
            elif key == ord('w'): TPOSE_Y_TOL += 0.01
            elif key == ord('a'): TPOSE_MIN_EXTEND = max(0.5, TPOSE_MIN_EXTEND - 0.05)
            elif key == ord('s'): TPOSE_MIN_EXTEND += 0.05
            elif key == ord('z'): TPOSE_MIN_FRAMES = max(1, TPOSE_MIN_FRAMES - 1)
            elif key == ord('x'): TPOSE_MIN_FRAMES += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Stopped.")

if __name__ == "__main__":
    main()