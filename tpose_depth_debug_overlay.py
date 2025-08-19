import cv2, numpy as np, pyrealsense2 as pr, mediapipe as mp
import time

# ------------------ SETTINGS ------------------
W, H, FPS = 640, 480, 30
MIRROR = True

# By default use COLOR for MediaPipe (reliable).
USE_COLOR_FOR_POSE = True        # toggle with 'i'
APPLY_CLAHE_IR     = True        # toggle with 'h' (IR mode only)

# T-pose thresholds
TPOSE_Y_TOL       = 0.08   # зап’ястя ~на одній висоті з плечима (норм. координати)
TPOSE_MIN_EXTEND  = 1.4    # наскільки далі за половину міжплечової відстані мають бути зап’ястя
TPOSE_MIN_FRAMES  = 4      # скільки послідовних кадрів підтверджувати
TPOSE_COOLDOWN    = 20     # затримка перед наступним спрацюванням

DEPTH_ALPHA = 0.03

mp_pose  = mp.solutions.pose
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

def draw_landmarks(image_bgr, results):
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            image_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_style.get_default_pose_landmarks_style()
        )

def put(text, org, img, scale=0.55, color=(230,230,230), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def circ(img, p, r=6, color=(0,255,255), thick=-1):
    cv2.circle(img, p, r, color, thick, cv2.LINE_AA)

def to_px(x, y):
    return int(np.clip(x*W, 0, W-1)), int(np.clip(y*H, 0, H-1))

def is_t_pose(lm):
    LSH = mp_pose.PoseLandmark.LEFT_SHOULDER
    RSH = mp_pose.PoseLandmark.RIGHT_SHOULDER
    LE  = mp_pose.PoseLandmark.LEFT_ELBOW
    RE  = mp_pose.PoseLandmark.RIGHT_ELBOW
    LW  = mp_pose.PoseLandmark.LEFT_WRIST
    RW  = mp_pose.PoseLandmark.RIGHT_WRIST

    ls = lm[LSH]; rs_ = lm[RSH]
    le = lm[LE];  re = lm[RE]
    lw = lm[LW];  rw = lm[RW]

    shoulder_cx = (ls.x + rs_.x) / 2.0
    shoulder_cy = (ls.y + rs_.y) / 2.0
    shoulder_w  = abs(ls.x - rs_.x) + 1e-6

    dyL = lw.y - shoulder_cy
    dyR = rw.y - shoulder_cy
    left_y_ok  = abs(dyL) <= TPOSE_Y_TOL
    right_y_ok = abs(dyR) <= TPOSE_Y_TOL

    left_elbow_y_ok  = abs(le.y - shoulder_cy) <= (TPOSE_Y_TOL + 0.03)
    right_elbow_y_ok = abs(re.y - shoulder_cy) <= (TPOSE_Y_TOL + 0.03)

    min_off = (shoulder_w * 0.5) * TPOSE_MIN_EXTEND
    dxL = shoulder_cx - lw.x
    dxR = rw.x - shoulder_cx
    left_x_ok  = dxL >= min_off
    right_x_ok = dxR >= min_off

    ok = left_y_ok and right_y_ok and left_x_ok and right_x_ok and left_elbow_y_ok and right_elbow_y_ok
    dbg = {
        "shoulder_w": round(shoulder_w,3),
        "min_off": round(min_off,3),
        "dyL": round(dyL,3), "dyR": round(dyR,3),
        "dxL": round(dxL,3), "dxR": round(dxR,3),
        "okY": (left_y_ok, right_y_ok),
        "okX": (left_x_ok, right_x_ok)
    }
    return ok, dbg, (ls, rs_, le, re, lw, rw)

def main():
    # оголошуємо глобалки, які змінюємо в main (гарячими клавішами)
    global TPOSE_Y_TOL, TPOSE_MIN_EXTEND, TPOSE_MIN_FRAMES, USE_COLOR_FOR_POSE, APPLY_CLAHE_IR

    pipeline = pr.pipeline()
    cfg = pr.config()
    cfg.enable_stream(pr.stream.depth, W, H, pr.format.z16, FPS)
    cfg.enable_stream(pr.stream.color, W, H, pr.format.bgr8, FPS)   # для RGB
    cfg.enable_stream(pr.stream.infrared, W, H, pr.format.y8,  FPS) # для IR (опційно)
    pipeline.start(cfg)
    align = pr.align(pr.stream.color)  # вирівнюємо depth під колір

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,                # трішки краще ніж 0
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    tpose_frames = 0
    tpose_cooldown = 0
    last_event_time = 0

    print("[INFO] Running... Esc=exit. Hotkeys: q/w tolY, a/s extend, z/x frames, i RGB<->IR, h CLAHE(IR)")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            ir    = frames.get_infrared_frame()

            if not depth:
                continue

            depth_img = np.asanyarray(depth.get_data())
            if MIRROR:
                depth_img = cv2.flip(depth_img, 1)

            # Вибір джерела для MediaPipe
            if USE_COLOR_FOR_POSE and color:
                img_for_pose = np.asanyarray(color.get_data())          # BGR
                if MIRROR: img_for_pose = cv2.flip(img_for_pose, 1)
                vis = img_for_pose.copy()
            elif ir is not None:
                ir_img = np.asanyarray(ir.get_data())                    # 8-bit
                if APPLY_CLAHE_IR:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    ir_img = clahe.apply(ir_img)
                if MIRROR: ir_img = cv2.flip(ir_img, 1)
                img_for_pose = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR)  # -> BGR для візуалізації/MP
                vis = img_for_pose.copy()
            else:
                # немає відповідного фрейма
                continue

            # MediaPipe
            rgb_for_mp = cv2.cvtColor(img_for_pose, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_for_mp)
            draw_landmarks(vis, results)

            # Depth візуалізація
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=DEPTH_ALPHA),
                cv2.COLORMAP_JET
            )

            if results.pose_landmarks:
                ok_t, dbg, (ls, rs_, le, re, lw, rw) = is_t_pose(results.pose_landmarks.landmark)
                pts = {
                    "LS": to_px(ls.x, ls.y),
                    "RS": to_px(rs_.x, rs_.y),
                    "LE": to_px(le.x, le.y),
                    "RE": to_px(re.x, re.y),
                    "LW": to_px(lw.x, lw.y),
                    "RW": to_px(rw.x, rw.y),
                }
                cv2.line(vis, pts["LS"], pts["RS"], (0,255,255), 2, cv2.LINE_AA)
                for k,c in [("LS",(0,255,255)),("RS",(0,255,255)),("LE",(255,200,0)),("RE",(255,200,0)),("LW",(0,255,0)),("RW",(0,255,0))]:
                    circ(vis, pts[k], color=c)

                put(f"dxL={dbg['dxL']} dyL={dbg['dyL']}", (pts["LW"][0]+8, pts["LW"][1]-8), vis, 0.5, (0,255,0))
                put(f"dxR={dbg['dxR']} dyR={dbg['dyR']}", (pts["RW"][0]+8, pts["RW"][1]-8), vis, 0.5, (0,255,0))

                put(f"TPOSE tolY={TPOSE_Y_TOL:.2f}  extend>={TPOSE_MIN_EXTEND:.2f}  need={TPOSE_MIN_FRAMES}  "
                    f"src={'RGB' if USE_COLOR_FOR_POSE else 'IR'}  CLAHE={APPLY_CLAHE_IR and not USE_COLOR_FOR_POSE}",
                    (10, 26), vis, 0.6, (200,200,50))

                if tpose_cooldown > 0:
                    tpose_cooldown -= 1

                if ok_t and tpose_cooldown == 0:
                    tpose_frames += 1
                    put(f"T-Pose? frames={tpose_frames}", (10, 50), vis, 0.7, (0,200,255))
                    if tpose_frames >= TPOSE_MIN_FRAMES:
                        print(f"[EVENT] T_POSE  dbg={dbg}  dt={round(time.time()-last_event_time,2)}s")
                        last_event_time = time.time()
                        tpose_frames = 0
                        tpose_cooldown = TPOSE_COOLDOWN
                else:
                    tpose_frames = 0
            else:
                put(f"NO PERSON  src={'RGB' if USE_COLOR_FOR_POSE else 'IR'}", (10, 26), vis, 0.8, (0,0,255))

            stacked = np.hstack([vis, depth_vis])
            cv2.imshow("Pose view (left) | Depth (right)", stacked)

            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            elif key == ord('q'): TPOSE_Y_TOL = max(0.0, TPOSE_Y_TOL - 0.01)
            elif key == ord('w'): TPOSE_Y_TOL += 0.01
            elif key == ord('a'): TPOSE_MIN_EXTEND = max(0.5, TPOSE_MIN_EXTEND - 0.05)
            elif key == ord('s'): TPOSE_MIN_EXTEND += 0.05
            elif key == ord('z'): TPOSE_MIN_FRAMES = max(1, TPOSE_MIN_FRAMES - 1)
            elif key == ord('x'): TPOSE_MIN_FRAMES += 1
            elif key == ord('i'): USE_COLOR_FOR_POSE = not USE_COLOR_FOR_POSE
            elif key == ord('h'): APPLY_CLAHE_IR = not APPLY_CLAHE_IR

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Stopped.")

if __name__ == "__main__":
    main()