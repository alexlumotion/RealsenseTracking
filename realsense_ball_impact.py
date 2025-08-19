# realsense_ball_impact.py
# Python 3.9+, pip install pyrealsense2 opencv-python numpy

import argparse
import time
from collections import deque
import numpy as np
import cv2
import pyrealsense2 as rs

# -------------------- дефолти/константи --------------------
W_DEFAULT, H_DEFAULT, FPS_DEFAULT = 640, 480, 60

# Пороги (метри)
PROTRUSION_MIN_DEFAULT   = 0.020   # мін. виступ від площини стіни (20 мм)
PROTRUSION_MAX_DEFAULT   = 0.300   # макс. виступ (зріз далеких/випадкових)
CONTACT_THRESH_DEFAULT   = 0.015   # контакт, якщо виступ < 15 мм
APPROACH_THRESH_DEFAULT  = 0.040   # фаза «підходу» — якщо p > 40 мм

# Морфологія/блоб
MIN_BLOB_AREA            = 200     # мінімальна площа компоненти (px)
ERODE_ITERS              = 1
DILATE_ITERS             = 1

# Параметри м’яча
BALL_DIAMETER_M          = 0.05    # 5 см
CIRCULARITY_MIN          = 0.60
CIRCULARITY_MAX          = 1.30
AREA_TOL_LOW             = 0.30    # допустиме відхилення від очікуваної площі
AREA_TOL_HIGH            = 3.00

# Трекінг
MAX_TRACK_AGE            = 8       # скільки кадрів тримати трек без оновлення
MAX_ASSOC_DIST           = 40      # px для nearest-neighbor
HIST_LEN                 = 8       # довжина історії p(t)

DRAW_DEBUG               = True
CALIB_FRAMES             = 25      # кадри для медіани стіни


# -------------------- утиліти --------------------
def median_depth_stack(buf: list[np.ndarray]) -> np.ndarray:
    """Медіанна карта глибин із стеку кадрів (у метрах)."""
    return np.median(np.stack(buf, axis=0), axis=0).astype(np.float32)

def gaussian_blur_float(img: np.ndarray, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def deproject_to_3d(intr: rs.intrinsics, u: int, v: int, depth_m: float):
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(depth_m))
    return float(X), float(Y), float(Z)

def contour_circularity(cnt):
    area = cv2.contourArea(cnt)
    per  = cv2.arcLength(cnt, True)
    if per <= 1e-6 or area <= 1e-6:
        return 0.0
    return 4.0*np.pi*area/(per*per)

def expected_ball_area_px(intr: rs.intrinsics, Z: float, D: float):
    # радіус у пікселях: r = fx * (D/2) / Z
    r = intr.fx * (D*0.5) / max(Z, 1e-6)
    return float(np.pi * r * r)


# -------------------- трек --------------------
class Track:
    _next_id = 1
    def __init__(self, cx, cy, p, approach_thresh):
        self.id = Track._next_id; Track._next_id += 1
        self.cx, self.cy = int(cx), int(cy)
        self.missed = 0
        self.history_p = deque([float(p)], maxlen=HIST_LEN)
        self.had_approach = (p > approach_thresh)
        self.last_contact_ts = 0.0

    def update(self, cx, cy, p, approach_thresh):
        self.cx, self.cy = int(cx), int(cy)
        self.missed = 0
        self.history_p.append(float(p))
        if p > approach_thresh:
            self.had_approach = True

    def step_missed(self):
        self.missed += 1

    def p(self):
        return float(self.history_p[-1])


# -------------------- основна програма --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w",   type=int, default=W_DEFAULT)
    ap.add_argument("--h",   type=int, default=H_DEFAULT)
    ap.add_argument("--fps", type=int, default=FPS_DEFAULT)
    ap.add_argument("--flip", action="store_true", help="дзеркальний фліп (як у дзеркалі)")
    args = ap.parse_args()

    W, H, FPS = args.w, args.h, args.fps

    # Локальні пороги/параметри, які можна міняти гарячими клавішами
    protrusion_min  = PROTRUSION_MIN_DEFAULT
    protrusion_max  = PROTRUSION_MAX_DEFAULT
    contact_thresh  = CONTACT_THRESH_DEFAULT
    approach_thresh = APPROACH_THRESH_DEFAULT

    # --- RealSense init (тільки depth) ---
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    profile = pipeline.start(cfg)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()  # м на одиницю
    intr         = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    print(f"[INFO] D455f started: {W}x{H}@{FPS}, depth_scale={depth_scale:.6f} m")
    print("[HINT] 'c': калібрування (стіна без м'яча); '+/-': contact; '['/']': protr-min; '{'/'}': protr-max; 'q'/Esc: вихід.")

    wall_model = None
    tracks: list[Track] = []

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth  = frames.get_depth_frame()
            if not depth:
                continue

            z_raw = np.asanyarray(depth.get_data()).astype(np.float32)  # "кроки"
            z     = z_raw * depth_scale                                  # метри

            if args.flip:
                z = cv2.flip(z, 1)

            key = cv2.waitKey(1) & 0xFF

            # -------- калібрування стіни --------
            if wall_model is None:
                vis = cv2.convertScaleAbs(z, alpha=255.0 / max(np.max(z), 1e-6))
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                cv2.putText(vis, "Press 'c' to calibrate (clear wall).", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2, cv2.LINE_AA)
                cv2.imshow("Depth", vis)

                if key == ord('c'):
                    print(f"[CALIB] collecting {CALIB_FRAMES} frames...")
                    buf = []
                    t0 = time.time()
                    for _ in range(CALIB_FRAMES):
                        frames = pipeline.wait_for_frames()
                        depth  = frames.get_depth_frame()
                        z_raw  = np.asanyarray(depth.get_data()).astype(np.float32)
                        z_m    = z_raw * depth_scale
                        if args.flip: z_m = cv2.flip(z_m, 1)
                        buf.append(z_m)
                        cv2.waitKey(1)
                    wall_model = median_depth_stack(buf)
                    print(f"[CALIB] done in {time.time()-t0:.2f}s.")
                elif key in (27, ord('q')):
                    break
                continue

            # -------- обробка кадру --------
            z_smooth   = gaussian_blur_float(z, ksize=5)
            protrusion = wall_model - z_smooth  # >0: ближче за стіну

            # маска кандидатів
            mask = cv2.inRange(protrusion, protrusion_min, protrusion_max)
            if ERODE_ITERS:  mask = cv2.erode(mask, None, iterations=ERODE_ITERS)
            if DILATE_ITERS: mask = cv2.dilate(mask, None, iterations=DILATE_ITERS)

            # контури + фільтри (площа/круглість/очікуваний розмір м’яча)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            blobs = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_BLOB_AREA:
                    continue
                circ = contour_circularity(cnt)
                if not (CIRCULARITY_MIN <= circ <= CIRCULARITY_MAX):
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
                Zw = float(wall_model[cy, cx])  # глибина стіни в цій точці
                exp_area = expected_ball_area_px(intr, Zw, BALL_DIAMETER_M)
                if not (AREA_TOL_LOW*exp_area <= area <= AREA_TOL_HIGH*exp_area):
                    continue
                blobs.append((cx, cy, area, cnt))

            # --- оновлення треків ---
            used = set()
            for (cx, cy, area, cnt) in blobs:
                p_here = float(protrusion[cy, cx])
                # пошук найближчого треку
                best_i, best_d = -1, 1e9
                for i, tr in enumerate(tracks):
                    d = np.hypot(tr.cx - cx, tr.cy - cy)
                    if d < best_d: best_d, best_i = d, i
                if best_i >= 0 and best_d < MAX_ASSOC_DIST:
                    tracks[best_i].update(cx, cy, p_here, approach_thresh)
                    used.add(best_i)
                else:
                    tracks.append(Track(cx, cy, p_here, approach_thresh))

            # позначити пропущені та прибрати старі
            for i, tr in enumerate(tracks):
                if i not in used:
                    tr.step_missed()
            tracks = [t for t in tracks if t.missed <= MAX_TRACK_AGE]

            # --- детекція контакту ---
            impacts = []
            now = time.time()
            for tr in tracks:
                p_now = tr.p()
                # послідовне падіння p(t) для стабільності
                falling = False
                if len(tr.history_p) >= 3:
                    p2, p1 = tr.history_p[-3], tr.history_p[-2]
                    falling = (p2 > p1 > p_now)

                if tr.had_approach and falling and (p_now < contact_thresh) and (now - tr.last_contact_ts > 0.2):
                    Zw = float(wall_model[tr.cy, tr.cx])
                    X, Y, Zc = deproject_to_3d(intr, tr.cx, tr.cy, Zw)
                    impacts.append((tr.cx, tr.cy, X, Y, Zc, p_now))
                    tr.last_contact_ts = now
                    tr.had_approach = False
                    print(f"[IMPACT] px=({tr.cx},{tr.cy})  XYZ=({X:.3f},{Y:.3f},{Zc:.3f})  protr={p_now*1000:.0f}mm")

                # якщо знову відійшов — дозволимо наступний контакт
                if p_now > approach_thresh:
                    tr.had_approach = True

            # --- візуалізація ---
            if DRAW_DEBUG:
                depth_vis = cv2.convertScaleAbs(z, alpha=255.0 / max(np.max(z), 1e-6))
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                m_col = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                m_col[:, :, 1] = np.maximum(m_col[:, :, 1], mask)
                overlay = cv2.addWeighted(depth_vis, 0.8, m_col, 0.3, 0)

                for tr in tracks:
                    cv2.circle(overlay, (tr.cx, tr.cy), 6, (0,255,255), -1)
                    cv2.putText(overlay, f"id{tr.id} p={tr.p():.3f}",
                                (tr.cx+8, tr.cy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0,255,255), 1, cv2.LINE_AA)

                for (u, v, X, Y, Zc, p_now) in impacts:
                    cv2.circle(overlay, (u, v), 10, (0,0,255), 2)
                    cv2.putText(overlay, f"IMPACT X={X:.3f} Y={Y:.3f} Z={Zc:.3f}",
                                (u+10, v-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0,0,255), 2, cv2.LINE_AA)

                info = (
                    f"protr>={protrusion_min*1000:.0f}..{protrusion_max*1000:.0f}mm   "
                    f"contact<{contact_thresh*1000:.0f}mm   approach>{approach_thresh*1000:.0f}mm"
                )
                cv2.putText(overlay, "c: calibrate   +/-: contact   [ ]: protr-min   { }: protr-max   q/ESC: quit",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 2, cv2.LINE_AA)
                cv2.putText(overlay, info, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,100), 2, cv2.LINE_AA)

                cv2.imshow("Ball impact (depth)", overlay)

            # --- гарячі клавіші ---
            if key in (27, ord('q')):
                break
            elif key == ord('+'):
                contact_thresh += 0.002
            elif key == ord('-'):
                contact_thresh = max(0.003, contact_thresh - 0.002)
            elif key == ord(']'):
                protrusion_min += 0.001
            elif key == ord('['):
                protrusion_min = max(0.001, protrusion_min - 0.001)
            elif key == ord('}'):
                protrusion_max += 0.01
            elif key == ord('{'):
                protrusion_max = max(0.05, protrusion_max - 0.01)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] stopped.")


if __name__ == "__main__":
    main()