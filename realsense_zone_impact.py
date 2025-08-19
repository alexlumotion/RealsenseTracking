# realsense_zone_impact.py
# pip install pyrealsense2 opencv-python numpy
import numpy as np
import cv2
import pyrealsense2 as rs
import time

# ---- налаштування ----
W, H, FPS = 640, 480, 60
CALIB_FRAMES = 25

# фільтри & пороги
IMPACT_THRESHOLD = 0.05      # м: базовий поріг «близько до стіни»
IMPACT_WINDOW_LO = 0.005     # м: нижня межа (щоб не ловити тремтіння біля 0)
IMPACT_WINDOW_HI_EXTRA = 0.020  # м: IMPACT_THRESHOLD + це
PROTRUSION_MIN = 0.015       # м: для грубої маски «щось перед стіною»
PROTRUSION_MAX = 0.350       # м
GAUSS_KSIZE = 5
MORPH_KERNEL = 3
MIN_BLOB_AREA = 180          # px
COOLDOWN_S = 0.20

# візуалізація
flip_view = False
use_colormap = True

# вибір зони
select_points = []
have_zone = False

# пер-піксельна карта стіни (у зоні)
wall_map = None
last_impact_ts = 0.0

zone_hint = ["клік 1: верх-ліво", "клік 2: верх-право", "клік 3: низ-право", "клік 4: низ-ліво"]

def mouse_cb(event, x, y, flags, param):
    global select_points, have_zone, wall_map
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(select_points) < 4:
            select_points.append((x, y))
        if len(select_points) == 4:
            have_zone = True
            wall_map = None  # треба перекалібрувати

def draw_zone_guides(img):
    for i, pt in enumerate(select_points):
        cv2.circle(img, pt, 5, (0,255,255), -1)
        cv2.putText(img, f"{i+1}", (pt[0]+6, pt[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
    if len(select_points) >= 2: cv2.line(img, select_points[0], select_points[1], (0,255,255), 1)
    if len(select_points) >= 3: cv2.line(img, select_points[1], select_points[2], (0,255,255), 1)
    if len(select_points) == 4:
        cv2.line(img, select_points[2], select_points[3], (0,255,255), 1)
        cv2.line(img, select_points[3], select_points[0], (0,255,255), 1)

def build_zone_mask(h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(select_points, dtype=np.int32)
    cv2.fillConvexPoly(mask, pts, 255)
    return mask

def visualize_depth(z_m):
    if not use_colormap:
        z_show = cv2.normalize(z_m, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(z_show.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    z_mm = (z_m * 1000.0).astype(np.float32)
    z_mm = np.clip(z_mm, 0, np.percentile(z_mm, 99))
    z_u8 = cv2.convertScaleAbs(z_mm, alpha=255.0/max(np.max(z_mm),1e-6))
    return cv2.applyColorMap(z_u8, cv2.COLORMAP_JET)

def main():
    global flip_view, use_colormap, have_zone, select_points, wall_map, last_impact_ts, IMPACT_THRESHOLD

    # --- RealSense init (depth + фільтри) ---
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    profile = pipeline.start(cfg)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    dec = rs.decimation_filter()    # зменшує шум (і розмір)
    dec.set_option(rs.option.filter_magnitude, 2)

    spat = rs.spatial_filter()      # просторовий
    spat.set_option(rs.option.filter_magnitude, 2)
    spat.set_option(rs.option.filter_smooth_alpha, 0.5)
    spat.set_option(rs.option.filter_smooth_delta, 20)

    temp = rs. temporal_filter()    # часовий
    temp.set_option(rs.option.filter_smooth_alpha, 0.4)
    temp.set_option(rs.option.filter_smooth_delta, 20)
    temp.set_option(rs.option.holes_fill, 1)

    hole = rs.hole_filling_filter(1)

    print(f"[INFO] D455f {W}x{H}@{FPS}, depth_scale={depth_scale:.6f} m/step")
    cv2.namedWindow("Calib/Track")
    cv2.setMouseCallback("Calib/Track", mouse_cb)

    calib_buf = []
    tracking_enabled = False

    try:
        while True:
            frames = pipeline.wait_for_frames()
            d = frames.get_depth_frame()
            if not d: continue

            # застосовуємо ланцюг фільтрів до depth
            d = dec.process(d)
            d = spat.process(d)
            d = temp.process(d)
            d = hole.process(d)

            z_raw = np.asanyarray(d.get_data()).astype(np.float32)
            z = z_raw * depth_scale
            if flip_view: z = cv2.flip(z, 1)

            vis = visualize_depth(z)

            key = cv2.waitKey(1) & 0xFF

            # --- вибір зони ---
            if not have_zone:
                cv2.putText(vis, "C: clear/select 4 corners (UL, UR, BR, BL). ENTER/T: track. Q/Esc: quit",
                            (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 2, cv2.LINE_AA)
                if len(select_points) < 4:
                    cv2.putText(vis, f"Click corners: {zone_hint[len(select_points)]}",
                                (10,48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
                draw_zone_guides(vis)
                cv2.imshow("Calib/Track", vis)

                if key == ord('c'):
                    select_points=[]; have_zone=False; wall_map=None; calib_buf.clear()
                elif key in (27, ord('q')):
                    break
                elif key == ord('f'):
                    flip_view = not flip_view
                elif key == ord('m'):
                    use_colormap = not use_colormap
                elif key in (13, ord('t')) and len(select_points)==4:
                    have_zone = True
                continue

            zone_mask = build_zone_mask(z.shape[0], z.shape[1])

            # --- калібрування пер-піксельної стіни ---
            if wall_map is None:
                cv2.putText(vis, "Calibrating wall map... stand clear of zone",
                            (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2, cv2.LINE_AA)
                draw_zone_guides(vis); cv2.imshow("Calib/Track", vis)

                calib_buf.append(z.copy())
                if len(calib_buf) >= CALIB_FRAMES:
                    stack = np.stack(calib_buf, axis=0)  # (N,H,W)
                    wall_map = np.median(stack, axis=0).astype(np.float32)
                    wall_map[zone_mask==0] = np.nan  # беремо лише в зоні
                    calib_buf.clear()
                    print("[CALIB] wall_map built.")
                    tracking_enabled = True
                # клавіші під час калібру
                if key == ord('c'):
                    select_points=[]; have_zone=False; wall_map=None; calib_buf.clear()
                elif key in (27, ord('q')):
                    break
                elif key == ord('f'):
                    flip_view = not flip_view
                elif key == ord('m'):
                    use_colormap = not use_colormap
                continue

            # ---- режим трекінгу ----
            txt1 = f"T/ENTER: tracking={'ON' if tracking_enabled else 'OFF'}   C: re-select zone   +/- thr={IMPACT_THRESHOLD*1000:.0f}mm"
            cv2.putText(vis, txt1, (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 2, cv2.LINE_AA)
            draw_zone_guides(vis)

            # лише в межах зони
            z_zone = z.copy()
            z_zone[zone_mask==0] = np.nan

            # згладимо для стабільнішої різниці
            if GAUSS_KSIZE>=3 and GAUSS_KSIZE%2==1:
                z_zone = cv2.GaussianBlur(z_zone, (GAUSS_KSIZE, GAUSS_KSIZE), 0)

            # пер-піксельний «виступ»
            protr = wall_map - z_zone  # >0 — ближче за стіну

            # груба маска — щось перед стіною
            mask_gross = np.zeros_like(zone_mask)
            valid = np.isfinite(protr)
            cond_gross = valid & (protr > PROTRUSION_MIN) & (protr < PROTRUSION_MAX)
            mask_gross[cond_gross] = 255

            if MORPH_KERNEL>=3 and MORPH_KERNEL%2==1:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
                mask_gross = cv2.morphologyEx(mask_gross, cv2.MORPH_OPEN, k, iterations=1)

            # «вікно удару» — дуже близько до стіни
            mask_hit = np.zeros_like(zone_mask)
            hi = IMPACT_THRESHOLD + IMPACT_WINDOW_HI_EXTRA
            cond_hit = valid & (protr >= IMPACT_WINDOW_LO) & (protr <= hi)
            mask_hit[cond_hit] = 255
            mask_hit = cv2.bitwise_and(mask_hit, zone_mask)

            # контури і відбір
            contours, _ = cv2.findContours(mask_hit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hits = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < MIN_BLOB_AREA: continue
                M = cv2.moments(c)
                if M["m00"] == 0: continue
                cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
                hits.append((cx, cy))

            # малюємо маску поверх depth
            m_col = cv2.cvtColor(mask_gross, cv2.COLOR_GRAY2BGR)
            m_col[:,:,1] = np.maximum(m_col[:,:,1], mask_gross)  # зелена
            ov = cv2.addWeighted(vis, 0.8, m_col, 0.3, 0)

            # позначаємо попадання
            now = time.time()
            for (cx, cy) in hits:
                cv2.circle(ov, (cx, cy), 10, (0,0,255), 2)
                if tracking_enabled and (now - last_impact_ts > COOLDOWN_S):
                    p_here = protr[cy, cx]
                    if np.isfinite(p_here):
                        print(f"[IMPACT] px=({cx},{cy}) protr={p_here:.3f} m")
                        last_impact_ts = now

            cv2.imshow("Calib/Track", ov)

            # --- клавіші ---
            if key in (27, ord('q')): break
            elif key == ord('c'):
                select_points=[]; have_zone=False; wall_map=None
            elif key in (13, ord('t')):
                tracking_enabled = not tracking_enabled
            elif key == ord('+'):
                IMPACT_THRESHOLD += 0.005
            elif key == ord('-'):
                IMPACT_THRESHOLD = max(0.005, IMPACT_THRESHOLD - 0.005)
            elif key == ord('f'):
                flip_view = not flip_view
            elif key == ord('m'):
                use_colormap = not use_colormap

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] stopped.")

if __name__ == "__main__":
    main()