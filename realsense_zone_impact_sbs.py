# realsense_zone_impact_sbs.py
# pip install pyrealsense2 opencv-python numpy
import numpy as np
import cv2
import pyrealsense2 as rs
import time

# -------- налаштування --------
W, H, FPS = 640, 480, 60

CALIB_FRAMES = 25

# Пороги/фільтри
IMPACT_THRESHOLD = 0.05      # м: наскільки ближче за стіну вважаємо «удар»
IMPACT_WINDOW_LO = 0.005     # м: нижня межа вікна удару (відсікти тремтіння біля 0)
IMPACT_WINDOW_HI_EXTRA = 0.020  # м: верх вікна = IMPACT_THRESHOLD + це
PROTRUSION_MIN = 0.015       # м: груба маска «щось перед стіною»
PROTRUSION_MAX = 0.350       # м: верх для грубої маски
GAUSS_KSIZE = 5              # згладжування depth перед дифом
MORPH_KERNEL = 3             # морфологія масок
MIN_BLOB_AREA = 180          # px: мін. площа компоненти
COOLDOWN_S = 0.20            # анти-спам між подіями

# Відображення
flip_view = False
use_colormap = True

# Вибір зони (клікати на ЛІВІЙ половині — depth)
select_points = []
have_zone = False
zone_mask = None

# Пер-піксельна карта стіни в зоні
wall_map = None
last_impact_ts = 0.0

# Для підказок порядку кліків
zone_hint = ["клік 1: верх-ліво", "клік 2: верх-право",
             "клік 3: низ-право", "клік 4: низ-ліво"]

# Список останніх попадань для візуалізації
CONTACT_POINTS = []  # елементи: {"pos":(x,y), "ttl":N}, у координатах кадру (depth/color)

POINT_LIFETIME = 30  # кадри (скільки тримаємо мітку попадання)

# ---- callback миші (клікати по лівій половині) ----
def mouse_cb(event, x, y, flags, param):
    global select_points, have_zone, wall_map, zone_mask
    # composite має ширину 2*W; depth — ліва половина [0..W)
    if event == cv2.EVENT_LBUTTONDOWN and x < W:
        if len(select_points) < 4:
            select_points.append((x, y))
        if len(select_points) == 4:
            have_zone = True
            wall_map = None    # форсуємо перекалібру
            zone_mask = build_zone_mask(H, W, select_points)

def draw_zone_guides(img):
    # рисуємо полігон на переданому img (BGR) за select_points
    if len(select_points) == 0:
        return
    for i, pt in enumerate(select_points):
        cv2.circle(img, pt, 5, (0, 255, 255), -1)
        cv2.putText(img, f"{i+1}", (pt[0]+6, pt[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
    if len(select_points) >= 2:
        cv2.line(img, select_points[0], select_points[1], (0,255,255), 1)
    if len(select_points) >= 3:
        cv2.line(img, select_points[1], select_points[2], (0,255,255), 1)
    if len(select_points) == 4:
        cv2.line(img, select_points[2], select_points[3], (0,255,255), 1)
        cv2.line(img, select_points[3], select_points[0], (0,255,255), 1)

def build_zone_mask(h, w, pts):
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pts) == 4:
        poly = np.array(pts, dtype=np.int32)
        cv2.fillConvexPoly(mask, poly, 255)
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
    global flip_view, use_colormap, have_zone, select_points, wall_map, zone_mask, last_impact_ts, IMPACT_THRESHOLD, CONTACT_POINTS

    # --- RealSense init: увімкнемо і depth, і color, і вирівняємо depth до color ---
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    profile = pipeline.start(cfg)

    # розрахунок масштабу depth
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # фільтри RealSense
    dec = rs.decimation_filter()
    dec.set_option(rs.option.filter_magnitude, 1)

    spat = rs.spatial_filter()
    spat.set_option(rs.option.filter_magnitude, 2)
    spat.set_option(rs.option.filter_smooth_alpha, 0.5)
    spat.set_option(rs.option.filter_smooth_delta, 20)

    temp = rs.temporal_filter()
    temp.set_option(rs.option.filter_smooth_alpha, 0.4)
    temp.set_option(rs.option.filter_smooth_delta, 20)
    temp.set_option(rs.option.holes_fill, 1)

    hole = rs.hole_filling_filter(1)

    # Align depth -> color, щоб координати збігались
    align = rs.align(rs.stream.color)

    print(f"[INFO] D455f {W}x{H}@{FPS}, depth_scale={depth_scale:.6f} m/step")
    cv2.namedWindow("View (Depth | RGB)")
    cv2.setMouseCallback("View (Depth | RGB)", mouse_cb)

    calib_buf = []
    tracking_enabled = False

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            d = frames.get_depth_frame()
            c = frames.get_color_frame()
            if not d or not c:
                continue

            # ланцюжок фільтрів до depth
            d_f = dec.process(d)
            d_f = spat.process(d_f)
            d_f = temp.process(d_f)
            d_f = hole.process(d_f)

            z_raw = np.asanyarray(d_f.get_data()).astype(np.float32)
            z = z_raw * depth_scale  # метри
            color = np.asanyarray(c.get_data())

            if flip_view:
                z = cv2.flip(z, 1)
                color = cv2.flip(color, 1)

            # підготовка візу depth
            depth_vis = visualize_depth(z)

            key = cv2.waitKey(1) & 0xFF

            # --- етап вибору зони ---
            if not have_zone:
                # підказки
                cv2.putText(depth_vis, "C: clear/select 4 corners (UL, UR, BR, BL) on LEFT half; ENTER/T: track; Q/Esc: quit",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 2, cv2.LINE_AA)
                if len(select_points) < 4:
                    cv2.putText(depth_vis, f"Click corners: {zone_hint[len(select_points)]}",
                                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
                draw_zone_guides(depth_vis)

                # composite: depth left, rgb right (без оверлеїв на RGB поки)
                composite = np.hstack([depth_vis, color])

                # показ
                cv2.imshow("View (Depth | RGB)", composite)

                # керування
                if key == ord('c'):
                    select_points = []
                    have_zone = False
                    wall_map = None
                    zone_mask = None
                    calib_buf.clear()
                elif key in (27, ord('q')):
                    break
                elif key == ord('f'):
                    flip_view = not flip_view
                elif key == ord('m'):
                    use_colormap = not use_colormap
                elif key in (13, ord('t')) and len(select_points) == 4:
                    have_zone = True
                continue

            # --- калібрування пер‑піксельної карти стіни ---
            if wall_map is None:
                cv2.putText(depth_vis, "Calibrating wall map... stand clear of zone",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2, cv2.LINE_AA)
                draw_zone_guides(depth_vis)

                calib_buf.append(z.copy())
                if len(calib_buf) >= CALIB_FRAMES:
                    stack = np.stack(calib_buf, axis=0)
                    wall_map = np.median(stack, axis=0).astype(np.float32)
                    if zone_mask is not None:
                        wall_map[zone_mask == 0] = np.nan
                    calib_buf.clear()
                    print("[CALIB] wall_map built.")
                    tracking_enabled = True

                # показ side-by-side
                comp = np.hstack([depth_vis, color])
                cv2.imshow("View (Depth | RGB)", comp)

                # клавіші у калібруванні
                if key == ord('c'):
                    select_points = []
                    have_zone = False
                    wall_map = None
                    zone_mask = None
                    calib_buf.clear()
                elif key in (27, ord('q')):
                    break
                elif key == ord('f'):
                    flip_view = not flip_view
                elif key == ord('m'):
                    use_colormap = not use_colormap
                continue

            # ---- РЕЖИМ ТРЕКІНГУ ----
            # тексти
            txt1 = f"T/ENTER: tracking={'ON' if tracking_enabled else 'OFF'}   C: re-select zone   +/- thr={IMPACT_THRESHOLD*1000:.0f}mm"
            cv2.putText(depth_vis, txt1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 2, cv2.LINE_AA)
            draw_zone_guides(depth_vis)

            # працюємо лише в межах зони
            z_zone = z.copy()
            if zone_mask is not None:
                z_zone[zone_mask == 0] = np.nan

            # згладимо depth
            if GAUSS_KSIZE >= 3 and GAUSS_KSIZE % 2 == 1:
                z_zone = cv2.GaussianBlur(z_zone, (GAUSS_KSIZE, GAUSS_KSIZE), 0)

            # пер‑піксельний «виступ»
            protr = wall_map - z_zone  # >0 — ближче за стіну

            # груба маска «щось перед стіною»
            mask_gross = np.zeros((H, W), dtype=np.uint8)
            valid = np.isfinite(protr)
            cond_gross = valid & (protr > PROTRUSION_MIN) & (protr < PROTRUSION_MAX)
            mask_gross[cond_gross] = 255

            if MORPH_KERNEL >= 3 and MORPH_KERNEL % 2 == 1:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
                mask_gross = cv2.morphologyEx(mask_gross, cv2.MORPH_OPEN, k, iterations=1)

            # «вікно удару» — дуже близько до стіни
            mask_hit = np.zeros((H, W), dtype=np.uint8)
            hi = IMPACT_THRESHOLD + IMPACT_WINDOW_HI_EXTRA
            cond_hit = valid & (protr >= IMPACT_WINDOW_LO) & (protr <= hi)
            mask_hit[cond_hit] = 255
            if zone_mask is not None:
                mask_hit = cv2.bitwise_and(mask_hit, zone_mask)

            # контури ударів
            contours, _ = cv2.findContours(mask_hit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hits = []
            for ctn in contours:
                area = cv2.contourArea(ctn)
                if area < MIN_BLOB_AREA:
                    continue
                M = cv2.moments(ctn)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
                hits.append((cx, cy))

            # оновлюємо список точок попадань (TTL)
            now = time.time()
            for (cx, cy) in hits:
                # анти-спам
                if tracking_enabled and (now - last_impact_ts > COOLDOWN_S):
                    p_here = protr[cy, cx]
                    if np.isfinite(p_here):
                        CONTACT_POINTS.append({"pos": (cx, cy), "ttl": POINT_LIFETIME})
                        print(f"[IMPACT] px=({cx},{cy}) protr={p_here:.3f} m")
                        last_impact_ts = now

            # малюємо мітки попадань на обох візуалізаціях
            # (копії, щоб не псувати базові матриці)
            depth_draw = depth_vis.copy()
            color_draw = color.copy()

            # зелена маска «щось перед стіною» — тільки на depth
            m_col = cv2.cvtColor(mask_gross, cv2.COLOR_GRAY2BGR)
            m_col[:, :, 1] = np.maximum(m_col[:, :, 1], mask_gross)  # зелена
            depth_draw = cv2.addWeighted(depth_draw, 0.8, m_col, 0.3, 0)

            # малюємо всі активні точки з TTL
            new_points = []
            for pt in CONTACT_POINTS:
                (px, py) = pt["pos"]
                # depth
                cv2.circle(depth_draw, (px, py), 10, (0, 0, 255), 2)
                # rgb
                cv2.circle(color_draw, (px, py), 10, (0, 0, 255), 2)
                # зменшити TTL і зберегти, якщо ще >0
                pt["ttl"] -= 1
                if pt["ttl"] > 0:
                    new_points.append(pt)
            CONTACT_POINTS = new_points

            # склеюємо в одне вікно: зліва depth, справа rgb
            composite = np.hstack([depth_draw, color_draw])
            cv2.imshow("View (Depth | RGB)", composite)

            # --- клавіші ---
            if key in (27, ord('q')):
                break
            elif key == ord('c'):
                select_points = []
                have_zone = False
                wall_map = None
                zone_mask = None
                calib_buf.clear()
                CONTACT_POINTS.clear()
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