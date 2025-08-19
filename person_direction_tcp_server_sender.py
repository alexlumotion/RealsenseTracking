# person_direction_udp_server_debug.py
import cv2, numpy as np, pyrealsense2 as rs, mediapipe as mp
from collections import deque
import socket, time, csv, os
import json

# ---------- UDP ----------
UNITY_IP   = "192.168.68.104"   # <-- IP твого Mac (або увімкни broadcast)
UNITY_PORT = 5005
USE_BROADCAST = False

udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
if USE_BROADCAST:
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
TARGET_ADDR = ("255.255.255.255" if USE_BROADCAST else UNITY_IP, UNITY_PORT)
print(f"[UDP] send => {TARGET_ADDR[0]}:{TARGET_ADDR[1]}  (broadcast={USE_BROADCAST})")

def send_event(event: str, **payload):
    msg = {"type":"event","value":event,"ts":time.time()}
    if payload: msg.update(payload)
    line = (json.dumps(msg) + "\n").encode("utf-8")   # <— було str(msg), стало JSON
    udp.sendto(line, TARGET_ADDR)
    print(f"[UDP] {event}  extra={payload}")

# ---------- DETECTOR PARAMS ----------
W, H, FPS = 640, 480, 30
MAX_VALID_DISTANCE_M = 3.0
ABSENCE_FRAMES = 12

EMA_ALPHA = 0.35          # ↑ зроби більшим для меншої інерції (0..1)
VEL_ALPHA = 0.6
VEL_DEADBAND = 0.0015
MIN_SPEED = 0.006         # ↓ роби меншим, щоб ловити повільні рухи
WINDOW = 10               # ↓ роби меншим для швидшої реакції
MIN_DELTA = 0.06          # ↓ роби меншим, щоб ловити коротші зсуви
COOLDOWN_FRAMES = 10

DRAW_TRAIL = True
TRAIL_LEN  = 25

# ---------- DEBUG ----------
DEBUG_PRINT_EVERY = 10     # друкувати в консоль кожні N кадрів
WRITE_CSV = True
CSV_PATH = "realsense_motion_debug.csv"

# ---------- INIT ----------
if WRITE_CSV:
    start_new = not os.path.exists(CSV_PATH)
    csv_f = open(CSV_PATH, "a", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    if start_new:
        csv_w.writerow(["ts","x","delta","speed","dist_m",
                        "MIN_DELTA","MIN_SPEED","WINDOW",
                        "EMA_ALPHA","VEL_ALPHA","event"])

pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
profile = pipeline.start(cfg)
align = rs.align(rs.stream.color)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

ema_x = None
ema_v = 0.0
xs = deque(maxlen=WINDOW)
cooldown = 0
no_person_ctr = 0
trail = deque(maxlen=TRAIL_LEN)
frame_id = 0
last_event = ""

def emit(event, **extra):
    global last_event
    last_event = event
    send_event(event, **extra)

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            continue
        frame_id += 1

        color_img = np.asanyarray(color.get_data())
        depth_img  = np.asanyarray(depth.get_data())

        # ===== ФЛІП ЯК ЗЕРКАЛО =====
        color_img = cv2.flip(color_img, 1)
        depth_img = cv2.flip(depth_img, 1)  # <-- додано

        # --- MediaPipe ---
        rgb_for_mp = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb_for_mp)

        person_present = False
        x_norm = y_norm = None
        dist_m = 0.0

        if res.pose_landmarks:
            lh = res.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            rh = res.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            x_norm = float((lh.x + rh.x) / 2.0)
            y_norm = float((lh.y + rh.y) / 2.0)

            x_px = int(np.clip(x_norm * W, 0, W-1))
            y_px = int(np.clip(y_norm * H, 0, H-1))
            dist_m = depth.get_distance(x_px, y_px)

            if 0.1 < dist_m < MAX_VALID_DISTANCE_M:
                person_present = True
                cv2.circle(color_img, (x_px, y_px), 6, (0,255,255), -1)

        # --- Tracking ---
        delta = 0.0
        speed = 0.0

        if person_present:
            no_person_ctr = 0
            if ema_x is None:
                ema_x = x_norm
                ema_v = 0.0
                xs.clear()
            else:
                prev_x = ema_x
                ema_x = EMA_ALPHA * x_norm + (1 - EMA_ALPHA) * ema_x
                inst_v = ema_x - prev_x
                ema_v = VEL_ALPHA * inst_v + (1 - VEL_ALPHA) * ema_v

            xs.append(ema_x)

            if len(xs) >= 2:
                delta = xs[-1] - xs[0]
                speed = ema_v if abs(ema_v) >= VEL_DEADBAND else 0.0

            if DRAW_TRAIL and y_norm is not None:
                trail.append((int(ema_x * W), int(y_norm * H)))
                for i in range(1, len(trail)):
                    cv2.line(color_img, trail[i-1], trail[i], (0,255,0), 2)

            if cooldown > 0:
                cooldown -= 1
            else:
                moved_right = (delta >  MIN_DELTA) and (speed >  MIN_SPEED)
                moved_left  = (delta < -MIN_DELTA) and (speed < -MIN_SPEED)

                if moved_right:
                    emit("MOVE_RIGHT", x=round(ema_x,3), speed=round(speed,3),
                         delta=round(delta,3), dist=round(dist_m,2))
                    cooldown = COOLDOWN_FRAMES
                    xs.clear()
                elif moved_left:
                    emit("MOVE_LEFT", x=round(ema_x,3), speed=round(speed,3),
                         delta=round(delta,3), dist=round(dist_m,2))
                    cooldown = COOLDOWN_FRAMES
                    xs.clear()
        else:
            no_person_ctr += 1
            if no_person_ctr == ABSENCE_FRAMES:
                emit("NO_PERSON")
                ema_x = None
                ema_v = 0.0
                xs.clear()
                trail.clear()

        # --- Overlay ---
        cv2.putText(color_img, f"x={0.0 if ema_x is None else ema_x:.3f}  "
                               f"delta={delta:+.3f}  v={speed:+.3f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,220,50), 2, cv2.LINE_AA)
        cv2.putText(color_img, f"thr: d={MIN_DELTA:.3f}  v={MIN_SPEED:.3f}  win={WINDOW}  cooldown={cooldown}",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,50), 2, cv2.LINE_AA)
        if last_event:
            cv2.putText(color_img, f"last: {last_event}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2, cv2.LINE_AA)
        if person_present and dist_m:
            cv2.putText(color_img, f"dist={dist_m:.2f}m",
                        (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET
        )
        view = np.hstack([color_img, depth_vis])
        cv2.imshow("UDP push | Direction with debug", view)

        # --- Console & CSV ---
        if frame_id % DEBUG_PRINT_EVERY == 0:
            print(f"[DBG] x={0.0 if ema_x is None else round(ema_x,3)}  "
                  f"delta={round(delta,3)}  v={round(speed,3)}  "
                  f"thr(d/v)={MIN_DELTA}/{MIN_SPEED}  win={WINDOW}  cd={cooldown}  "
                  f"dist={round(dist_m,2)}")

        if WRITE_CSV:
            csv_w.writerow([time.time(), 0.0 if ema_x is None else ema_x,
                            delta, speed, dist_m,
                            MIN_DELTA, MIN_SPEED, WINDOW,
                            EMA_ALPHA, VEL_ALPHA, last_event])

        # --- Hotkeys (швидкий тюнінг) ---
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break  # Esc
        elif key == ord('q'): MIN_DELTA = max(0.0, MIN_DELTA - 0.005)
        elif key == ord('w'): MIN_DELTA += 0.005
        elif key == ord('a'): MIN_SPEED = max(0.0, MIN_SPEED - 0.001)
        elif key == ord('s'): MIN_SPEED += 0.001
        elif key == ord('z'):
            WINDOW = max(3, WINDOW - 1); xs = deque(xs, maxlen=WINDOW)
        elif key == ord('x'):
            WINDOW = min(60, WINDOW + 1); xs = deque(xs, maxlen=WINDOW)
        elif key == ord('e'): EMA_ALPHA = min(0.95, EMA_ALPHA + 0.05)
        elif key == ord('r'): EMA_ALPHA = max(0.05, EMA_ALPHA - 0.05)
        elif key == ord('d'): VEL_ALPHA = min(0.95, VEL_ALPHA + 0.05)
        elif key == ord('f'): VEL_ALPHA = max(0.05, VEL_ALPHA - 0.05)
        elif key == ord('c'): COOLDOWN_FRAMES = max(0, COOLDOWN_FRAMES - 1)
        elif key == ord('v'): COOLDOWN_FRAMES += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    try:
        if WRITE_CSV:
            csv_f.close()
    except: pass