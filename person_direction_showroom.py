# person_direction_showroom.py
import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
from collections import deque
import socket, threading, json, time

# ============== TCP SERVER ==============
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5050

clients = []
clients_lock = threading.Lock()

def accept_loop():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((SERVER_HOST, SERVER_PORT))
    srv.listen(8)
    print(f"[TCP] Listening on {SERVER_HOST}:{SERVER_PORT}")
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with clients_lock:
            clients.append(conn)
        print(f"[TCP] Client connected: {addr}")

def broadcast(event: str, payload: dict | None = None):
    msg = {"type": "event", "value": event, "ts": time.time()}
    if payload:
        msg.update(payload)
    data = (json.dumps(msg) + "\n").encode("utf-8")

    to_remove = []
    with clients_lock:
        for c in clients:
            try:
                c.sendall(data)
            except Exception:
                to_remove.append(c)
        for c in to_remove:
            try:
                c.close()
            except:
                pass
            clients.remove(c)

# окрема нитка для accept()
threading.Thread(target=accept_loop, daemon=True).start()

# ============== DETECTOR (напрямок + кроки) ==============
W, H, FPS = 640, 480, 30
MAX_VALID_DISTANCE_M = 3.0
ABSENCE_FRAMES = 12

EMA_ALPHA = 0.25
VEL_ALPHA = 0.4
VEL_DEADBAND = 0.003
MIN_SPEED = 0.004
WINDOW = 15
MIN_DELTA = 0.06
COOLDOWN_FRAMES = 18

DRAW_TRAIL = True
TRAIL_LEN = 25

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

baseline_x = None
shift_target = None
shift_confirm_frames = 0

def emit(event: str, extra: dict | None = None):
    print(f"EVENT: {event} {extra if extra else ''}")
    broadcast(event, extra)

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            continue

        color_img = np.asanyarray(color.get_data())
        depth_img  = np.asanyarray(depth.get_data())

        # MediaPipe
        rgb_for_mp = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb_for_mp)

        person_present = False
        x_norm = None
        y_norm = None
        dist_m = 0.0

        if res.pose_landmarks:
            lh = res.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            rh = res.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            x_norm = float((lh.x + rh.x) / 2.0)
            y_norm = float((lh.y + rh.y) / 2.0)

            x_px = int(np.clip(x_norm * W, 0, W - 1))
            y_px = int(np.clip(y_norm * H, 0, H - 1))
            dist_m = depth.get_distance(x_px, y_px)

            if 0.1 < dist_m < MAX_VALID_DISTANCE_M:
                person_present = True
                cv2.circle(color_img, (x_px, y_px), 6, (0, 255, 255), -1)
                cv2.putText(color_img, f"{dist_m:.2f} m", (x_px+8, y_px-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)

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

            if DRAW_TRAIL and y_norm is not None:
                trail.append((int(ema_x * W), int(y_norm * H)))
                for i in range(1, len(trail)):
                    cv2.line(color_img, trail[i-1], trail[i], (0, 255, 0), 2)

            if cooldown > 0:
                cooldown -= 1
            else:
                if len(xs) >= 2:
                    delta = xs[-1] - xs[0]  # >0 вправо; <0 вліво
                    speed = ema_v
                    if abs(speed) < VEL_DEADBAND:
                        speed = 0.0

                    moved_right = (delta > MIN_DELTA) and (speed > MIN_SPEED)
                    moved_left  = (delta < -MIN_DELTA) and (speed < -MIN_SPEED)

                    if moved_right:
                        emit("MOVE_LEFT", {"x": round(ema_x, 3), "speed": round(speed, 3)})
                        cooldown = COOLDOWN_FRAMES
                        xs.clear()
                    elif moved_left:
                        emit("MOVE_RIGHT", {"x": round(ema_x, 3), "speed": round(speed, 3)})
                        cooldown = COOLDOWN_FRAMES
                        xs.clear()

            cv2.putText(color_img, f"x={ema_x:.3f} v={ema_v:+.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2, cv2.LINE_AA)
        else:
            no_person_ctr += 1
            if no_person_ctr == ABSENCE_FRAMES:
                emit("NO_PERSON")
                ema_x = None
                ema_v = 0.0
                xs.clear()
                trail.clear()

            cv2.putText(color_img, "NO PERSON",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Showroom | Direction (color)", color_img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
