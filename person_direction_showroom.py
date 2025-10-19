# person_direction_showroom.py
import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
from collections import deque
import socket, threading, json, time, os

# ============== TCP============
TCP_HOST = "127.0.0.1"
TCP_PORT = 5050

class TcpSink:
    def __init__(self, host=TCP_HOST, port=TCP_PORT):
        self.host = host
        self.port = port
        self.srv = None
        self.cli = None
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((self.host, self.port))
        self.srv.listen(1)
        self.srv.settimeout(0.2)
        self.running = True
        print(f"[TCP] Listening on {self.host}:{self.port}")

    def _accept_once(self):
        try:
            conn, addr = self.srv.accept()
            conn.settimeout(0.0)
            with self.lock:
                if self.cli:
                    try: self.cli.close()
                    except: pass
                self.cli = conn
            print(f"[TCP] Client connected: {addr}")
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[TCP] accept() error: {e}")

    def send_json(self, obj: dict):
        payload = (json.dumps(obj) + "\n").encode("utf-8")
        with self.lock:
            if not self.cli:
                return False
            try:
                self.cli.sendall(payload)
                return True
            except (BrokenPipeError, ConnectionResetError, OSError):
                try: self.cli.close()
                except: pass
                self.cli = None
                print("[TCP] Client disconnected")
                return False

    def step(self):
        if self.running and self.cli is None:
            self._accept_once()

    def stop(self):
        self.running = False
        with self.lock:
            try:
                if self.cli:
                    self.cli.close()
            except: pass
            try:
                if self.srv:
                    self.srv.close()
            except: pass
        self.cli = None
        self.srv = None
        print("[TCP] Stopped")

tcp = TcpSink()
tcp.start()

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

# =============== T-POSE SETTINGS ===============
TPOSE_EXTEND_THR = 0.25
TPOSE_Y_THR      = 0.15

CAL_FILE         = "tpose_calibration.json"
EXT_TOL          = 0.08
Y_TOL            = 0.06
MIN_TPOSE_FRAMES = 4

calibrated = False
EXTEND_REF = None
DWL_REF    = None
DWR_REF    = None
if os.path.exists(CAL_FILE):
    try:
        with open(CAL_FILE, "r") as f:
            data = json.load(f)
        EXTEND_REF = float(data["extend_ref"])
        DWL_REF    = float(data["dwL_ref"])
        DWR_REF    = float(data["dwR_ref"])
        calibrated = True
        print(f"[CAL] Loaded: |ext|={abs(EXTEND_REF):.3f} dwL={DWL_REF:+.3f} dwR={DWR_REF:+.3f}")
    except Exception as e:
        print("[CAL] Failed to load calibration:", e)

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
tpose_streak = 0
last_tpose_state = None

def send_event(event: str, **payload):
    msg = {"type": "event", "value": event, "ts": time.time()}
    if payload:
        msg.update(payload)
    ok = tcp.send_json(msg)
    log = f"[TCP] => {event} {payload if payload else ''}"
    if ok:
        print(log)
    else:
        print(log + " (no client)")

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
        extend_val = 0.0
        dwL = dwR = 0.0

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

            ls = res.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            rs_ = res.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            lw = res.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            rw = res.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            shoulder_y = 0.5 * (ls.y + rs_.y)
            dwL = lw.y - shoulder_y
            dwR = rw.y - shoulder_y
            extend_val = rw.x - lw.x

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
                        send_event("MOVE_LEFT", x=round(ema_x, 3), speed=round(speed, 3))
                        cooldown = COOLDOWN_FRAMES
                        xs.clear()
                    elif moved_left:
                        send_event("MOVE_RIGHT", x=round(ema_x, 3), speed=round(speed, 3))
                        cooldown = COOLDOWN_FRAMES
                        xs.clear()

            cv2.putText(color_img, f"x={ema_x:.3f} v={ema_v:+.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2, cv2.LINE_AA)
        else:
            no_person_ctr += 1
            if no_person_ctr == ABSENCE_FRAMES:
                send_event("NO_PERSON")
                ema_x = None
                ema_v = 0.0
                xs.clear()
                trail.clear()

            cv2.putText(color_img, "NO PERSON",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        # ---------- T-POSE ----------
        is_tpose_frame = False
        if res.pose_landmarks:
            if calibrated and EXTEND_REF is not None:
                ext_ok = abs(abs(extend_val) - abs(EXTEND_REF)) <= EXT_TOL
                y_ok   = (abs(dwL - DWL_REF) <= Y_TOL) and (abs(dwR - DWR_REF) <= Y_TOL)
                is_tpose_frame = ext_ok and y_ok
            else:
                is_tpose_frame = (extend_val > TPOSE_EXTEND_THR and
                                  abs(dwL) < TPOSE_Y_THR and abs(dwR) < TPOSE_Y_THR)

        if is_tpose_frame:
            tpose_streak += 1
        else:
            tpose_streak = max(0, tpose_streak - 1)
        is_tpose = tpose_streak >= MIN_TPOSE_FRAMES

        curr_state = "ON" if is_tpose else "OFF"
        if curr_state != last_tpose_state:
            send_event("TPOSE_ON" if is_tpose else "TPOSE_OFF",
                       extend=round(extend_val, 3), dwL=round(dwL, 3), dwR=round(dwR, 3))
            last_tpose_state = curr_state

        if res.pose_landmarks:
            label = "T-POSE ✅" if is_tpose else ("T-POSE …" if is_tpose_frame else "T-POSE ❌")
            color = (0,255,0) if is_tpose else ((0,200,255) if is_tpose_frame else (0,0,255))
            cv2.putText(color_img, label, (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            if calibrated and EXTEND_REF is not None:
                cv2.putText(
                    color_img,
                    f"|ext|={abs(extend_val):.3f} ref={abs(EXTEND_REF):.3f}±{EXT_TOL:.2f} "
                    f"dwL={dwL:+.3f}/{DWL_REF:+.3f} dwR={dwR:+.3f}/{DWR_REF:+.3f}",
                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA
                )
            else:
                cv2.putText(
                    color_img,
                    f"ext>{TPOSE_EXTEND_THR:.2f} |dw|<{TPOSE_Y_THR:.2f}",
                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA
                )

        cv2.imshow("Showroom | Direction (color)", color_img)
        tcp.step()

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    try: tcp.stop()
    except: pass
    pipeline.stop()
    cv2.destroyAllWindows()
