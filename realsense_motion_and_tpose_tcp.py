# realsense_motion_and_tpose_tcp.py
# RealSense + MediaPipe Pose
# 1) Рух ліво/право (EMA/velocity) -> TCP JSON події
# 2) T-pose (калібрований з tpose_calibration.json, або простий fallback)
# Дзеркальне відображення, оверлеї, хоткеї.

import cv2, numpy as np, pyrealsense2 as rs, mediapipe as mp
from collections import deque
from queue import Queue, Empty
import socket, time, json, os, threading, math

# =============== TCP (локально) ===============
TCP_HOST = "127.0.0.1"
TCP_PORT = 5050

class TcpSink:
    """
    Простіший TCP-сервер:
    - слухає 127.0.0.1:5050
    - приймає одного клієнта
    - send(line) відправляє JSON-рядок \n
    - при розриві з'єднання акуратно повертається в стан очікування клієнта
    """
    def __init__(self, host=TCP_HOST, port=TCP_PORT):
        self.host = host
        self.port = port
        self.srv = None
        self.cli = None
        self.cli_addr = None
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # миттєвий ребінд при рестартах
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((self.host, self.port))
        self.srv.listen(1)
        self.srv.settimeout(0.2)  # щоби не блокувати цикл
        self.running = True
        print(f"[TCP] Listening on {self.host}:{self.port}")

    def _accept_once(self):
        try:
            conn, addr = self.srv.accept()
            conn.settimeout(0.0)  # неблокуючий
            with self.lock:
                # закриваємо старого клієнта, якщо був
                if self.cli:
                    try: self.cli.close()
                    except: pass
                self.cli = conn
                self.cli_addr = addr
            print(f"[TCP] Client connected: {addr}")
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[TCP] accept() error: {e}")

    def send_json(self, obj: dict):
        line = (json.dumps(obj) + "\n").encode("utf-8")
        with self.lock:
            if not self.cli:
                return False
            try:
                self.cli.sendall(line)
                return True
            except (BrokenPipeError, ConnectionResetError, OSError):
                # клієнт відвалився
                try: self.cli.close()
                except: pass
                self.cli = None
                self.cli_addr = None
                print("[TCP] Client disconnected")
                return False

    def step(self):
        # пробуємо прийняти клієнта, якщо його немає
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

def send_event(event: str, **payload):
    msg = {"type":"event","value":event,"ts":time.time()}
    if payload: msg.update(payload)
    ok = tcp.send_json(msg)
    log = f"[TCP] => {event} {payload if payload else ''}"
    if ok:
        print(log)
    else:
        # без клієнта просто лог — камера продовжує працювати
        print(log + "  (no client)")

# =============== POSE WORKER CONFIG ===============
POSE_INPUT_SIZE = (320, 240)
POSE_QUEUE_SIZE = 2

class PoseWorker(threading.Thread):
    """Асинхронний воркер для MediaPipe Pose (щоб зняти затримку з основного циклу)."""
    def __init__(self, input_size, queue_size=POSE_QUEUE_SIZE):
        super().__init__(daemon=True)
        self.input_size = tuple(input_size)
        self.queue = Queue(maxsize=queue_size)
        self.latest = (None, -1)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def submit(self, frame_id: int, frame_bgr: np.ndarray):
        """Пробує додати кадр до черги, якщо вона не переповнена."""
        if self.stop_event.is_set():
            return
        try:
            self.queue.put_nowait((frame_id, frame_bgr))
        except:
            # якщо черга переповнена — просто скидаємо кадр
            pass

    def latest_result(self):
        with self.lock:
            return self.latest

    def stop(self):
        self.stop_event.set()
        try:
            self.queue.put_nowait((None, None))
        except:
            pass

    def run(self):
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.queue.get(timeout=0.1)
                except Empty:
                    continue

                if item is None:
                    self.queue.task_done()
                    break
                frame_id, frame_bgr = item
                if frame_id is None or frame_bgr is None:
                    self.queue.task_done()
                    continue

                h, w = frame_bgr.shape[:2]
                if (w, h) != self.input_size:
                    frame_small = cv2.resize(frame_bgr, self.input_size, interpolation=cv2.INTER_AREA)
                else:
                    frame_small = frame_bgr
                rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                with self.lock:
                    self.latest = (res, frame_id)
                self.queue.task_done()
        finally:
            pose.close()

# =============== STREAM ===============
W, H, FPS = 640, 480, 30
MAX_VALID_DISTANCE_M = 4.0
ABSENCE_FRAMES = 12

# =============== MOTION PARAMS (ліва/права) ===============
EMA_ALPHA = 0.35
VEL_ALPHA = 0.6
VEL_DEADBAND = 0.0015
MIN_SPEED = 0.006
WINDOW = 10
MIN_DELTA = 0.06
COOLDOWN_FRAMES = 10

DRAW_TRAIL = False
TRAIL_LEN  = 25

# =============== T-POSE (fallback) ===============
TPOSE_EXTEND_THR = 0.25   # мін. розліт рук (r_wrist.x - l_wrist.x)
TPOSE_Y_THR      = 0.15   # |y(wrist) - y(лінії плечей)|

# =============== T-POSE (calibrated) ===============
CAL_FILE         = "tpose_calibration.json"
EXT_TOL          = 0.08   # | |ext|-|ext_ref| | <= EXT_TOL
Y_TOL            = 0.06   # |dw - dw_ref| <= Y_TOL
MIN_TPOSE_FRAMES = 4      # антифлікер

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

# =============== INIT PIPELINE ===============
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
profile = pipeline.start(cfg)

depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
depth_intr = depth_stream.get_intrinsics()
color_intr = color_stream.get_intrinsics()
depth_to_color = depth_stream.get_extrinsics_to(color_stream)
color_to_depth = color_stream.get_extrinsics_to(depth_stream)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
DEPTH_MIN = 0.15

mp_pose = mp.solutions.pose
pose_worker = PoseWorker(POSE_INPUT_SIZE)
pose_worker.start()

MIRROR_VIEW = True

ema_x = None
ema_v = 0.0
xs = deque(maxlen=WINDOW)
cooldown = 0
no_person_ctr = 0
trail = deque(maxlen=TRAIL_LEN)
frame_id = 0
last_move_event = ""
tpose_streak = 0
last_tpose_state = None
fps_counter = 0
fps_timer = time.time()
fps_value = 0.0
pose_landmarks = None
last_pose_frame_id = -1

def put(img, text, org, color=(0,255,255), scale=0.7, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def depth_at_point(depth_frame, x_norm, y_norm):
    if depth_frame is None or x_norm is None or y_norm is None:
        return 0.0
    x_clamped = float(np.clip(x_norm, 0.0, 1.0))
    y_clamped = float(np.clip(y_norm, 0.0, 1.0))
    if MIRROR_VIEW:
        color_x = (1.0 - x_clamped) * W
    else:
        color_x = x_clamped * W
    color_y = y_clamped * H
    mapped = rs.rs2_project_color_pixel_to_depth_pixel(
        depth_frame, depth_scale, DEPTH_MIN, MAX_VALID_DISTANCE_M,
        color_intr, depth_intr, color_to_depth, depth_to_color,
        [color_x, color_y]
    )
    if not mapped or any(not math.isfinite(v) for v in mapped):
        dx = int(np.clip(color_x, 0, depth_intr.width - 1))
        dy = int(np.clip(color_y, 0, depth_intr.height - 1))
        return depth_frame.get_distance(dx, dy)
    dx, dy = int(mapped[0]), int(mapped[1])
    if 0 <= dx < depth_intr.width and 0 <= dy < depth_intr.height:
        return depth_frame.get_distance(dx, dy)
    return 0.0

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            tcp.step()
            continue
        frame_id += 1

        color_img = np.asanyarray(color.get_data())
        if MIRROR_VIEW:
            color_img = cv2.flip(color_img, 1)

        pose_input = cv2.resize(color_img, POSE_INPUT_SIZE, interpolation=cv2.INTER_AREA)
        pose_worker.submit(frame_id, pose_input)

        pose_res, pose_res_frame_id = pose_worker.latest_result()
        if pose_res_frame_id != -1 and pose_res_frame_id != last_pose_frame_id:
            last_pose_frame_id = pose_res_frame_id
            pose_landmarks = pose_res.pose_landmarks.landmark if pose_res and pose_res.pose_landmarks else None

        person_present = False
        x_norm = y_norm = None
        dist_m = 0.0
        dwL = dwR = 0.0
        extend_val = 0.0

        if pose_landmarks:
            lm = pose_landmarks
            LSH = mp_pose.PoseLandmark.LEFT_SHOULDER
            RSH = mp_pose.PoseLandmark.RIGHT_SHOULDER
            LH  = mp_pose.PoseLandmark.LEFT_HIP
            RH  = mp_pose.PoseLandmark.RIGHT_HIP
            LW  = mp_pose.PoseLandmark.LEFT_WRIST
            RW  = mp_pose.PoseLandmark.RIGHT_WRIST
            LE  = mp_pose.PoseLandmark.LEFT_ELBOW
            RE  = mp_pose.PoseLandmark.RIGHT_ELBOW

            lh = lm[LH]; rh = lm[RH]
            x_norm = float((lh.x + rh.x) / 2.0)
            y_norm = float((lh.y + rh.y) / 2.0)

            dist_m = depth_at_point(depth, x_norm, y_norm)
            if 0.1 < dist_m < MAX_VALID_DISTANCE_M:
                person_present = True
                hip_x = int(np.clip((1.0 - x_norm) * W if MIRROR_VIEW else x_norm * W, 0, W-1))
                hip_y = int(np.clip(y_norm * H, 0, H-1))
                cv2.circle(color_img, (hip_x, hip_y), 6, (0,255,255), -1)

            ls = lm[LSH]; rs_ = lm[RSH]
            lw = lm[LW];  rw  = lm[RW]
            le = lm[LE];  re  = lm[RE]

            shoulder_y = 0.5 * (ls.y + rs_.y)
            dwL = lw.y - shoulder_y
            dwR = rw.y - shoulder_y
            extend_val = rw.x - lw.x

        delta = 0.0
        speed = 0.0
        if person_present and x_norm is not None:
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
                if len(trail) > 1:
                    pts = np.array(trail, dtype=np.int32)
                    cv2.polylines(color_img, [pts], False, (0,255,0), 2)

            if cooldown > 0:
                cooldown -= 1
            else:
                moved_right = (delta > MIN_DELTA) and (speed > MIN_SPEED)
                moved_left  = (delta < -MIN_DELTA) and (speed < -MIN_SPEED)
                if moved_right:
                    send_event("MOVE_RIGHT", x=round(ema_x,3), speed=round(speed,3),
                               delta=round(delta,3), dist=round(dist_m,2))
                    last_move_event = "MOVE_RIGHT"
                    cooldown = COOLDOWN_FRAMES
                    xs.clear()
                elif moved_left:
                    send_event("MOVE_LEFT", x=round(ema_x,3), speed=round(speed,3),
                               delta=round(delta,3), dist=round(dist_m,2))
                    last_move_event = "MOVE_LEFT"
                    cooldown = COOLDOWN_FRAMES
                    xs.clear()
        else:
            no_person_ctr += 1
            if no_person_ctr == ABSENCE_FRAMES:
                send_event("NO_PERSON")
                ema_x = None
                ema_v = 0.0
                xs.clear()
                trail.clear()
                last_move_event = ""

        is_tpose_frame = False
        if pose_landmarks:
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

        curr_tpose_state = "ON" if is_tpose else "OFF"
        if curr_tpose_state != last_tpose_state:
            send_event("TPOSE_ON" if is_tpose else "TPOSE_OFF",
                       extend=round(extend_val,3), dwL=round(dwL,3), dwR=round(dwR,3))
            last_tpose_state = curr_tpose_state

        fps_counter += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps_value = fps_counter / (now - fps_timer)
            fps_counter = 0
            fps_timer = now

        put(color_img, f"x={0.0 if ema_x is None else ema_x:.3f} d={delta:+.3f} v={speed:+.3f} fps={fps_value:4.1f}",
            (10, 28), (50,220,50), 0.7, 2)
        if person_present and dist_m:
            put(color_img, f"dist={dist_m:.2f}m", (10, 52), (0,255,255), 0.6, 2)
        if last_move_event:
            put(color_img, f"last move: {last_move_event}", (10, 76), (0,200,255), 0.7, 2)
        put(color_img, f"T-Pose {'ON' if is_tpose else ('...' if is_tpose_frame else 'OFF')}",
            (10, 100), (0,255,0) if is_tpose else ((0,200,255) if is_tpose_frame else (0,0,255)), 0.75, 2)
        put(color_img, "Hotkeys: q/w Δthr | a/s Δspeed | z/x win | e/r ema | d/f vel | c/v cooldown | u/j ext | i/k y",
            (10, H-24), (200,200,200), 0.55, 1)

        cv2.imshow("RealSense Motion & T-pose", color_img)

        tcp.step()

        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
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
        elif key == ord('u'): EXT_TOL = min(0.5, EXT_TOL + 0.01)
        elif key == ord('j'): EXT_TOL = max(0.0, EXT_TOL - 0.01)
        elif key == ord('i'): Y_TOL   = min(0.5, Y_TOL   + 0.01)
        elif key == ord('k'): Y_TOL   = max(0.0, Y_TOL   - 0.01)

finally:
    try: pose_worker.stop()
    except: pass
    try: pose_worker.join(timeout=1.0)
    except: pass
    try: pipeline.stop()
    except: pass
    try: tcp.stop()
    except: pass
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")
