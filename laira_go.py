#!/usr/bin/env python3
import asyncio
import json
import time
import cv2
import numpy as np
import threading
import serial
import lewansoul_lx16a
import aiohttp
import websockets
from av import VideoFrame
from aiortc import (
    VideoStreamTrack,
    MediaStreamError,
    RTCPeerConnection,
    RTCConfiguration,
    RTCIceServer,
    RTCSessionDescription
)
from aiortc.contrib.signaling import candidate_from_sdp
from aiortc.contrib.media import MediaPlayer
import os

# ─── Monkey‐patch asyncio DatagramTransport._fatal_error ───────────────
for module_name, cls_name in (
    ("asyncio.selector_events", "_SelectorDatagramTransport"),
    ("asyncio.proactor_events", "_ProactorDatagramTransport"),
):
    try:
        m = __import__(module_name, fromlist=[cls_name])
        cls = getattr(m, cls_name)
        if hasattr(cls, "_fatal_error"):
            _orig = cls._fatal_error
            def _safe(self, msg, exc, _orig=_orig):
                loop = getattr(self, "_loop", None)
                if not loop:
                    return
                return _orig(self, msg, exc)
            cls._fatal_error = _safe
    except Exception:
        pass

# ============== CONFIGURATION ==============
WS_URL = "wss://laira.onrender.com/laiRA"
TURN_URL = (
    "https://laira.metered.live/api/v1/turn/credentials?"
    "apiKey=cc105e937e35e7e785bafba97c04621dea2a"
)

# Motor control configuration
running_motors = True  # Set to False to disable motor control (simulation mode)
SERIAL_PORT = '/dev/ttyUSB0'
BAUD = 115200
MAX_RETRY = 30
RETRY_BASE = 0.5

# ============== ROBOT CONSTANTS ==============
LEG_PAIR_WIDTH = 168   # mm, distance between left/right legs
LEG_PAIR_LENGTH = 305  # mm, distance between front/back legs
BASE_HEIGHT = 150      # mm
L_HIP = 30             # mm (Hip offset)
L_UPPER = 120          # mm (Upper leg length)
L_LOWER = 150          # mm (Lower leg length)

# ============== GLOBAL VARIABLES ==============
# Control variables
keystates = {"w": False, "a": False, "s": False, "d": False, "q": False, "e": False}
keystates_lock = asyncio.Lock()

# Movement variables
max_z = 20
bounds = 30
base_stance_step = 1.0
gait_speed = 20.0
bringback_speed = 1.0
rotation_mode = False
reset_state = False
idle_timer = 0
idle_threshold = 0.3
in_animation = False
in_angle_mode = False

# Leg control
leg_target = {1: [L_HIP, 0, 0],
              2: [L_HIP, 0, 0],
              3: [L_HIP, 0, 0],
              4: [L_HIP, 0, 0]}
leg_backstep = {1: False, 2: False, 3: False, 4: False}
leg_keys = {
    1: {'forward': False, 'backward': False, 'left': False, 'right': False},
    2: {'forward': False, 'backward': False, 'left': False, 'right': False},
    3: {'forward': False, 'backward': False, 'left': False, 'right': False},
    4: {'forward': False, 'backward': False, 'left': False, 'right': False}
}

# Freeze logic
freeze_pair_y = False
uhjk_freeze_arrow = False
uhjk_freeze_arrow_LR = False

# Motor configuration
motor_nums = [[1,2,3,4,5],[11,13,12,14,15],[21,22,23,25,24],[31,32,33,35,34]]
motor_adjusts = [[50,-70,0],[60,20,0],[-15,0,60],[0,30,80]]
angle_buffer = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

# Serial control
ctrl = None
bus_ok = False
bus_lock = threading.Lock()

# WebSocket health monitoring
ws_last_message_time = time.time()
ws_connection = None
ws_healthy = True
WS_TIMEOUT = 30  # Consider unhealthy after 30 seconds of no messages
WS_PING_INTERVAL = 10  # Send ping every 10 seconds

# ICE connection state
ice_active = True
last_ice_active_time = time.time()
motors_resting = False
SERVO_ID_ALL = 0xFE
off_due_to_disconnect = False

# ============== SERIAL COMMUNICATION ==============
def _open_port():
    if not running_motors:
        return None
    return serial.Serial(SERIAL_PORT, BAUD, timeout=1)

def _build_ctrl():
    return lewansoul_lx16a.ServoController(_open_port())

def _try_init():
    global ctrl, bus_ok
    try:
        new = _build_ctrl()
        with bus_lock:
            if ctrl and hasattr(ctrl, "serial"):
                try:
                    ctrl.serial.close()
                except Exception:
                    pass
            ctrl = new
            bus_ok = True
        print(f"[motor-bus] online via {SERIAL_PORT}")
        return True
    except Exception as e:
        print(f"[motor-bus] init fail: {e}")
        return False

def _auto_reconnector():
    delay = RETRY_BASE
    attempts = 0
    while attempts < MAX_RETRY and not _try_init():
        attempts += 1
        time.sleep(delay)
        delay = min(delay * 1.5, 5)

# Initialize serial connection
if running_motors:
    _auto_reconnector()
if not bus_ok:
    print("[motor-bus] running in simulation mode")

def ensure_bus():
    global bus_ok
    if bus_ok:
        return True
    if bus_ok is False:
        bus_ok = "pending"
        threading.Thread(target=_auto_reconnector, daemon=True).start()
    return False

# ============== MOTOR MAPPING ==============
amplitude_motor1 = 1.55

def linear_map(x, x0, y0, x1, y1):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def map_angle1(angle):
    a = max(0, min(angle, 180))
    amp = amplitude_motor1
    if a <= 90:
        base = linear_map(a, 0, 900, 90, 500)
    else:
        base = linear_map(a, 90, 500, 180, 100)
    return int(500 + amp * (base - 500))

def map_angle2(angle):
    a = max(90, min(angle, 270))
    if a <= 180:
        return linear_map(a, 90, 100, 180, 500)
    else:
        return linear_map(a, 180, 500, 270, 900)

def map_angle3(angle):
    a = max(0, min(angle, 180))
    if a <= 90:
        return linear_map(a, 0, 100, 90, 500)
    else:
        return linear_map(a, 90, 500, 180, 900)

def send_motor_commands(theta1, theta2, theta3, legnum, motor_speed=0, raw_1 = False, raw_2 = False, raw_3 = False):
    global bus_ok, reset_state
    
    if not running_motors:
        return
    if ensure_bus() is False:
        print("[motor_cmd] Bus not ready, skipping")
        return
    
    if reset_state:
        motor_speed = 100
    
    hip = map_angle1(theta1)
    upper = map_angle2(theta2)
    lower = map_angle3(theta3)
    if raw_3 == False or raw_2 == False or raw_1 == False:
        try:
            with bus_lock:
                ctrl.move(motor_nums[legnum][0], hip + motor_adjusts[legnum][0], motor_speed)
                ctrl.move(motor_nums[legnum][1], upper + motor_adjusts[legnum][1], motor_speed)
                ctrl.move(motor_nums[legnum][2], 500 - (motor_adjusts[legnum][1] + upper - 500), motor_speed)
                ctrl.move(motor_nums[legnum][3], lower + motor_adjusts[legnum][2], motor_speed)
                ctrl.move(motor_nums[legnum][4], 500 - (motor_adjusts[legnum][2] + lower - 500), motor_speed)
        except (serial.SerialException, OSError) as e:
            print(f"[motor-bus] lost connection → {e}")
            bus_ok = False
    else:
        try:
            with bus_lock:
                ctrl.move(motor_nums[legnum][0], raw_1, motor_speed)
                ctrl.move(motor_nums[legnum][1], raw_2, motor_speed)
                ctrl.move(motor_nums[legnum][2], 500 - (raw_2), motor_speed)
                ctrl.move(motor_nums[legnum][3], raw_3, motor_speed)
                ctrl.move(motor_nums[legnum][4], 500 - (raw_3), motor_speed)
        except (serial.SerialException, OSError) as e:
            print(f"[motor-bus] lost connection → {e}")
            bus_ok = False

# Async wrapper for motor commands
async def send_motor_commands_async(theta1, theta2, theta3, legnum, motor_speed=0, raw_1=False, raw_2=False, raw_3=False):
    """Async wrapper for motor commands to prevent blocking"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        send_motor_commands,
        theta1, theta2, theta3, legnum, motor_speed, raw_1, raw_2, raw_3
    )

# ============== KINEMATICS ==============
def RotY(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def RotX(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def compute_leg_candidates(F, A):
    Q = RotY(-A) @ F
    P_chain = Q - np.array([L_HIP, 0, 0])
    X_target = P_chain[2]
    Y_target = -P_chain[1]
    D = np.hypot(X_target, Y_target)
    if D > (L_UPPER + L_LOWER) or D < abs(L_UPPER - L_LOWER):
        return []
    cos_v = (D**2 - L_UPPER**2 - L_LOWER**2) / (2 * L_UPPER * L_LOWER)
    cos_v = np.clip(cos_v, -1, 1)
    v_eff = np.arccos(cos_v)
    phi2 = np.arctan2(Y_target, X_target)
    beta = np.arctan2(L_LOWER * np.sin(v_eff), L_UPPER + L_LOWER * np.cos(v_eff))
    theta2_1 = (np.degrees(phi2 - beta) + 90) % 360
    theta3_1 = np.degrees(v_eff) % 360
    theta2_2 = (np.degrees(phi2 + beta) + 90) % 360
    theta3_2 = (-np.degrees(v_eff)) % 360
    return [(theta2_1, theta3_1), (theta2_2, theta3_2)]

def inverse_kinematics(x, y, z):
    F = np.array([x, y, z - BASE_HEIGHT])
    R_val = np.hypot(x, z - BASE_HEIGHT)
    if R_val < L_HIP:
        return None, None, None
    phi = np.arctan2(z - BASE_HEIGHT, x)
    try:
        alpha = np.arccos(L_HIP / R_val)
    except ValueError:
        return None, None, None
    A1 = alpha - phi
    A2 = -alpha - phi
    candidates = []
    for A in [A1, A2]:
        theta1_candidate = (np.degrees(A) + 90) % 360
        for sol in compute_leg_candidates(F, A):
            candidates.append((theta1_candidate, sol[0], sol[1]))
    if not candidates:
        sols = compute_leg_candidates(F, A1)
        return ((np.degrees(A1) + 90) % 360, sols[0][0], sols[0][1]) if sols else (None, None, None)
    return min(candidates, key=lambda cand: abs(cand[0] - 90))

def inverse_kinematics_priority(x, y, z, tol=1e-3):
    if abs(x - L_HIP) < tol:
        fixed_theta1 = 90
        F = np.array([x, y, z - BASE_HEIGHT])
        sols = compute_leg_candidates(F, 0)
        return (fixed_theta1, sols[0][0], sols[0][1]) if sols else inverse_kinematics(x, y, z)
    return inverse_kinematics(x, y, z)

# ============== MOVEMENT CONTROL ==============
def get_normal_speed():
    global gait_speed
    return base_stance_step * gait_speed

def get_bringback_speed():
    global gait_speed
    return base_stance_step * bringback_speed * gait_speed

async def reset_function(reset_to_wasd=False, reset_to_rotation=False, reset_positions_only=True):
    global leg_target, leg_backstep, reset_state, rotation_mode, in_animation
    
    if not reset_state or in_animation:
        for i in leg_target:
            leg_target[i] = [L_HIP, 0, 0]
        for i in leg_backstep:
            leg_backstep[i] = False
        reset_state = True
        
        await asyncio.sleep(0.2)
        reset_state = False
        
        if not reset_positions_only:
            if reset_to_wasd:
                rotation_mode = False
            elif reset_to_rotation:
                rotation_mode = True

def update_simulation_leg(target, keys, backstep_state, leg_num):
    global freeze_pair_y, uhjk_freeze_arrow_LR, uhjk_freeze_arrow, idle_timer, reset_state, gait_speed
    neutral_x = L_HIP
    x, y, z = target
    s_bound = bounds
    right_edge = x >= neutral_x + s_bound
    left_edge = x <= neutral_x - s_bound
    top_edge = y <= -bounds
    bottom_edge = y >= bounds
    isarrow = (leg_num == 2 or leg_num == 3)
    
    # Dynamic freeze logic
    if leg_num == 2:
        if not rotation_mode:
            if leg_backstep[1] and abs(y - leg_target[4][1]) == 0 and bounds <= abs(y):
                uhjk_freeze_arrow = True
            elif uhjk_freeze_arrow and not leg_backstep[1]:
                uhjk_freeze_arrow = False
            if leg_backstep[1] and abs(x - leg_target[4][0]) == 0 and s_bound <= abs(x - L_HIP):
                uhjk_freeze_arrow_LR = True
            elif uhjk_freeze_arrow_LR and not leg_backstep[1]:
                uhjk_freeze_arrow_LR = False
        else:
            if leg_backstep[1] and abs((x-30) + (leg_target[4][0] -30)) == 0 and s_bound <= abs(x - L_HIP):
                uhjk_freeze_arrow_LR = True
            elif uhjk_freeze_arrow_LR and not leg_backstep[1]:
                uhjk_freeze_arrow_LR = False
    
    # Edge detection and backstep control
    if right_edge:
        backstep_state = keys['right'] if not backstep_state else (False if keys['left'] else backstep_state)
    elif left_edge:
        backstep_state = keys['left'] if not backstep_state else (False if keys['right'] else backstep_state)
    if bottom_edge:
        backstep_state = keys['backward'] if not backstep_state else (False if keys['forward'] else backstep_state)
    elif top_edge:
        backstep_state = keys['forward'] if not backstep_state else (False if keys['backward'] else backstep_state)
    
    # Movement updates
    if backstep_state:
        if not (uhjk_freeze_arrow and isarrow):
            y += get_bringback_speed() if keys['forward'] else 0
            y -= get_bringback_speed() if keys['backward'] else 0
        if not (uhjk_freeze_arrow_LR and isarrow):
            x += get_bringback_speed() if keys['left'] else 0
            x -= get_bringback_speed() if keys['right'] else 0
    else:
        if not (uhjk_freeze_arrow and isarrow):
            y += get_normal_speed() if keys['backward'] else 0
            y -= get_normal_speed() if keys['forward'] else 0
        if not (uhjk_freeze_arrow_LR and isarrow):
            x += get_normal_speed() if keys['right'] else 0
            x -= get_normal_speed() if keys['left'] else 0
    
    # Z height control
    if backstep_state:
        progress = np.clip(np.hypot((x - neutral_x), y) / bounds, 0, 1)
        z = np.cos(np.deg2rad(90*progress)) * max_z
    else:
        z = 0
    
    # Diagonal correction
    if backstep_state and sum(keys.values()) == 1:
        if not (isarrow and uhjk_freeze_arrow):
            if keys['forward']:
                try:
                    x -= (x - L_HIP) / (bounds - y) * get_bringback_speed()
                except:
                    x = L_HIP
                if abs(x - L_HIP) < get_bringback_speed():
                    x = L_HIP
            if keys['backward']:
                try:
                    x -= (x - L_HIP) / (bounds + y) * get_bringback_speed()
                except:
                    x = L_HIP
                if abs(x - L_HIP) < get_bringback_speed():
                    x = L_HIP
        if not (isarrow and uhjk_freeze_arrow_LR):
            if keys['left']:
                try:
                    y -= y / (bounds - (x - L_HIP)) * get_bringback_speed()
                except:
                    y = 0
                if abs(y) < get_bringback_speed():
                    y = 0
            if keys['right']:
                try:
                    y -= y / (bounds + x) * get_bringback_speed()
                except:
                    y = 0
                if abs(y) < get_bringback_speed():
                    y = 0
    
    return [x, y, z], backstep_state

turn_mode = False
ai_mode = False

async def key_handling():
    global rotation_mode, leg_keys, reset_state, turn_mode, ai_mode, gait_speed
    
    if reset_state:
        return
    
    async with keystates_lock:
        current_keystates = keystates.copy()
    
    # Debug: print if we have any active keys
    active = [k for k, v in current_keystates.items() if v]
    if active:
        print(f"[key_handling] Active keys: {active}, Rotation mode: {rotation_mode}")
    
    # Check for mode transitions
    if (current_keystates.get('q', False) or current_keystates.get('e', False)) and not rotation_mode:
        return

    if (current_keystates.get('a', False) or current_keystates.get('d', False)) and not turn_mode:
        print("[key_handling] Switching to turn mode")
        turn_mode = True
        gait_speed = 10 if ai_mode else 20
        for leg in leg_keys:
            for direction in leg_keys[leg]:
                leg_keys[leg][direction] = False
        await reset_function(reset_to_wasd=True, reset_to_rotation=False, reset_positions_only=True)
        return
    
    if (current_keystates.get('w', False)  or 
        current_keystates.get('s', False)) and (rotation_mode or turn_mode):
        print("[key_handling] Switching to WASD mode")
        turn_mode = False
        gait_speed = 20
        for leg in leg_keys:
            for direction in leg_keys[leg]:
                leg_keys[leg][direction] = False
        await reset_function(reset_to_wasd=True, reset_to_rotation=False, reset_positions_only=False)
        return
    
    # Clear all leg keys
    for leg in leg_keys:
        for direction in leg_keys[leg]:
            leg_keys[leg][direction] = False
    
    # Update leg keys based on current mode
    if not rotation_mode:
        horizontal_pressed = current_keystates.get('a', False) or current_keystates.get('d', False)
        vertical_pressed = current_keystates.get('w', False) or current_keystates.get('s', False)
        
        if current_keystates.get('w', False) and not horizontal_pressed:
            print("[key_handling] Forward movement")
            for i in range(1, 5):
                leg_keys[i]['forward'] = True
        elif current_keystates.get('s', False) and not horizontal_pressed:
            print("[key_handling] Backward movement")
            for i in range(1, 5):
                leg_keys[i]['backward'] = True
        elif current_keystates.get('a', False) and not vertical_pressed:
            print("[key_handling] Left movement")
            leg_keys[1]['left'] = True
            leg_keys[2]['right'] = True
            leg_keys[3]['left'] = True
            leg_keys[4]['right'] = True
        elif current_keystates.get('d', False) and not vertical_pressed:
            print("[key_handling] Right movement")
            leg_keys[1]['right'] = True
            leg_keys[2]['left'] = True
            leg_keys[3]['right'] = True
            leg_keys[4]['left'] = True
    else:
        if current_keystates.get('q', False):
            print("[key_handling] Clockwise rotation")
            leg_keys[1]['left'] = True
            leg_keys[2]['right'] = True
            leg_keys[3]['right'] = True
            leg_keys[4]['left'] = True
        elif current_keystates.get('e', False):
            print("[key_handling] Counter-clockwise rotation")
            leg_keys[1]['right'] = True
            leg_keys[2]['left'] = True
            leg_keys[3]['left'] = True
            leg_keys[4]['right'] = True

async def run_animation(cmd, mode):
    global in_angle_mode, angle_buffer, leg_target
    if mode == 'ik':
        for i in range(4):
            leg_target[i+1] = cmd[i]
    elif mode == 'angle':
        in_angle_mode = True
        for i in range(4):
            angle_buffer[i] = cmd[i]
            await send_motor_commands_async(cmd[i][0], cmd[i][1], cmd[i][2], i, 200)
    elif mode == 'fastangle':
        in_angle_mode = True
        for i in range(4):
            angle_buffer[i] = cmd[i]
            await send_motor_commands_async(cmd[i][0], cmd[i][1], cmd[i][2], i, 0)

async def control_loop():
    """Main control loop that replaces matplotlib animation"""
    global idle_timer, reset_state, in_animation, in_angle_mode, angle_buffer, motors_resting, off_due_to_disconnect
    ICE_LOST_SHUTOFF_DELAY = 5  # seconds
    
    print("🎮 Control loop started")
    
    while True:
        # Process key inputs
        await key_handling()
        
        # Update each leg
        for leg_num in range(1, 5):
            old_target = leg_target[leg_num].copy()
            
            leg_target[leg_num], leg_backstep[leg_num] = update_simulation_leg(
                leg_target[leg_num], leg_keys[leg_num], leg_backstep[leg_num], leg_num
            )
            
            # Debug: print if position changed
            if old_target != leg_target[leg_num]:
                print(f"Leg {leg_num} moved: {old_target} -> {leg_target[leg_num]}")
            
            # Calculate inverse kinematics
            effective_target = leg_target[leg_num]
            ik = inverse_kinematics_priority(effective_target[0], effective_target[1], effective_target[2])
            
            if ik[0] is not None:
                if not in_angle_mode:
                    theta1, theta2, theta3 = ik
                else:
                    theta1, theta2, theta3 = angle_buffer[leg_num-1]
                
                if not in_angle_mode:
                    await send_motor_commands_async(theta1, theta2, theta3, leg_num-1)
        
        # Handle idle timeout
        epsilon = 1e-5
        all_default = all(abs(eff[0] - L_HIP) < epsilon and abs(eff[1]) < epsilon and abs(eff[2]) < epsilon 
                         for eff in leg_target.values())
        all_keys_off = all(not any(leg_keys[ln].values()) for ln in leg_keys)
        
        if all_default:
            idle_timer = 0
        elif not reset_state and all_keys_off and not any(keystates.values()) and not in_animation:
            idle_timer += 0.05
        else:
            idle_timer = 0
        
        if idle_timer >= idle_threshold and not all_default:
            await reset_function(reset_positions_only=True)
            idle_timer = 0
        
        # Control loop rate (20Hz, similar to matplotlib 50ms interval)
        await asyncio.sleep(0.05)

# ============== VIDEO STREAMING ==============
class USBVideoTrack(VideoStreamTrack):
    def __init__(self, device=0, backend=cv2.CAP_V4L2):
        super().__init__()
        self.cap = None
        for _ in range(3):
            cap = cv2.VideoCapture(device, backend)
            if cap.isOpened():
                self.cap = cap
                break
            cap.release()
            time.sleep(0.5)
        if not self.cap:
            print("⚠️ [USBVideoTrack] No camera – using test pattern")

    async def recv(self):
        if not self.cap:
            raise MediaStreamError
        pts, tb = await self.next_timestamp()
        ret, frame = await asyncio.get_event_loop().run_in_executor(None, self.cap.read)
        if not ret:
            raise MediaStreamError
        vf = VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts, vf.time_base = pts, tb
        return vf

    def stop(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        return super().stop()

class TestVideoTrack(VideoStreamTrack):
    def __init__(self, w=640, h=480):
        super().__init__()
        self.w, self.h = w, h
        self.c = 0

    async def recv(self):
        pts, tb = await self.next_timestamp()
        img = np.zeros((self.h, self.w, 3), np.uint8)
        img[:] = ((self.c*5)%256, (self.c*3)%256, (self.c*7)%256)
        vf = VideoFrame.from_ndarray(img, format="bgr24")
        vf.pts, vf.time_base = pts, tb
        self.c += 1
        return vf

# ============== AUDIO CAPTURE (webcam mic) ==============
def _make_audio_player():
    preferred = os.getenv("LAIRA_AUDIO_DEVICE")
    fmt = os.getenv("LAIRA_AUDIO_FORMAT") or "alsa"

    attempts = []
    if preferred:
        attempts.append((preferred, fmt))

    # Use the correct card/device ID
    attempts += [
        ("hw:2,0", "alsa"),
        ("plughw:2,0", "alsa"),
        ("hw:C960,0", "alsa"),
        ("plughw:C960,0", "alsa"),
        ("sysdefault:CARD=C960", "alsa"),
        ("default", "alsa"),
        ("default", "pulse"),
    ]

    for src, f in attempts:
        try:
            player = MediaPlayer(
                src,
                format=f,
                options={
                    "channels": "1",
                    "sample_rate": "48000",
                    "ac": "1",
                    "ar": "48000",
                }
            )
            if player and player.audio:
                print(f"[audio] using input: {src} (format={f})")
                return player
        except Exception as e:
            print(f"[audio] try {src} (format={f}) → {e}")

    print("[audio] no input found; starting without mic")
    return None







# ============== WEBSOCKET MESSAGE HANDLING ==============
async def handle_message(msg):
    """Process incoming WebSocket messages for robot control"""
    global keystates, in_animation, in_angle_mode, motors_resting, off_due_to_disconnect, ai_mode, gait_speed
    global ws_last_message_time
    
    # Update last message time
    ws_last_message_time = time.time()
    
    try:
        if 'message' not in msg:
            return
            
        message = msg['message']
        print(json.loads(message))
        
        if 'rest' in message:
            ctrl.motor_off(SERVO_ID_ALL)
            motors_resting = True
            return
        if 'shake' in message:
            ctrl.motor_off(14)
            ctrl.motor_off(15)
            motors_resting = True
            return

        if 'keystates' in message or 'ik' in message or 'angle' in message or 'aimode' in message or 'noaimode' in message:
            if motors_resting:
                ctrl.motor_on(SERVO_ID_ALL)
                motors_resting = False
                off_due_to_disconnect = False

            inner = json.loads(message)

            if 'noaimode' in message:
                ai_mode = False
                gait_speed = 20
                return
            elif 'aimode' in message:
                ai_mode = True
                gait_speed = 10
            
            if 'ik' in message:
                in_animation = True
                await run_animation(inner['ik'], 'ik')
                print("IK command:", inner['ik'])
            elif 'fastangle' in message:
                in_animation = True
                await run_animation(inner['fastangle'], 'fastangle')
                print("Fast Angle command:", inner['fastangle'])
            elif 'angle' in message:
                in_animation = True
                await run_animation(inner['angle'], 'angle')
                print("Angle command:", inner['angle'])
            elif 'keystates' in message:
                if in_animation:
                    for key, value in inner['keystates'].items():
                        if value:
                            await reset_function(reset_to_wasd=True, reset_to_rotation=False, reset_positions_only=False)
                            in_animation = False
                            in_angle_mode = False
                            break
                if not in_animation:
                    new_keystates = inner['keystates']
                    for key in new_keystates:
                        if isinstance(new_keystates[key], str):
                            new_keystates[key] = (new_keystates[key].lower() == 'true')
                    async with keystates_lock:
                        for key in new_keystates:
                            if key in keystates:
                                keystates[key] = new_keystates[key]
                    # print("Keystates updated:", keystates)
                    
    except Exception as e:
        print(f"Error handling message: {e}")

# ============== WEBSOCKET HEALTH MONITORING ==============
async def websocket_health_monitor():
    """Monitor WebSocket health and trigger reconnection if needed"""
    global ws_healthy, ws_last_message_time, ws_connection
    
    while True:
        try:
            current_time = time.time()
            time_since_last_message = current_time - ws_last_message_time
            
            if time_since_last_message > WS_TIMEOUT:
                if ws_healthy:
                    print(f"⚠️ WebSocket unhealthy - no messages for {time_since_last_message:.1f}s")
                    ws_healthy = False
                    if ws_connection and not ws_connection.closed:
                        print("🔄 Forcing WebSocket reconnection...")
                        await ws_connection.close()
            else:
                if not ws_healthy:
                    print("✅ WebSocket health restored")
                    ws_healthy = True
            
            if ws_connection and not ws_connection.closed and ws_healthy:
                try:
                    await ws_connection.ping()
                except Exception as e:
                    print(f"⚠️ Ping failed: {e}")
                    ws_healthy = False
            
            await asyncio.sleep(WS_PING_INTERVAL)
            
        except Exception as e:
            print(f"Error in health monitor: {e}")
            await asyncio.sleep(5)

# ============== MAIN WEBRTC/WEBSOCKET HANDLER ==============
async def make_pc(ice_servers):
    config = RTCConfiguration([RTCIceServer(**srv) for srv in ice_servers])
    pc = RTCPeerConnection(config)

    # Video track
    real = USBVideoTrack()
    track = real if real.cap else TestVideoTrack()
    pc.addTrack(track)
    pc._video_track = track

    # Audio track
    pc._audio_player = _make_audio_player()
    if pc._audio_player and pc._audio_player.audio:
        pc.addTrack(pc._audio_player.audio)

    @pc.on("iceconnectionstatechange")
    def ice_change():
        global ice_active, last_ice_active_time
        print("🔗 ICE state:", pc.iceConnectionState)
        if pc.iceConnectionState in ("connected", "completed"):
            ice_active = True
            last_ice_active_time = time.time()
        else:
            ice_active = False

    @pc.on("connectionstatechange")
    def conn_change():
        print("🔗 PC state:", pc.connectionState)

    return pc

async def websocket_handler():
    """Main WebSocket connection handler with improved error handling"""
    global ws_connection, ws_last_message_time, ws_healthy
    
    reconnect_delay = 1
    max_reconnect_delay = 30
    
    while True:
        ws_connection = None
        pc = None
        
        try:
            # Get TURN credentials
            async with aiohttp.ClientSession() as sess:
                resp = await sess.get(TURN_URL)
                ice_servers = await resp.json()

            pc = await make_pc(ice_servers)
            queue = []
            got_offer = False

            # Connect to broker
            async with websockets.connect(
                WS_URL,
                ping_interval=10,
                ping_timeout=10,
                close_timeout=5,
                max_size=10**7,
                compression=None
            ) as ws:
                ws_connection = ws
                ws_last_message_time = time.time()
                ws_healthy = True
                reconnect_delay = 1  # Reset delay on successful connection
                
                print("✅ WebSocket connected")
                await ws.send(json.dumps({"type": "laiRA-connected"}))

                @pc.on("icecandidate")
                async def send_ice(cand):
                    if cand and ws_connection and not ws_connection.closed:
                        try:
                            await ws_connection.send(json.dumps({
                                "type": "ice-candidate",
                                "candidate": cand.toJSON()
                            }))
                        except Exception as e:
                            print(f"Failed to send ICE candidate: {e}")

                # Message handling loop with timeout
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT)
                        ws_last_message_time = time.time()
                        
                    except asyncio.TimeoutError:
                        print(f"⚠️ WebSocket receive timeout after {WS_TIMEOUT}s")
                        break
                    except websockets.ConnectionClosed as e:
                        print(f"⚠️ WebSocket closed: {e}")
                        break
                    except Exception as e:
                        print(f"⚠️ WebSocket receive error: {e}")
                        break

                    # Process message
                    try:
                        msg = json.loads(raw)
                        msg_type = msg.get("type")
                        
                        if msg_type == "offer":
                            if not got_offer:
                                print("📬 Received OFFER → sending answer")
                                await pc.setRemoteDescription(
                                    RTCSessionDescription(sdp=msg["sdp"], type="offer")
                                )
                                got_offer = True
                                
                                # Process queued ICE candidates
                                for c in queue:
                                    ice = candidate_from_sdp(c["candidate"])
                                    ice.sdpMid, ice.sdpMLineIndex = c.get("sdpMid"), c.get("sdpMLineIndex")
                                    await pc.addIceCandidate(ice)
                                queue.clear()
                                
                                # Create and send answer
                                ans = await pc.createAnswer()
                                await pc.setLocalDescription(ans)
                                await ws.send(json.dumps({
                                    "type": pc.localDescription.type,
                                    "sdp": pc.localDescription.sdp
                                }))
                            else:
                                print("⚠️ Duplicate OFFER → reconnecting")
                                break

                        elif msg_type == "ice-candidate":
                            c = msg["candidate"]
                            ice = candidate_from_sdp(c["candidate"])
                            ice.sdpMid, ice.sdpMLineIndex = c.get("sdpMid"), c.get("sdpMLineIndex")
                            if got_offer:
                                await pc.addIceCandidate(ice)
                            else:
                                queue.append(c)

                        elif msg_type == "chat-message":
                            # Handle control messages
                            await handle_message(msg)
                            
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON: {e}")
                    except Exception as e:
                        print(f"Error processing message: {e}")

        except aiohttp.ClientError as e:
            print(f"⚠️ Failed to get TURN credentials: {e}")
        except Exception as e:
            print(f"⚠️ WebSocket error: {e}")

        finally:
            ws_connection = None
            ws_healthy = False
            
            # Clean up peer connection
            if pc:
                if hasattr(pc, "_video_track"):
                    pc._video_track.stop()
                # Stop audio capture if present
                if hasattr(pc, "_audio_player") and pc._audio_player:
                    try:
                        pc._audio_player.stop()
                    except Exception:
                        pass
                await pc.close()
            
            # Exponential backoff for reconnection
            print(f"⏳ Reconnecting in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

# ============== MAIN ENTRY POINT ==============
async def main():
    """Main async function that runs control loop, WebSocket handler, and health monitor"""
    print("🤖 Starting unified robot control system...")
    
    control_task = asyncio.create_task(control_loop())
    websocket_task = asyncio.create_task(websocket_handler())
    health_task = asyncio.create_task(websocket_health_monitor())
    
    await asyncio.gather(control_task, websocket_task, health_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
