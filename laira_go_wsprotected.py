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
import av
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
from aiortc.contrib.media import MediaPlayer, MediaRelay #added media relay [NEW]
from aiortc.mediastreams import AudioStreamTrack

import os
# Vosk was ripped out 2026-04-24 — replaced by openwakeword (wake word)
# + silero VAD (speech-end detection) + faster-whisper (command STT).
# See laira_stt.py for the new pipeline. Vosk's quality on "laira"
# specifically was terrible (consistently misheard as lyra/iris/higher/
# etc.) which made the wake word unreliable.
# ─── Load ~/laira_code/.env into os.environ (zero-dep) ─────────────────
# Minimal KEY=VALUE parser, no quoting/escape support beyond whitespace trim.
# The file is chmod 600 and contains T1_PROMPT_KEY, METERED_API_KEY,
# MODAL_TRACKER_URL, OPENAI_API_KEY, LAIRA_WAKE_MODEL, etc. Not committed.
#
# MUST run BEFORE `import laira_stt` (and any other project module that
# reads env vars at module-load time). laira_stt freezes seven settings
# from os.environ at import time — wake model name, silence thresholds,
# pre-roll length, STT backend, etc. — and if .env hasn't loaded yet,
# every one of them silently falls back to the hardcoded default. This
# was a real bug: LAIRA_WAKE_MODEL=lyra in .env was being ignored
# entirely because the import happened first.
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_ENV_PATH):
    try:
        with open(_ENV_PATH) as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith('#') or '=' not in _line:
                    continue
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip())
        print(f"[env] loaded {_ENV_PATH}")
    except Exception as _e:
        print(f"[env] failed to load {_ENV_PATH}: {_e}")

import laira_frames  # shared frame cache for the brain module
import laira_brain   # Pi-resident orchestrator (T1/T2/T3)
import laira_stt     # wake-word + command STT pipeline (reads env at import)

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

# Metered (TURN) API key now lives in ~/laira_code/.env as METERED_API_KEY.
# If the env var isn't populated we fall back to empty and log a warning; the
# TURN credentials endpoint will reject the request and aiortc will surface
# the failure, rather than silently running with an expired hardcoded key.
_METERED_API_KEY = os.environ.get("METERED_API_KEY", "")
if not _METERED_API_KEY:
    print("[env] WARNING: METERED_API_KEY not set; TURN credentials will fail")
TURN_URL = (
    "https://laira.metered.live/api/v1/turn/credentials?"
    f"apiKey={_METERED_API_KEY}"
)

# Motor control configuration
running_motors = True  # Set to False to disable motor control (simulation mode)
SERIAL_PORT = '/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0'
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

# (VOSK model load removed — replaced by laira_stt module which lazy-
# loads openwakeword + silero_vad + faster-whisper on first audio frame.
# Lazy load means the service starts in ~1s instead of blocking ~3s on
# Vosk acoustic model memory-map at boot.)

# --- global audio singletons ---
# Audio capture is decoupled from PC lifecycle: the audio pump task pulls
# frames from MediaPlayer.audio continuously and fans them out to (a) the
# STT queue (always) and (b) any per-PC subscriber queues (when browsers
# are connected). Without this, the wake word could only fire while a
# browser was connected — voice control was effectively gated on having
# /smart_laira_test open, which makes it useless as a standalone agent.
global_audio_player = None         # aiortc MediaPlayer wrapping the ALSA mic
global_stt_q = None               # asyncio.Queue feeding STT/wake pipeline
global_stt_task = None            # background STT worker task
global_audio_pump_task = None     # always-on consumer of MediaPlayer.audio
_pc_audio_subscribers = []        # list of asyncio.Queue, one per active PC

# --- global video singletons ---
# Same architecture as audio: cv2.VideoCapture is opened ONCE at startup
# and a reader task keeps laira_frames.cache fresh independent of any PC.
# PCs that connect get their own VideoStreamTrack that pulls frames out
# of the shared cache. This was the root cause of "AI commands stop
# working when nobody's on the website" — the camera capture was scoped
# to the PC, so without a browser the brain saw no frames and tier-2
# tools refused to run.
global_video_capture = None       # cv2.VideoCapture
global_camera_task = None         # background camera reader task

# Movement variables
max_z = 20
bounds = 30
base_stance_step = 1.0
# Gait smoothing: divide each per-tick step by this, AND divide the control
# loop tick interval by the same factor. Net: the overall velocity is
# unchanged but the motion is computed at N× finer granularity — each
# individual step is smaller and happens more often. This is pure anti-
# aliasing of the gait; does not change what "speed=5" means to callers.
# Was 1 (implicit) before this commit. Increase if you still see stepping
# in slow-tracking scenarios; decrease to cut Pi CPU load at the cost of
# visible aliasing.
SUBSAMPLE_FACTOR = 3
gait_speed = 20.0
bringback_speed = 1.0
rotation_mode = False
reset_state = False
idle_timer = 0
idle_threshold = 0.3
in_animation = False
in_angle_mode = False

# Gait-suppression flag for the paw twitch. control_loop normally sends
# all-motor commands every tick (20Hz) to maintain standing — that
# overwrites paw_twitch's hip moves within 50ms before the servos can
# physically respond. While this flag is True, control_loop skips its
# send_motor_commands call, letting the twitch's commands actually
# execute. Cleared by paw_twitch in finally so a crash mid-twitch
# doesn't leave the gait permanently frozen.
_suppress_gait_during_twitch = False

# ── Tick-rate speed control ───────────────────────────────────────────
# The main control_loop used to sleep a fixed 0.05s (20Hz). Now the sleep
# duration is driven by an optional 'speed' field on incoming keystate
# messages: speed=5 is the baseline (50ms), speed=10 doubles the tick rate
# (25ms), speed=1 is 5x slower (250ms). Linear in period.
#
# If no 'speed' is ever received, current_tick_interval stays at
# BASE_TICK_INTERVAL and the loop behaves exactly as before — keeping
# /user.html's AI Follow (which doesn't know about 'speed') working
# identically to today. Step-size-based slowdown via gait_speed (the old
# mechanism used by ai_mode) is untouched.
BASE_TICK_INTERVAL = 0.05 / SUBSAMPLE_FACTOR   # 16.67ms at factor=3 = 60Hz
current_tick_interval = BASE_TICK_INTERVAL
current_speed_value = 5.0          # 5 = baseline; clamped [1, 10] on update

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

# Per-coupled-pair sum correction. The original move formula constrains
# target_A + target_B to always equal exactly 1000 (perfect mirror around
# 500). On well-coupled pairs that's the right target. On pairs where the
# two motors are mounted with a slight mechanical asymmetry, the joint at
# rest naturally settles to a sum != 1000 — and any commanded pose that
# enforces sum=1000 makes the motors fight each other (each one pushed
# Δ/2 units away from where mechanical reality places it).
#
# Each entry is the offset to add to the pair's *target sum*, split
# evenly between motor A and motor B. Derived from rest-state encoder
# readings (motors off, joint settled at lying-down pose) — see
# read_motor_pair_diagnostic. Indices: [upper_pair, lower_pair].
#
# Procedure to retune: lie LaiRA down, send SIGUSR1, average two reads,
# set the correction = (measured_sum - 1000) per pair.
pair_sum_corrections = [
    [0,    -13],   # leg0: UPPER clean, LOWER avg Δ -13
    [-19,    0],   # leg1: UPPER avg Δ -19, LOWER unreliable (varies by pose) → 0
    [0,      0],   # leg2: both clean
    [+22,  -13],   # leg3: UPPER avg Δ +22, LOWER avg Δ -13
]
angle_buffer = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

# Serial control
ctrl = None
bus_ok = False
bus_lock = threading.Lock()

# WebSocket health monitoring
ws_last_message_time = time.time()
ws_connection = None
ws_healthy = True
# WS_TIMEOUT is the receive-side dead-connection backstop. Server now
# sends `{"type":"alive"}` every 15s on /laiRA (see server.js); Pi
# resets ws_last_message_time on any incoming frame INCLUDING `alive`
# (handled in the loop below). So normal idle = a heartbeat every 15s,
# real death = no traffic for WS_TIMEOUT seconds.
#
# History: this was 30s because the server sent NOTHING during idle
# periods, so the Pi's `ws.recv()` would time out every 30s on a
# perfectly healthy connection. Now that server-side heartbeats exist,
# 60s is plenty of headroom (4 missed heartbeats = real problem) and
# avoids needless reconnect churn. The websockets library's own
# ping_interval=10/ping_timeout=10 is a separate, lower-level keepalive
# that catches genuine TCP death within ~20s regardless.
WS_TIMEOUT = 60
WS_PING_INTERVAL = 10

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
    # Pair sum corrections: each pair's correction is split evenly between
    # the two motors (each gets c/2). Net effect: target_A + target_B =
    # 1000 + correction, matching the natural mechanical mirror where
    # motors don't fight. Integer halves (well, halves rounded toward
    # zero) so we don't drift fractional positions across calls; small
    # rounding error here is irrelevant vs encoder noise.
    upper_corr = pair_sum_corrections[legnum][0]
    lower_corr = pair_sum_corrections[legnum][1]
    upper_corr_half = upper_corr // 2
    lower_corr_half = lower_corr // 2
    if raw_3 == False or raw_2 == False or raw_1 == False:
        try:
            with bus_lock:
                ctrl.move(motor_nums[legnum][0], hip + motor_adjusts[legnum][0], motor_speed)
                ctrl.move(motor_nums[legnum][1], upper + motor_adjusts[legnum][1] + upper_corr_half, motor_speed)
                ctrl.move(motor_nums[legnum][2], 500 - (motor_adjusts[legnum][1] + upper - 500) + upper_corr_half, motor_speed)
                ctrl.move(motor_nums[legnum][3], lower + motor_adjusts[legnum][2] + lower_corr_half, motor_speed)
                ctrl.move(motor_nums[legnum][4], 500 - (motor_adjusts[legnum][2] + lower - 500) + lower_corr_half, motor_speed)
        except (serial.SerialException, OSError) as e:
            print(f"[motor-bus] lost connection → {e}")
            bus_ok = False
    else:
        try:
            with bus_lock:
                ctrl.move(motor_nums[legnum][0], raw_1, motor_speed)
                ctrl.move(motor_nums[legnum][1], raw_2 + upper_corr_half, motor_speed)
                ctrl.move(motor_nums[legnum][2], 500 - (raw_2) + upper_corr_half, motor_speed)
                ctrl.move(motor_nums[legnum][3], raw_3 + lower_corr_half, motor_speed)
                ctrl.move(motor_nums[legnum][4], 500 - (raw_3) + lower_corr_half, motor_speed)
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
# Step size per tick. Divided by SUBSAMPLE_FACTOR so that the control loop
# (now running at SUBSAMPLE_FACTOR× the old rate) ends up with the same
# effective velocity as before. Without this scaling, stepping faster would
# also mean moving faster — we want just-as-fast-but-smoother.
def get_normal_speed():
    global gait_speed
    return base_stance_step * gait_speed / SUBSAMPLE_FACTOR

def get_bringback_speed():
    global gait_speed
    return base_stance_step * bringback_speed * gait_speed / SUBSAMPLE_FACTOR

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
        # print(f"[key_handling] Active keys: {active}, Rotation mode: {rotation_mode}")
        pass
    
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
        
        # Per-tick motion direction prints silenced — were firing at the
        # tick rate (now 60Hz) and starving the asyncio event loop so brain
        # broadcasts never made it out. Mode-transition prints above stay
        # (they're one-shot). Re-enable with a guard flag if needed.
        if current_keystates.get('w', False) and not horizontal_pressed:
            for i in range(1, 5):
                leg_keys[i]['forward'] = True
        elif current_keystates.get('s', False) and not horizontal_pressed:
            for i in range(1, 5):
                leg_keys[i]['backward'] = True
        elif current_keystates.get('a', False) and not vertical_pressed:
            leg_keys[1]['left'] = True
            leg_keys[2]['right'] = True
            leg_keys[3]['left'] = True
            leg_keys[4]['right'] = True
        elif current_keystates.get('d', False) and not vertical_pressed:
            leg_keys[1]['right'] = True
            leg_keys[2]['left'] = True
            leg_keys[3]['right'] = True
            leg_keys[4]['left'] = True
    else:
        if current_keystates.get('q', False):
            leg_keys[1]['left'] = True
            leg_keys[2]['right'] = True
            leg_keys[3]['right'] = True
            leg_keys[4]['left'] = True
        elif current_keystates.get('e', False):
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
            
            # Debug: previously printed the per-leg, per-tick delta. At
            # 60Hz tick × 4 legs = ~240 sync prints/sec to journald, which
            # starves the asyncio event loop so badly the brain's ws
            # broadcasts (bbox, reasoning, tool_call) never make it out.
            # Silenced — re-enable only temporarily when debugging motion.
            # if old_target != leg_target[leg_num]:
            #     print(f"Leg {leg_num} moved: {old_target} -> {leg_target[leg_num]}")
            
            # Calculate inverse kinematics
            effective_target = leg_target[leg_num]
            ik = inverse_kinematics_priority(effective_target[0], effective_target[1], effective_target[2])
            
            if ik[0] is not None:
                if not in_angle_mode:
                    theta1, theta2, theta3 = ik
                else:
                    theta1, theta2, theta3 = angle_buffer[leg_num-1]
                
                if not in_angle_mode and not _suppress_gait_during_twitch:
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
        
        # Control loop rate. BASE_TICK_INTERVAL (0.05s = 20Hz) is the
        # baseline that matches the pre-speed-param behavior. The interval
        # shrinks/grows based on the most recent 'speed' field from a
        # keystate message (see handle_message). If no speed has ever been
        # received, current_tick_interval stays at BASE_TICK_INTERVAL and
        # this call is identical to the old `await asyncio.sleep(0.05)`.
        await asyncio.sleep(current_tick_interval)

# ============== VIDEO STREAMING ==============
def _open_camera(device=0, backend=cv2.CAP_V4L2, retries=3):
    """Open a cv2.VideoCapture handle for the USB camera. Tries a few
    times because the V4L2 device occasionally takes a beat to enumerate
    after boot.

    Sets CAP_PROP_READ_TIMEOUT_MSEC=1000 so a stuck cap.read() fails
    in 1s instead of OpenCV's V4L2-backend default of ~10s select()
    timeout. Without this, USB hiccups cause 30-50s of black frame
    before the consecutive-failure threshold trips a reopen."""
    for _ in range(retries):
        cap = cv2.VideoCapture(device, backend)
        if cap.isOpened():
            try:
                # Property is available in OpenCV ≥ 4.5; harmless no-op
                # on older builds (returns False but doesn't raise).
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)
            except Exception:
                pass
            return cap
        cap.release()
        time.sleep(0.5)
    return None


async def _camera_reader_loop():
    """Always-on camera reader. Owns the cv2.VideoCapture, reads frames
    in a tight loop, and pushes the latest into laira_frames.cache so
    the brain (and any connected PC's CachedVideoTrack) can consume.

    Runs until the service shuts down. Does NOT close the camera between
    PC reconnects — that was the original bug, where every PC churn
    destroyed the camera handle and the brain saw no frames in between.

    Auto-recovery: if cap.read() fails too many times consecutively
    (V4L2 select() timeout, USB hiccup, kernel driver wedged), release
    the handle and reopen. Without this, a single bad period would
    leave the camera permanently dead until manual service restart."""
    global global_video_capture
    if global_video_capture is None:
        global_video_capture = _open_camera()
    cap = global_video_capture
    if cap is None:
        print("⚠️ [camera] no USB camera; brain will run with stale/no frames")
        return
    print("[camera] reader loop online")
    loop = asyncio.get_event_loop()
    consecutive_failures = 0
    # Lowered from 5 to 2: with the new 1s read timeout (set in
    # _open_camera), each failure costs ~1s. Two failures = ~2s of
    # black frame before reopen, vs the old 50s worst case.
    FAILURE_THRESHOLD = 2
    while True:
        try:
            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= FAILURE_THRESHOLD:
                    print(f"[camera] {consecutive_failures} consecutive read failures — reopening V4L2 device")
                    try:
                        cap.release()
                    except Exception as e:
                        print(f"[camera] release errored: {e!r}")
                    await asyncio.sleep(1.0)  # let kernel fully release
                    new_cap = await loop.run_in_executor(None, _open_camera)
                    if new_cap is not None:
                        cap = new_cap
                        global_video_capture = new_cap
                        consecutive_failures = 0
                        print("[camera] V4L2 device reopened successfully")
                    else:
                        # Reopen failed; back off but don't crash. Reset
                        # the counter so we'll try reopen again after
                        # another N failures rather than burst-retrying.
                        consecutive_failures = 0
                        print("[camera] reopen failed — backing off 5s, will retry")
                        await asyncio.sleep(5.0)
                else:
                    # Sub-threshold failure — give the kernel a beat and
                    # try again rather than immediately tearing down.
                    await asyncio.sleep(0.05)
                continue
            consecutive_failures = 0
            laira_frames.cache.put(frame)
            # cv2 device on the Pi 5 caps around 30fps anyway, but yield
            # so other tasks (motor control, websocket, audio pump) get
            # the event loop. Without this the reader can monopolize when
            # frames are arriving back-to-back.
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[camera] read error: {e!r}")
            await asyncio.sleep(0.1)


async def init_global_camera_once():
    """Idempotent: spawns the camera reader on first call. Subsequent
    calls are no-ops. Called from main() before any PC is built."""
    global global_camera_task
    if global_camera_task and not global_camera_task.done():
        return
    global_camera_task = asyncio.create_task(_camera_reader_loop())


class CachedVideoTrack(VideoStreamTrack):
    """WebRTC video track that reads from laira_frames.cache instead of
    directly from cv2. Decouples the PC from camera ownership: any number
    of PCs can have one of these and they all see the same shared frame.

    If the cache is empty (camera not yet warmed up), recv() emits a
    black placeholder frame at the correct rate rather than failing —
    keeps the WebRTC handshake alive while the camera spins up."""

    _PLACEHOLDER_W = 640
    _PLACEHOLDER_H = 480

    async def recv(self):
        pts, tb = await self.next_timestamp()
        entry = laira_frames.cache.get_latest()
        if entry is None:
            # No frames yet. Emit a black placeholder so aiortc has
            # something to encode rather than raising MediaStreamError.
            frame = np.zeros((self._PLACEHOLDER_H, self._PLACEHOLDER_W, 3), np.uint8)
        else:
            frame, _wh, _ts = entry
        vf = VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts, vf.time_base = pts, tb
        return vf

class TestVideoTrack(VideoStreamTrack):
    def __init__(self, w=640, h=480):
        super().__init__()
        self.w, self.h = w, h
        self.c = 0

    async def recv(self):
        pts, tb = await self.next_timestamp()
        img = np.zeros((self.h, self.w, 3), np.uint8)
        img[:] = ((self.c*5)%256, (self.c*3)%256, (self.c*7)%256)
        # Expose the synthetic frame too so the brain works in sim mode.
        laira_frames.cache.put(img)
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
                    # The Logitech/eMeet C960 actually exposes a stereo
                    # capture stream (FL+FR mics ~5cm apart on the front
                    # of the camera). We were previously asking for mono
                    # and the kernel was downmixing for us — silently
                    # discarding the spatial information. Going stereo
                    # gives us direction-of-arrival data (which side the
                    # user spoke from) and optional beamforming for
                    # SNR improvement on quiet/distant speech.
                    "channels": "2",
                    "sample_rate": "48000",
                    "ac": "2",
                    "ar": "48000",
                    # Larger buffers absorb brief USB-bus contention from
                    # the motor controller (also USB) without producing
                    # userspace frame gaps. thread_queue_size: number of
                    # decoded frames libav can buffer; default ~8 was
                    # too small. audio_buffer_size: ALSA period buffer
                    # in microseconds — 200ms gives us breathing room
                    # for USB hiccups. Adds ~200ms latency to outbound
                    # browser audio, which is fine for a robot mic.
                    "thread_queue_size": "512",
                    "audio_buffer_size": "200000",
                }
            )
            if player and player.audio:
                print(f"[audio] using input: {src} (format={f})")
                return player
        except Exception as e:
            print(f"[audio] try {src} (format={f}) → {e}")

    print("[audio] no input found; starting without mic")
    return None



def _push_drop_oldest(q: asyncio.Queue, item):
    """Put `item` on `q`. If `q` is full, drop the OLDEST entry to make
    room. This is the right policy for a real-time audio pipeline:
    keeping the newest frame minimizes latency, whereas dropping the
    newest (the previous pattern) lets the queue stagnate at 400ms+ of
    stale audio whenever the consumer briefly falls behind. That's what
    'voice commands arrive later and later over time' was."""
    try:
        q.put_nowait(item)
        return
    except asyncio.QueueFull:
        pass
    try:
        q.get_nowait()
    except asyncio.QueueEmpty:
        pass
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        pass  # extreme contention; fine to drop


async def _audio_pump_loop():
    """Always-on consumer of MediaPlayer.audio.recv(). Fans every captured
    frame to the STT queue (always) and to each registered per-PC audio
    queue (when browsers are connected). Decouples STT from PC lifecycle
    so wake-word detection runs even with no browser open.

    Drop-oldest semantics on each downstream queue prevent the latency
    creep that the previous AudioForkTrack design was vulnerable to.

    Periodic diagnostic logging (every ~5s) shows per-second frame rate
    and stt_q depth — useful for confirming the pump is keeping up with
    real-time and the queue isn't backing up.

    On persistent recv failures (MediaPlayer's internal av container
    dies — happens occasionally on USB hiccup or after long uptime),
    we rebuild the MediaPlayer. Without this the pump spins forever on
    a dead source and STT goes deaf. Threshold is chosen so transient
    glitches don't cause unnecessary reopens (which themselves cost
    ~1-2s of audio)."""
    global global_audio_player
    src = global_audio_player.audio
    print("[audio] pump online")
    last_log_t = time.time()
    frames_since_log = 0
    last_frame_time = time.time()
    consecutive_recv_errors = 0
    REBUILD_AFTER_N_ERRORS = 20  # ~1s of failures before we rebuild
    GLITCH_THRESHOLD_S = 0.040  # 40ms = ~2 frames worth — actionable
    while True:
        try:
            frame = await src.recv()
            consecutive_recv_errors = 0
        except asyncio.CancelledError:
            raise
        except Exception as e:
            consecutive_recv_errors += 1
            # Only log first error in a streak + every 50th to avoid
            # spamming the journal with the same MediaStreamError.
            if consecutive_recv_errors == 1 or consecutive_recv_errors % 50 == 0:
                print(f"[audio] pump recv error #{consecutive_recv_errors}: {e!r}")
            if consecutive_recv_errors >= REBUILD_AFTER_N_ERRORS:
                print(f"[audio] pump dead after {consecutive_recv_errors} recv errors — rebuilding MediaPlayer")
                try:
                    if global_audio_player is not None:
                        try:
                            if hasattr(global_audio_player, "_stop"):
                                global_audio_player._stop(None)
                        except Exception as _e:
                            print(f"[audio] old player stop failed (ignoring): {_e!r}")
                    new_player = _make_audio_player()
                    if new_player is None or new_player.audio is None:
                        print("[audio] rebuild failed — MediaPlayer factory returned no audio. Will retry in 2s.")
                        await asyncio.sleep(2.0)
                        consecutive_recv_errors = 0
                        continue
                    global_audio_player = new_player
                    src = global_audio_player.audio
                    consecutive_recv_errors = 0
                    last_frame_time = time.time()
                    print("[audio] MediaPlayer rebuilt, pump resuming")
                    continue
                except Exception as rebuild_e:
                    print(f"[audio] rebuild raised: {rebuild_e!r} — sleeping 2s and retrying loop")
                    await asyncio.sleep(2.0)
                    consecutive_recv_errors = 0
                    continue
            await asyncio.sleep(0.05)
            continue
        now_t = time.time()
        gap = now_t - last_frame_time
        if gap > GLITCH_THRESHOLD_S:
            print(f"[audio:glitch] frame gap {gap*1000:.0f}ms (target 20ms)")
        last_frame_time = now_t
        if global_stt_q is not None:
            _push_drop_oldest(global_stt_q, frame)
        for q in list(_pc_audio_subscribers):
            _push_drop_oldest(q, frame)
        # Diagnostic: rate + queue depth. If FPS is consistently > ~52
        # the pump is rapid-draining buffered audio (catchup phase). If
        # stt_q.qsize() == _STT_QUEUE_MAX consistently the worker can't
        # keep up. Steady state should be ~50 fps and qsize ≤ 1.
        frames_since_log += 1
        now = time.time()
        if now - last_log_t >= 5.0:
            elapsed = now - last_log_t
            fps = frames_since_log / elapsed if elapsed > 0 else 0
            stt_depth = global_stt_q.qsize() if global_stt_q is not None else -1
            sub_depths = [q.qsize() for q in list(_pc_audio_subscribers)]
            print(
                f"[audio:diag] {fps:.1f}fps stt_q={stt_depth}/{_STT_QUEUE_MAX} "
                f"pc_subs={sub_depths}"
            )
            last_log_t = now
            frames_since_log = 0


async def init_global_audio_once():
    """Open ALSA mic once, start the always-on audio pump, launch the
    persistent STT worker. Idempotent."""
    global global_audio_player, global_stt_q, global_stt_task, global_audio_pump_task

    if global_audio_player and global_audio_pump_task and global_stt_task:
        print("[audio] global audio already initialized")
        return

    # open mic once — survives PC reconnects, WebRTC churn, etc.
    global_audio_player = _make_audio_player()
    if not (global_audio_player and global_audio_player.audio):
        print("[audio] no input found; STT/WebRTC will have no mic")
        return

    global_stt_q = asyncio.Queue(maxsize=_STT_QUEUE_MAX)
    print("[audio] starting STT worker + audio pump (PC-independent)")
    global_stt_task = asyncio.create_task(stt_worker(global_stt_q, on_text=stt_action_router))
    global_audio_pump_task = asyncio.create_task(_audio_pump_loop())




# ============== WEBSOCKET MESSAGE HANDLING ==============
async def brain_motors_on():
    """Ensure motors are powered so pose/drive commands actually produce
    movement. Idempotent — no-op if motors are already on. Mirrors the
    motor_on path in handle_message's keystates branch, which the brain
    previously bypassed (writes to `keystates` don't go through
    handle_message, so brain-initiated motion on top of a resting/lying
    LaiRA did nothing)."""
    global motors_resting, off_due_to_disconnect
    if motors_resting:
        if ctrl is not None:
            try:
                ctrl.motor_on(SERVO_ID_ALL)
            except Exception as e:
                print(f"[brain_motors_on] motor_on failed: {e}")
        motors_resting = False
        off_due_to_disconnect = False


async def brain_stand_up():
    """Full transition to a standing stance: motors on, exit angle-mode,
    reset leg targets to neutral, wait for servo interpolation to settle.
    Called by the brain before any motion tool (drive / go_to / follow)
    or when the stand pose itself fires. Idempotent — if already standing
    this is basically free.

    This fixes two related bugs:
      1. drive() / tracking keystates from a lying state used to be a
         no-op because motors were off and control_loop was still in
         angle_mode reading stale angle_buffer.
      2. The brain's "stand" pose was sending a zero-keystate, which does
         NOT wake her up (zero-keystates = "stop moving", not "transition
         out of lying down"). Now stand goes through here and actually
         stands her up.
    """
    global in_animation, in_angle_mode
    was_not_standing = in_animation or in_angle_mode or motors_resting
    await brain_motors_on()
    if in_animation or in_angle_mode:
        await reset_function(
            reset_to_wasd=True, reset_to_rotation=False, reset_positions_only=False
        )
        in_animation = False
        in_angle_mode = False
    if was_not_standing:
        # Servos are interpolating to neutral; give them a beat before a
        # subsequent drive keystate tries to walk, or she'll try to walk
        # mid-stand-up and look drunk.
        await asyncio.sleep(0.8)


async def read_motor_pair_diagnostic(label="manual"):
    """Read positions on every coupled motor pair (upper + lower joint
    on each leg, 4 legs × 2 pairs = 8 motors total) and log the data.

    Each leg uses two motors per joint that are mechanically coupled to
    drive the same axis. The control code sends mirrored targets:
    target_A = u + offset, target_B = 1000 - u - offset. So target_A +
    target_B is always exactly 1000 by design. If the motors are well-
    calibrated and not fighting, their *measured* positions should also
    sum to ~1000 (any deviation is the amount each motor is fighting
    the other).

    Use cases:
      - At rest (motors_resting=True, no torque applied): the joint
        settles to whatever the mechanical zero is; readings show how
        the motors actually sit relative to each other when free.
      - Standing: shows how much fighting (if any) is happening when
        the motors are actively driving the pose.

    Triggered via SIGUSR1 (kill -USR1 <pid>) — keeps the diagnostic
    out of the regular control flow but easy to fire on demand."""
    if not bus_ok or ctrl is None:
        print(f"[motor_diag:{label}] bus not ready")
        return
    print(f"[motor_diag:{label}] === motor pair positions ===")
    for legnum in range(4):
        ids = motor_nums[legnum]
        upper_a, upper_b = ids[1], ids[2]
        lower_a, lower_b = ids[3], ids[4]
        try:
            with bus_lock:
                ua = ctrl.get_position(upper_a, timeout=0.2)
                ub = ctrl.get_position(upper_b, timeout=0.2)
                la = ctrl.get_position(lower_a, timeout=0.2)
                lb = ctrl.get_position(lower_b, timeout=0.2)
        except Exception as e:
            print(f"[motor_diag:{label}] leg{legnum}: read fail {e!r}")
            continue
        upper_sum = ua + ub
        lower_sum = la + lb
        upper_dev = upper_sum - 1000
        lower_dev = lower_sum - 1000
        print(
            f"[motor_diag:{label}] leg{legnum} "
            f"UPPER id{upper_a}={ua} id{upper_b}={ub} sum={upper_sum} (Δ={upper_dev:+d}) | "
            f"LOWER id{lower_a}={la} id{lower_b}={lb} sum={lower_sum} (Δ={lower_dev:+d})"
        )
    print(f"[motor_diag:{label}] === end ===")


async def brain_paw_twitch():
    """Twitch all four hip motors back and forth for ~150ms total.
    Produces an audible servo whine + visible jiggle on every leg,
    giving the user immediate confirmation that the wake word fired
    even before STT/transcription has a chance to complete or fail.

    Called from laira_brain.handle_wake_detected (which the STT pipeline
    invokes the instant the wake model triggers).

    Was previously a single-leg ±25 twitch — too subtle to notice
    across a room. Bumped to all-4-legs ±50 (~12° each way) per user
    request. Costs more USB bus time (~80ms of serial I/O burst vs
    ~25ms before), but the brief audio glitch on the WebRTC outbound
    is acceptable for the much clearer feedback.

    Runs all serial commands in a worker thread so they don't stall
    the asyncio loop. No-op when the motor bus isn't initialized."""
    if not bus_ok or ctrl is None:
        return
    global _suppress_gait_during_twitch
    hip_servos = [motor_nums[i][0] for i in range(4)]

    def _mark_bus_dead(why):
        """When a serial read/write fails with EIO/SerialException, the
        underlying USB device very likely re-enumerated (Genesys hub
        instability — see C960 USB hiccup pattern). The serial fd we
        hold is now stale; subsequent operations will keep failing
        until we close + reopen via the by-id symlink. Setting
        bus_ok=False causes the next ensure_bus() call to spawn an
        _auto_reconnector thread that does exactly that."""
        global bus_ok
        if bus_ok:
            print(f"[paw_twitch] {why} — marking bus dead, reconnect will trigger on next motor op")
            bus_ok = False
            # Kick reconnect immediately rather than waiting for the
            # next caller to notice. ensure_bus() spawns the thread
            # safely (idempotent via the bus_ok='pending' guard).
            ensure_bus()

    def _read_all():
        positions = {}
        for sid in hip_servos:
            try:
                with bus_lock:
                    positions[sid] = ctrl.get_position(sid, timeout=0.15)
            except (serial.SerialException, OSError) as e:
                print(f"[paw_twitch] read failed (servo={sid}): {e!r}")
                positions[sid] = None
                _mark_bus_dead(f"read failed on servo {sid}")
                # No point trying remaining servos — fd is stale, they'll all EIO.
                return positions
            except Exception as e:
                print(f"[paw_twitch] read failed (servo={sid}): {e!r}")
                positions[sid] = None
        return positions

    def _move_all(targets):
        for sid, pos in targets.items():
            if pos is None:
                continue
            try:
                with bus_lock:
                    ctrl.move(sid, pos, 60)
            except (serial.SerialException, OSError) as e:
                print(f"[paw_twitch] move failed (servo={sid}): {e!r}")
                _mark_bus_dead(f"move failed on servo {sid}")
                return
            except Exception as e:
                print(f"[paw_twitch] move failed (servo={sid}): {e!r}")

    loop = asyncio.get_event_loop()
    # Suppress the gait loop's per-tick motor sends for the twitch
    # window. Without this, the gait re-issues the hip's standing-pose
    # target every 50ms, so our move command gets overwritten before
    # the servo physically completes its interpolation. Set BEFORE the
    # first read+move so even position-read-time gait ticks are gated.
    _suppress_gait_during_twitch = True
    try:
        current = await loop.run_in_executor(None, _read_all)
        if all(v is None for v in current.values()):
            return
        # ±50 = ~12° each way. Highly visible across a room, clearly
        # audible servo movement on all four legs at once.
        raised = {
            sid: max(0, min(1000, pos + 50))
            for sid, pos in current.items() if pos is not None
        }
        await loop.run_in_executor(None, _move_all, raised)
        await asyncio.sleep(0.12)
        await loop.run_in_executor(None, _move_all, current)
        # Brief settle window: let the "move back" complete its
        # interpolation before gait resumes its tick. Without this,
        # the next gait tick (within 50ms) might overlap with our
        # final move's interpolation and produce a small jerk.
        await asyncio.sleep(0.08)
    finally:
        _suppress_gait_during_twitch = False


async def brain_set_keystates(ks, speed=None):
    """Brain-driven keystate write. Goes through the SAME locked path as
    browser-driven keystates, so the gait loop's state machine sees a
    single coherent stream. Partial ks (e.g. {'w': True}) fills missing
    keys with False, matching executeKeystate() semantics on the client
    side. Optional speed is clamped and applied to the tick interval.
    """
    global current_tick_interval, current_speed_value
    if speed is not None:
        try:
            s = max(1.0, min(10.0, float(speed)))
            if s != current_speed_value:
                current_speed_value = s
                current_tick_interval = BASE_TICK_INTERVAL * (5.0 / s)
        except (TypeError, ValueError):
            pass
    async with keystates_lock:
        for k in ('w', 'a', 's', 'd', 'q', 'e'):
            keystates[k] = bool(ks.get(k, False))


async def handle_message(msg):
    """Process incoming WebSocket messages for robot control"""
    global keystates, in_animation, in_angle_mode, motors_resting, off_due_to_disconnect, ai_mode, gait_speed
    global ws_last_message_time
    global current_tick_interval, current_speed_value
    
    # Update last message time
    ws_last_message_time = time.time()
    
    try:
        if 'message' not in msg:
            print(f"msg recieved not with msg, prolly health packet")
            return
            
        message = msg['message']
        # Temporary diagnostic: log the FIRST 120 chars of every chat-
        # message we receive. Proves (or disproves) that the Pi is seeing
        # user_text / brain_params / hello_request / etc. coming in.
        # Remove once we've verified browser->Pi flow.
        try:
            _preview = message if isinstance(message, str) else repr(message)
            print(f"[recv] {_preview[:120]}")
        except Exception:
            pass

        # ── Brain-targeted messages from /smart_laira_test ───────────────
        # Wrapped in the same chat-message envelope but use new top-level
        # keys in the inner JSON. Substring check matches the style of the
        # surrounding branches; real parse happens immediately below so we
        # don't mis-route on false positives.
        if ('user_text' in message or 'bbox_stream' in message
                or 'brain_params' in message or '"cancel"' in message
                or 'hello_request' in message or 'stt_to_brain' in message):
            try:
                inner = json.loads(message)
            except Exception:
                inner = None
            if inner is not None:
                if inner.get('hello_request') is True:
                    # Browser just connected (or reconnected) and wants the
                    # current brain + modal state. Hello messages emitted at
                    # Pi-ws-open are lost if the browser isn't up yet — this
                    # request path fixes that startup race.
                    await laira_brain.on_hello_request()
                    return
                if 'user_text' in inner and inner['user_text']:
                    # Text command — kick off brain routing.
                    await laira_brain.submit_text(inner['user_text'])
                    return
                if 'bbox_stream' in inner:
                    laira_brain.set_bbox_stream(inner['bbox_stream'])
                    return
                if 'stt_to_brain' in inner:
                    # Browser toggle: if True, wake-word STT phrases get
                    # forwarded into the T1 pipeline in addition to the
                    # console broadcast. Default False.
                    laira_brain.set_stt_to_brain(inner['stt_to_brain'])
                    return
                if 'brain_params' in inner and isinstance(inner['brain_params'], dict):
                    bp = inner['brain_params']
                    if 'speed' in bp:
                        laira_brain.set_ui_speed(bp['speed'])
                    if 'closeness' in bp:
                        laira_brain.set_ui_closeness(bp['closeness'])
                    return
                if inner.get('cancel') is True:
                    # Explicit brain-plan cancel from a manual action.
                    await laira_brain.cancel_active()
                    return

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
                # Optional 'speed' field rides alongside keystates. Absent =
                # keep current tick interval (which is BASE_TICK_INTERVAL on
                # first boot or last-set value otherwise). Present = clamp
                # to [1, 10] and recompute tick interval. Malformed = ignore
                # silently (don't crash the motor loop on a bad client).
                if 'speed' in inner:
                    try:
                        s = float(inner['speed'])
                        s = max(1.0, min(10.0, s))
                        if s != current_speed_value:
                            current_speed_value = s
                            current_tick_interval = BASE_TICK_INTERVAL * (5.0 / s)
                            print(f"[speed] -> {s} (tick={current_tick_interval*1000:.0f}ms)")
                    except (TypeError, ValueError):
                        pass

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

def _ws_is_open(ws):
    try:
        # Old versions expose .closed (bool)
        if hasattr(ws, "closed"):
            return not ws.closed
        # Newer versions expose .state (enum with .name like "OPEN")
        state = getattr(ws, "state", None)
        return getattr(state, "name", "").upper() in ("OPEN", "CONNECTING")
    except Exception:
        return False


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
                    if _ws_is_open(ws_connection):
                        print("🔄 Forcing WebSocket reconnection...")
                        await ws_connection.close()
            else:
                if not ws_healthy:
                    print("✅ WebSocket health restored")
                    ws_healthy = True
            
            
            await asyncio.sleep(WS_PING_INTERVAL)
            
        except Exception as e:
            print(f"Error in health monitor: {e}")
            await asyncio.sleep(5)


# Shallow queues throughout — drop-oldest policy bounds latency to a few
# frames (~80ms) rather than the previous 400ms (20 × 20ms). The wake
# model only needs the most recent audio; old buffered frames are dead
# weight that just delays detection.
_STT_QUEUE_MAX = 4
_PC_AUDIO_QUEUE_MAX = 4


class SubscribedAudioTrack(AudioStreamTrack):
    """WebRTC outbound audio track. Owns a per-PC subscriber queue that
    the global audio pump pushes frames into. recv() is called by the
    PC's RTP sender; we just pop the next queued frame.

    Replaces the old AudioForkTrack pattern, which was driven BY the PC's
    recv() — meaning audio capture (and STT) only ran while a browser
    was connected. Now capture is always-on and PCs subscribe."""
    kind = "audio"

    def __init__(self):
        super().__init__()
        self._q = asyncio.Queue(maxsize=_PC_AUDIO_QUEUE_MAX)
        _pc_audio_subscribers.append(self._q)

    async def recv(self):
        return await self._q.get()

    def stop(self):
        try:
            _pc_audio_subscribers.remove(self._q)
        except ValueError:
            pass
        return super().stop()

async def stt_worker(stt_q: asyncio.Queue, on_text=None):
    """Drain aiortc audio frames from the shared queue, resample to
    16kHz mono int16, and feed into the laira_stt wake+record+transcribe
    pipeline. The pipeline calls `on_command` when it has a fully
    transcribed command to deliver.

    New two-stage design (replaces the old always-on-Vosk approach):
      - openWakeWord scores each 80ms chunk for the "hey_jarvis" (or
        custom-trained) wake word. Cheap (<few% CPU on Pi 5).
      - On wake detection, pipeline switches to RECORDING and buffers
        audio until silero VAD indicates N ms of silence (or max 12s).
      - Buffered audio runs through faster-whisper tiny.en — far better
        accuracy than Vosk for short commands. Runs in a thread so the
        motor event loop stays responsive during the ~0.5-1s transcribe.
      - Transcribed text goes to brain.handle_stt_phrase.
    """
    # Stereo resampler — captures the FL/FR mics from the C960 separately
    # so the SttPipeline can derive direction-of-arrival (which side the
    # user spoke from) and optionally beamform for SNR improvement. The
    # pipeline downmixes to mono internally for wake/VAD/STT.
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="stereo", rate=16000)
    # on_text is the callback the caller specifies (stt_action_router).
    # Now also receives the direction-of-arrival ("left"/"right"/"center")
    # so the brain can include it in the user_text for tier-2 reasoning.
    async def on_command(text, doa=None):
        if on_text:
            await on_text(text, doa)
    async def on_wake():
        # Routed straight into the brain — interrupts any active plan
        # immediately and twitches the front-leg hip for auditory feedback
        # (the user hears the servo whine and knows she registered the
        # wake before STT/transcription has a chance to fail).
        try:
            await laira_brain.handle_wake_detected()
        except Exception as e:
            print(f"[stt] handle_wake_detected failed: {e!r}")
    async def on_drop(reason):
        # Surface no-speech / hallucination drops to the browser so the
        # user knows their wake word fired but the command didn't make
        # it through — otherwise they see a paw twitch and then silence,
        # which looks like a hang.
        try:
            await laira_brain.handle_stt_dropped(reason)
        except Exception as e:
            print(f"[stt] handle_stt_dropped failed: {e!r}")
    pipeline = laira_stt.SttPipeline(on_command=on_command, on_wake=on_wake, on_drop=on_drop)
    # AGC removed: the per-chunk peak normalization was creating amplitude
    # jumps between consecutive 80ms chunks that confused Silero VAD,
    # which evaluates speech probability independently per-chunk. Symptom:
    # Whisper would transcribe the user's command perfectly ("Lyra, Lyra,
    # stand up.") but VAD never fired any single chunk above its 0.4
    # threshold during the entire recording, so the pipeline thought no
    # speech was detected and dropped. A fixed gain or smoother RMS-based
    # AGC would be safer if mic boost is needed again later, but for now
    # the natural mic level seems to be enough for VAD when paired with
    # the 0.4 threshold.

    try:
        while True:
            frame = await stt_q.get()  # av.AudioFrame (now stereo)
            out = resampler.resample(frame)
            if out is None:
                continue
            frames = out if isinstance(out, list) else [out]
            for f in frames:
                pcm = f.to_ndarray()  # interleaved s16 stereo
                # av's to_ndarray() for s16 stereo returns shape (1, 2*N)
                # — interleaved L0,R0,L1,R1,... Flatten + split.
                if pcm.ndim == 2 and pcm.shape[0] == 1:
                    pcm = pcm[0]
                # If somehow we got mono (shouldn't happen post-resampler),
                # duplicate to stereo so downstream code is uniform.
                if pcm.size % 2 != 0:
                    # odd sample count — drop the trailing sample
                    pcm = pcm[:-1]
                pcm_L = pcm[0::2]
                pcm_R = pcm[1::2]
                # Pass both L and R channels to the pipeline. The pipeline
                # itself decides whether to feed mono-downmix or beamformed
                # mono to wake/VAD/STT, and uses the stereo data at
                # finish-recording to compute DOA.
                await pipeline.feed_stereo(pcm_L, pcm_R)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[stt_worker] error: {e}")


async def stt_action_router(text: str, doa=None):
    """Dispatch a fully-captured voice command to the brain.

    Upstream (laira_stt) has already done wake-word detection + command
    recording + transcription — by the time we're here, `text` is the
    command phrase exactly as the user spoke it AFTER the wake word.
    We just hand it to the brain for broadcast + optional T1 forwarding."""
    try:
        await laira_brain.handle_stt_phrase(text, doa=doa)
    except Exception as e:
        print(f"[stt_action_router] error: {e}")



# ============== MAIN WEBRTC/WEBSOCKET HANDLER ==============
async def make_pc(ice_servers):
    config = RTCConfiguration([RTCIceServer(**srv) for srv in ice_servers])
    pc = RTCPeerConnection(config)

    # Video track: read from the always-on camera reader's frame cache.
    # If the camera failed to open at startup, fall back to TestVideoTrack
    # so the WebRTC handshake still completes (just with synthetic frames).
    if global_video_capture is not None:
        track = CachedVideoTrack()
    else:
        track = TestVideoTrack()
    pc.addTrack(track)
    pc._video_track = track

    # Audio track: subscribe a per-PC queue to the always-on audio pump.
    # Each new PC gets a fresh subscriber; SubscribedAudioTrack.stop()
    # auto-deregisters on close.
    if global_audio_player is not None and global_audio_player.audio is not None:
        audio_track = SubscribedAudioTrack()
        pc.addTrack(audio_track)
        pc._audio_track = audio_track
        print("[audio] added WebRTC audio track (subscribed to pump)")
    else:
        print("[audio] no global audio player; WebRTC will have no audio")


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
        state = pc.connectionState
        print("🔗 PC state:", state)
        # Terminal PC states need to force a WS reconnect so the outer
        # loop builds a fresh PeerConnection. Without this, the WS stays
        # alive (server heartbeats are arriving every 15s, see server.js)
        # but the WebRTC pipe is dead — video freezes and audio stops
        # flowing into STT, with no recovery. Pre-heartbeat-fix this used
        # to recover via WS_TIMEOUT, but now nothing reacts.
        if state in ("closed", "failed"):
            ws = ws_connection
            if ws is not None and _ws_is_open(ws):
                print(f"🔗 PC {state} → forcing WS close to trigger reconnect")
                try:
                    asyncio.get_event_loop().create_task(ws.close())
                except Exception as e:
                    print(f"   close failed: {e}")

    return pc

async def websocket_handler():
    """Main WebSocket connection handler with improved error handling"""
    global ws_connection, ws_last_message_time, ws_healthy
    
    reconnect_delay = 1
    max_reconnect_delay = 30
    
    # metered.live's TURN credentials endpoint is sometimes slow (10-20s).
    # Old 10s read timeout was failing on their slow responses, and those
    # failures bubble up as "Failed to get TURN credentials: timed out
    # during opening handshake" → full PC teardown → reconnect → repeat.
    # 45s total / 30s read gives the service room to breathe.
    TURN_TIMEOUT = aiohttp.ClientTimeout(total=45, connect=10, sock_connect=10, sock_read=30)

    while True:
        ws_connection = None
        pc = None
        
        try:
            # Get TURN credentials
            async with aiohttp.ClientSession(timeout=TURN_TIMEOUT) as sess:

                async with sess.get(TURN_URL) as resp:
                    resp.raise_for_status()
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
                max_queue=32,
                compression=None
            ) as ws:
                import socket
                sock = None
                try:
                    # Some websockets versions expose ws.transport; guard just in case.
                    transport = getattr(ws, "transport", None)
                    if transport is not None:
                        sock = transport.get_extra_info("socket")
                    if isinstance(sock, socket.socket):
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30)   # seconds
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
                except Exception as e:
                    print(f"[keepalive] could not set TCP keepalive: {e}")

                ws_connection = ws
                ws_last_message_time = time.time()
                ws_healthy = True
                reconnect_delay = 1  # Reset delay on successful connection

                print("✅ WebSocket connected")
                await ws.send(json.dumps({"type": "laiRA-connected"}))
                # Let the brain know it can now broadcast state/logs to any
                # connected browser. Also re-emit hello + current modal
                # status so a late-joining /smart_laira_test picks them up.
                laira_brain.set_outbound_ws(ws)
                await laira_brain.on_ws_opened()

                # Offer-wait safety net. The server broadcasts
                # `laiRA-connected` to all browsers when we attach, which
                # should cause them to send an offer within a couple
                # seconds. But sometimes — e.g., browser tab in some
                # half-dead WS state, or laiRA-disconnected broadcast
                # raced with the browser's own WS reconnect — that
                # broadcast doesn't trigger an offer, and the Pi sits
                # forever with no peer. Force a fresh reconnect cycle
                # if we don't see an offer in OFFER_WAIT_S so the server
                # rebroadcasts laiRA-disconnected then laiRA-connected,
                # giving the browser a clean retry trigger.
                OFFER_WAIT_S = 20
                ws_connect_time = time.time()
                async def _offer_watchdog():
                    await asyncio.sleep(OFFER_WAIT_S)
                    if not got_offer and _ws_is_open(ws):
                        print(f"⚠️ No OFFER within {OFFER_WAIT_S}s of WS connect → forcing reconnect")
                        try:
                            await ws.close()
                        except Exception:
                            pass
                offer_watchdog_task = asyncio.create_task(_offer_watchdog())

                @pc.on("icecandidate")
                async def send_ice(cand):
                    if cand and _ws_is_open(ws_connection):
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

                        if msg_type == "alive":
                            ws_last_message_time = time.time()
                            continue
                        
                        elif msg_type == "offer":

                            if got_offer:
                                print("⚠️ Duplicate OFFER → reconnecting")
                                break
                            print("📬 Received OFFER")
                            await pc.setRemoteDescription(RTCSessionDescription(sdp=msg["sdp"], type="offer"))
                            got_offer = True
                            # Watchdog satisfied — cancel it so it doesn't
                            # fire later and tear down a healthy session.
                            if offer_watchdog_task and not offer_watchdog_task.done():
                                offer_watchdog_task.cancel()

                            # flush queued ICE
                            for c in queue:
                                ice = candidate_from_sdp(c["candidate"])
                                ice.sdpMid, ice.sdpMLineIndex = c.get("sdpMid"), c.get("sdpMLineIndex")
                                await pc.addIceCandidate(ice)
                            queue.clear()
                            ans = await pc.createAnswer()
                            await pc.setLocalDescription(ans)
                            await ws.send(json.dumps({
                                "type": pc.localDescription.type,
                                "sdp": pc.localDescription.sdp
                            }))

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

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"⚠️ Failed to get TURN credentials: {e}")

        except Exception as e:
            print(f"⚠️ WebSocket error: {e}")

        finally:
            ws_connection = None
            ws_healthy = False
            # Tell the brain its broadcast channel is gone so its fire-and-
            # forget sends drop silently instead of raising on a closed ws.
            laira_brain.set_outbound_ws(None)

            # Cancel the offer watchdog if still pending — we're already
            # tearing down so its scheduled close would just be redundant.
            try:
                if offer_watchdog_task and not offer_watchdog_task.done():
                    offer_watchdog_task.cancel()
            except (NameError, Exception):
                pass

            # Clean up peer connection
            if pc:
                try:
                    if hasattr(pc, "_video_track"):
                        pc._video_track.stop()
                    # SubscribedAudioTrack.stop() deregisters its queue
                    # from _pc_audio_subscribers so the audio pump stops
                    # pushing to a dead PC. Without this, every reconnect
                    # would leak a subscriber queue.
                    if hasattr(pc, "_audio_track"):
                        try:
                            pc._audio_track.stop()
                        except Exception as e:
                            print(f"[audio_track.stop] {e}")
                    await pc.close()
                except Exception as e:
                    print(f"[pc.close] {e}")
            
            # Exponential backoff for reconnection
            print(f"⏳ Reconnecting in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

# ============== MAIN ENTRY POINT ==============
async def main():
    """Main async function that runs control loop, WebSocket handler, and health monitor"""
    print("🤖 Starting unified robot control system...")
    
    await init_global_audio_once()
    await init_global_camera_once()

    # SIGUSR1 → run the motor-pair diagnostic. Triggered via:
    #   ssh laira "kill -USR1 \$(pgrep -f laira_go_wsprotected)"
    # Output goes to the journal under [motor_diag:sigusr1]. Used for
    # one-off coupled-motor alignment checks without baking diagnostic
    # behavior into the regular control flow.
    import signal as _signal
    loop_ref = asyncio.get_event_loop()
    def _on_sigusr1(*_):
        loop_ref.call_soon_threadsafe(
            asyncio.create_task,
            read_motor_pair_diagnostic("sigusr1")
        )
    try:
        _signal.signal(_signal.SIGUSR1, _on_sigusr1)
        print("[motor_diag] SIGUSR1 handler armed")
    except Exception as e:
        print(f"[motor_diag] SIGUSR1 install failed: {e}")

    control_task = asyncio.create_task(control_loop())
    websocket_task = asyncio.create_task(websocket_handler())
    health_task = asyncio.create_task(websocket_health_monitor())
    # Brain runs in the same loop; safe to spawn as a peer task. Any
    # uncaught exception inside start() stays contained to that task —
    # the motor/gait path keeps running regardless.
    brain_task = asyncio.create_task(laira_brain.start())
    
    await asyncio.gather(control_task, websocket_task, health_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
