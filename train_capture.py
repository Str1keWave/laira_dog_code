#!/usr/bin/env python3
"""train_capture.py — interactive curriculum for collecting wake-word training data.

Run on the Pi (after stopping laira.service):
    sudo systemctl stop laira.service
    /home/emre/laira_code/venv/bin/python3 /home/emre/laira_code/train_capture.py

Output: ~/wake_training_data/{positive|negative|ambient}/{step_id}_{label}.wav
All files are 16kHz mono int16 WAV — directly consumable by openwakeword's
training pipeline. Each step records for a fixed duration; press Enter when
you're in position and ready.
"""

import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

CAPTURE_DEVICE = "plughw:2,0"
CAPTURE_RATE = 48000  # native rate that's reliable on this hw
TARGET_RATE = 16000
TMP_PATH = "/tmp/_capture.wav"
OUT_BASE = Path.home() / "wake_training_data"

# (step_id, category, label, duration_s, prompt)
CURRICULUM = [
    # Section A: Positive — vocal styles
    ("A1", "positive", "isolated_normal", 12,
     'Stand ~1m from LaiRA. Say "Lyra." FIVE TIMES in your normal voice, with ~1s pauses between each.'),
    ("A2", "positive", "midsentence_commands", 20,
     'Same position. Say each of these THREE TIMES (so 9 utterances total): "Hey Lyra come here" / "Lyra stop" / "Lyra stand up". Natural pace.'),
    ("A3", "positive", "excited", 10,
     'Say "Hey, Lyra!" THREE TIMES with energy — like calling a dog.'),
    ("A4", "positive", "drawn_out", 10,
     'Say "Lyyyra..." slowly THREE TIMES, drawing out the vowel.'),
    ("A5", "positive", "questioning", 8,
     'Say "Lyra?" with rising intonation THREE TIMES, like you\'re asking.'),
    ("A6", "positive", "whispered", 10,
     'Whisper "Lyra come here" THREE TIMES, like she\'s sleeping nearby.'),
    ("A7", "positive", "shouted", 10,
     'Shout "LYRA!" THREE TIMES — across-the-room volume.'),
    ("A8", "positive", "trailing_into_cmd", 12,
     'Say "Okay, Lyra, lie down" THREE TIMES naturally.'),
    ("A9", "positive", "question_cmd", 12,
     'Say "Lyra, what do you see?" THREE TIMES.'),
    ("A10", "positive", "rapid_repeats", 10,
     'Say "Lyra Lyra Lyra" rapidly THREE TIMES, like you\'re impatient.'),

    # Section B: Positive — distance/position
    ("B1", "positive", "close_30cm", 10,
     'Move to ~30cm in front of LaiRA. Say "Lyra come here" THREE TIMES.'),
    ("B2", "positive", "far_3m", 14,
     'Move ~3m away. Say "Lyra come here" THREE TIMES, normal voice.'),
    ("B3", "positive", "behind", 10,
     'Stand directly behind her. Say "Lyra" THREE TIMES.'),
    ("B4", "positive", "side_90deg", 12,
     'Stand 90° to her side (either side). Say "Lyra hello" THREE TIMES.'),
    ("B5", "positive", "other_room", 12,
     'Move to a different room or down a hallway. Say "Lyra come here" THREE TIMES.'),

    # Section C: Positive — environmental noise
    ("C1", "positive", "with_tv_music", 14,
     'Turn on TV or music at typical background volume. Stand ~1m from LaiRA. Say "Lyra come here" THREE TIMES.'),

    # Section D: Negative — phonetic neighbors
    ("D1", "negative", "phonetic_clean", 18,
     'Say each ONCE with ~1s pauses, normal voice: "Laura. Lara. Mira. Layer. Lira. Lyric. Liar. Library. Lawyer. Larry."'),
    ("D2", "negative", "phonetic_in_speech", 22,
     'Say these naturally as if in conversation: "I\'d like to leave. I\'m relying on you. Earlier today. I\'m flying out. Really not sure. Library books are due."'),
    ("D3", "negative", "ly_li_la_sounds", 18,
     'Say each: "Lying. Fly. Apply. Multiply. Family. Reply. Comply. Last. Listen. Lying down."'),

    # Section E: Negative — conversation
    ("E1", "negative", "natural_conversation", 60,
     'Talk for 60 seconds, normal conversation — about your day, weekend, food, whatever. DO NOT say "Lyra" or anything close to it.'),

    # Section F: Negative — household sounds
    ("F1", "negative", "tv_music_alone", 30,
     'Turn TV/music on at typical viewing volume. Sit silent for 30s.'),
    ("F2", "negative", "household_sounds", 30,
     'Make household noise for 30s: close a door, run water, type on a keyboard, drop a light object. Don\'t talk.'),
    ("F3", "negative", "bodily_sounds", 15,
     'Cough twice. Fake-sneeze. Laugh. Yawn loudly. Anything bodily. Don\'t say words.'),

    # Section G: Ambient
    ("G1", "ambient", "quiet_room", 60,
     'Sit completely silent in this room for 60s. Phones off / silenced. Hold still.'),
    ("G2", "ambient", "other_room_quiet", 30,
     'Carry LaiRA to a different room. Sit silent for 30s.'),
]


def capture_to_wav(out_path: Path, duration_s: float):
    """Record at native 48kHz mono via arecord, then downsample to 16kHz mono int16."""
    cmd = [
        "arecord",
        "-D", CAPTURE_DEVICE,
        "-c", "1",
        "-r", str(CAPTURE_RATE),
        "-f", "S16_LE",
        "-d", str(int(duration_s)),
        TMP_PATH,
    ]
    env = os.environ.copy()
    env["ALSA_CONFIG_PATH"] = "/usr/share/alsa/alsa.conf"
    subprocess.run(cmd, check=True, env=env, stderr=subprocess.DEVNULL)

    with wave.open(TMP_PATH, "rb") as w:
        raw = w.readframes(w.getnframes())
        sr_in = w.getframerate()
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if sr_in != TARGET_RATE:
        gcd = int(np.gcd(sr_in, TARGET_RATE))
        pcm = resample_poly(pcm, TARGET_RATE // gcd, sr_in // gcd)
    pcm = np.clip(pcm, -32768, 32767).astype(np.int16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_RATE)
        w.writeframes(pcm.tobytes())
    try:
        os.remove(TMP_PATH)
    except FileNotFoundError:
        pass


def main():
    result = subprocess.run(
        ["systemctl", "is-active", "laira.service"],
        capture_output=True, text=True,
    )
    if result.stdout.strip() == "active":
        print("ERROR: laira.service is running — it'll hold the mic.")
        print("Stop it first: sudo systemctl stop laira.service")
        return 1

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    total_s = sum(c[3] for c in CURRICULUM)
    print(f"\nWake-word training data collection")
    print(f"Output:    {OUT_BASE}")
    print(f"Steps:     {len(CURRICULUM)}")
    print(f"Total rec: {total_s}s ({total_s // 60}m {total_s % 60}s)")
    print(f"\nFor each step: read the prompt fully, then press Enter to start.")
    print(f"The recording auto-stops after the listed duration. If a step")
    print(f"goes wrong, just continue — you can re-run individual steps later.\n")
    input("Press Enter to begin curriculum...")

    for i, (step_id, category, label, duration_s, prompt) in enumerate(CURRICULUM, 1):
        out_path = OUT_BASE / category / f"{step_id}_{label}.wav"
        print(f"\n[{i:2d}/{len(CURRICULUM)}]  {step_id}  ({category})  duration={duration_s}s")
        print(f"   WHAT TO DO: {prompt}")
        if out_path.exists():
            print(f"   (existing file at {out_path} — will overwrite)")
        input(f"   Press Enter when ready... ")
        print(f"   RECORDING for {duration_s}s — GO!")
        try:
            capture_to_wav(out_path, duration_s)
        except subprocess.CalledProcessError as e:
            print(f"   ERROR capturing: {e}. Skipping. Re-run this step later.")
            continue
        print(f"   ✓ Saved: {out_path.name}")

    print(f"\n{'=' * 60}")
    print(f"Done! Audio collected at {OUT_BASE}")
    print(f"Run: sudo systemctl start laira.service  (to bring her back online)")
    print(f"Then tell Claude 'done' to proceed with training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
