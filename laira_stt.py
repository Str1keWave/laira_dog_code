"""
laira_stt.py — wake-word + command STT pipeline.

Replaces the old Vosk pipeline. Three-stage cascade:

  1. **openWakeWord** runs always-on against 80ms audio chunks. Dedicated
     wake-word NN, <4% CPU on Pi, very accurate.
  2. **Silero VAD** (ships with openwakeword) detects end-of-speech via
     voice-activity dropoff. Tells us when to stop recording the command.
  3. **STT**: either gpt-4o-mini-transcribe (cloud, near-perfect accuracy,
     ~300-700ms latency) OR faster-whisper tiny.en (local, free, ~0.5-1s,
     stumbles on proper nouns). Swappable via LAIRA_STT_BACKEND env var.

Audio source: reused from aiortc AudioForkTrack (16kHz mono int16 after
resampling). No new PyAudio mic.

Callers: laira_go_wsprotected feeds audio chunks via `SttPipeline.feed()`.
On command completion, the pipeline invokes the caller-supplied
`on_command` async callback with the transcribed text.
"""

import asyncio
import io
import os
import time
import wave
from collections import deque

import numpy as np
import laira_phonetic


# Configuration knobs. Lifted here so they can be tuned without touching
# the state-machine logic below.
WAKE_THRESHOLD = 0.4              # openwakeword score needed to trigger
# Sub-threshold peaks above this get logged (rate-limited) so we can
# diagnose recall failures: "I said lyra and nothing happened" → look
# at the journal for near-miss scores and decide whether to drop the
# threshold further or improve mic placement.
WAKE_NEAR_MISS_LOG_THRESHOLD = 0.25
WAKE_NEAR_MISS_MIN_INTERVAL_S = 1.0
WAKE_MODEL_NAME = os.environ.get("LAIRA_WAKE_MODEL", "hey_jarvis_v0.1")
# ^ env-overridable so once we train a custom "laira" model we just set
# LAIRA_WAKE_MODEL=laira in .env and restart — no code change.

# Lowered from 0.4 → 0.3 after observing recordings where the user
# spoke at moderate volume / from across the room, max VAD prob across
# the whole recording was 0.3-0.39, and VAD reported zero speech chunks
# even though the audio was clearly speech. Effect: more borderline
# speech registers, fewer recordings get prematurely truncated by
# silence detection. Trade-off: marginally more false-trigger of "is
# speech" on noise, but the silence-bump-to-LONG still requires VAD
# agreement so the noise-stretch bug stays fixed.
VAD_SPEECH_THRESHOLD = 0.3        # silero: >= threshold = speech
# 0.5 was missing quieter post-wake speech on the Pi's webcam mic — the
# wake word fired but the follow-up command didn't trip VAD, so the
# pipeline stayed in pre-speech mode for the full 4s window and Whisper
# got mostly silence to hallucinate from. 0.4 catches more of the user's
# command audio at the cost of slightly more false-positive "speech"
# from background noise (which only matters during the recording window
# and just delays the silence-end check by a beat).
# Adaptive silence threshold. Three regimes:
#   PRE_SPEECH (4000ms): user said the wake word but hasn't started the
#     actual command yet. Generous timeout so they can pause to think
#     ("lyra ... sit down" with a beat between). Without this, a
#     300-700ms gap between wake and command would terminate recording
#     with only the wake-word audio and no command.
#   SHORT (600ms): user is mid-command, no mid-phrase pauses yet.
#     Quick end so "sit" doesn't pay a 1.2s tax.
#   LONG (1200ms): user has already paused and resumed mid-command,
#     so they're giving a multi-clause utterance ("turn around until
#     you see me"). Long enough to absorb breath pauses.
SILENCE_END_MS_PRE_SPEECH = int(os.environ.get("LAIRA_SILENCE_END_MS_PRE_SPEECH", "4000"))
# Bumped from 600 → 900ms after observing quiet/distant speech
# cases where speech genuinely contains 200-400ms gaps between
# syllables (e.g., "Lyra... stand up"). 600ms was too aggressive —
# truncated recordings to 1.5-2s, capturing only the loudest
# syllable and missing the rest. 900ms gives borderline speech
# room to register without making fast commands feel laggy.
SILENCE_END_MS_SHORT = int(os.environ.get("LAIRA_SILENCE_END_MS_SHORT", "900"))
SILENCE_END_MS_LONG  = int(os.environ.get("LAIRA_SILENCE_END_MS_LONG",  "1200"))

# Per-chunk RMS threshold for speech detection. Silero VAD on this Pi
# webcam mic is unreliable — it routinely misses real speech (Whisper
# transcribed full commands while VAD insisted on silence). Pair Silero
# with a simple energy floor so we don't cut off recordings mid-
# utterance whenever Silero gets shy: a chunk counts as speech if
# EITHER VAD fires above its threshold OR the chunk's RMS is above
# this floor.
#
# Measured noise floor on this mic (idle, no speech): min=84 med=130
# max=180. Threshold needs to sit comfortably above 180 to avoid false
# positives on silence, while still triggering on voiced speech chunks
# (which peak well above this on real utterances). 300 gives ~120 RMS
# of headroom over the noisiest silence chunk.
# Raised from 300 to 600 after observing recordings where med RMS=189
# but noise peaks (>=300) were resetting the silence-detection counter
# even when no actual speech was present, stretching a 0.7s "stand up"
# command to a 7.3s recording. Real speech in this rig hits 800+; the
# bigger gap above the noise peaks (typically 500ish) makes the silence
# counter actually count silence.
RMS_SPEECH_THRESHOLD = int(os.environ.get("LAIRA_RMS_SPEECH_THRESHOLD", "600"))
COMMAND_MAX_MS = 12000            # safety cap — stop recording if user rambles
COMMAND_MIN_MS = 300              # too-short = probably a false positive wake

# Policy: don't cull audio with ms-based heuristics. Earlier iterations
# tried to surgically skip the "wake-word tail" (WAKE_TAIL_SKIP_MS) or
# start recording exactly at wake detection (PREROLL_MS=0) to avoid
# feeding wake-word fragments into Whisper. Both backfired — fragments
# are WORSE than complete words because Whisper invents filler for them
# ("…vis" → "Just"), and tail-skipping clips the first syllable when
# users run wake+command together.
#
# New policy: pre-roll enough audio to capture the ENTIRE wake word
# before the detection fires. Whisper gets a complete "hey jarvis turn
# around" — transcribes cleanly, no inventing. Downstream (Opus's
# robustness + extract_wake_word_command safety-net in laira_brain)
# correctly ignores or strips the wake-word prefix.
PREROLL_MS = int(os.environ.get("LAIRA_PREROLL_MS", "1000"))
# Kept as a knob in case we ever need it back, but defaulted off.
WAKE_TAIL_SKIP_MS = int(os.environ.get("LAIRA_WAKE_TAIL_SKIP_MS", "0"))

# openwakeword expects 80ms chunks at 16kHz = 1280 int16 samples
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280
CHUNK_MS = 1000 * CHUNK_SAMPLES // SAMPLE_RATE  # 80

# ── STT backend ─────────────────────────────────────────────────────────
# "openai" → gpt-4o-(mini-)transcribe cloud call. ~300-1500ms total
#            latency, good accuracy, $0.003-0.015/min. Requires
#            OPENAI_API_KEY. Hallucinates from quiet/silent audio.
# "local"  → faster-whisper tiny.en on Pi 5 CPU, ~0.5-1s per command,
#            free, offline. Mishears proper nouns + short words badly.
# "modal_parakeet" → NVIDIA Parakeet TDT 0.6B v2 on a self-hosted Modal
#            L4. Comparable WER to Whisper-large but ~10x faster (RTF
#            0.05) AND specifically trained on non-speech-noise so it
#            doesn't hallucinate Korean/Chinese from quiet audio. Uses
#            a streaming WebSocket protocol — Pi opens a session at
#            wake-fire, streams audio chunks during the command, closes
#            on VAD silence. Requires LAIRA_PARAKEET_WS_URL.
STT_BACKEND = os.environ.get("LAIRA_STT_BACKEND", "openai").lower()
OPENAI_TRANSCRIBE_MODEL = os.environ.get("LAIRA_OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
OPENAI_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"
PARAKEET_WS_URL = os.environ.get("LAIRA_PARAKEET_WS_URL", "")

# Stereo audio handling. The C960 webcam exposes FL+FR mics ~5cm apart
# on the front of the camera; we now capture both channels and use the
# spatial information for:
#
#   - Direction-of-arrival (DOA): which side of LaiRA the user spoke
#     from. Always computed and passed to the brain regardless of the
#     beamform flag below — gives the orchestrator a "voice came from
#     your left" hint so commands like "come over to me" don't 50/50
#     guess which way to rotate.
#
#   - Beamforming (optional): combine the two channels in a way that
#     emphasizes front-axis speech and rejects off-axis noise. Helps
#     in theory; needs real-world testing to confirm. Set
#     LAIRA_STEREO_BEAMFORM=1 to enable, 0 (default) for plain mono
#     downmix. Easy to flip back if it makes things worse.
STEREO_BEAMFORM = os.environ.get("LAIRA_STEREO_BEAMFORM", "0").strip() == "1"
# DOA disabled by default. Empirical testing on the C960 (the dog's
# webcam) showed both channels carry essentially identical audio
# regardless of source position — GCC-PHAT peak sits at lag=0 for
# hard-left, hard-right, and front sources alike. The two grills
# either feed a single capsule, are too close to produce 16kHz-
# resolvable ITD, or hardware DSP destroys the phase relationship.
# Set LAIRA_STEREO_DOA=1 to re-enable if a different mic gets
# wired up later. The compute_doa code stays in place for that
# case, but is skipped here unless the flag is on.
DOA_ENABLED = os.environ.get("LAIRA_STEREO_DOA", "0").strip() == "1"

# Only used when STT_BACKEND == "local":
WHISPER_MODEL_SIZE = os.environ.get("LAIRA_WHISPER_SIZE", "tiny.en")
WHISPER_COMPUTE_TYPE = "int8"

# Calibration target peak fraction for the preroll-derived gain. The
# wake word's peak in the preroll is normalized to this fraction of
# int16 range; same gain is then applied to every chunk streamed
# during the command. 0.6 = -4.4dBFS, leaves headroom for command
# audio that might be slightly louder than the wake word.
PREROLL_CALIBRATION_PEAK = float(os.environ.get("LAIRA_PREROLL_CALIBRATION_PEAK", "0.6"))

# Module-level singletons (loaded at startup via start_warmup()).
_wake_model = None
_vad = None
_whisper = None
_models_ready = asyncio.Event()
_warmup_started = False

# Callback fired when models transition ready. Set externally (by
# laira_brain) so the browser can be notified. Takes a single arg:
# status string ("warming" | "ready" | "error") with optional detail.
_on_status_change = None


def set_status_callback(callback):
    """Register an async callback fn(status, detail) that fires at
    warmup start (status='warming'), end (status='ready'), or failure
    (status='error'). Used to broadcast STT status to the browser."""
    global _on_status_change
    _on_status_change = callback


def is_ready():
    """For callers that want to check synchronously if STT can handle
    audio — e.g., the audio-frame consumer drops frames when not ready
    to avoid queuing up 30s of stale audio."""
    return _models_ready.is_set()


async def wait_ready(timeout=None):
    """Await model readiness. Useful for startup sequencing."""
    await asyncio.wait_for(_models_ready.wait(), timeout)


def _resolve_wake_model(name):
    """Given a wake model name (LAIRA_WAKE_MODEL env value), return what
    to pass to openwakeword.Model(wakeword_models=[...]).

    Lookup order:
      1. ~/laira_code/wake_models/<name>.onnx  ← custom trained models
      2. ~/.openwakeword/resources/models/<name>.onnx  (shouldn't happen
         but covers manual drop-ins)
      3. Pretrained name lookup (openwakeword resolves it internally)

    Returns (resolved_value, source_str) where resolved_value is either
    an absolute path (for custom) or the bare name (for pretrained).
    source_str is a human-readable description for logging."""
    base = os.path.dirname(os.path.abspath(__file__))
    # Primary custom-model directory — Colab-trained .onnx files drop here
    custom_paths = [
        os.path.join(base, "wake_models", f"{name}.onnx"),
        os.path.expanduser(f"~/laira_code/wake_models/{name}.onnx"),
        os.path.expanduser(f"~/.openwakeword/{name}.onnx"),
    ]
    for p in custom_paths:
        if os.path.exists(p):
            return p, f"custom ({p})"
    # Fall back to openwakeword's pretrained lookup by name. Returns
    # the bare name, which openwakeword resolves against its bundled
    # resources/models/ directory.
    return name, f"pretrained ({name})"


def _load_models_sync():
    """Blocking: loads all ONNX + whisper models + runs JIT warmup
    inferences. Runs in a thread via start_warmup()'s asyncio.to_thread
    so the event loop stays responsive during the multi-second load."""
    global _wake_model, _vad, _whisper
    print(f"[stt] loading models (backend={STT_BACKEND})...")
    t0 = time.time()
    from openwakeword.model import Model as OWWModel
    from openwakeword.vad import VAD as OWWVAD
    wake_value, wake_source = _resolve_wake_model(WAKE_MODEL_NAME)
    print(f"[stt] wake model source: {wake_source}")
    try:
        _wake_model = OWWModel(
            wakeword_models=[wake_value],
            inference_framework="onnx",
            vad_threshold=0.0,
        )
    except Exception as e:
        # If a user sets LAIRA_WAKE_MODEL=lyra before dropping the
        # .onnx in place, we shouldn't crash the whole service. Fall
        # back to the pretrained default and log loudly.
        print(f"[stt] FAILED to load wake model {wake_source}: {e!r}")
        print(f"[stt] falling back to pretrained 'hey_jarvis_v0.1'")
        _wake_model = OWWModel(
            wakeword_models=["hey_jarvis_v0.1"],
            inference_framework="onnx",
            vad_threshold=0.0,
        )
    _vad = OWWVAD()
    if STT_BACKEND == "local":
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    load_ms = int((time.time() - t0) * 1000)
    # JIT warmup: feed a single chunk of silence through each ONNX graph.
    # ONNX runtime's FIRST inference does graph optimization, adding a
    # ~1-3s latency spike on an otherwise hot path. This moves that cost
    # into startup where the user isn't waiting.
    print(f"[stt] models loaded ({load_ms}ms), warming JIT...")
    t1 = time.time()
    silence = np.zeros(CHUNK_SAMPLES, dtype=np.int16)
    try:
        _wake_model.predict(silence)
        _vad.predict(silence)
        if _whisper is not None:
            # Whisper JIT is also costly on first transcribe. 1.5s of
            # silence = a realistic minimum clip size.
            dummy_pcm = np.zeros(int(SAMPLE_RATE * 1.5), dtype=np.float32)
            list(_whisper.transcribe(
                dummy_pcm, language="en", beam_size=1,
                vad_filter=False, condition_on_previous_text=False,
            )[0])  # drain the generator
    except Exception as e:
        print(f"[stt] warmup inference failed (non-fatal): {e}")
    warm_ms = int((time.time() - t1) * 1000)
    total_ms = load_ms + warm_ms
    tail = f"whisper={WHISPER_MODEL_SIZE}/{WHISPER_COMPUTE_TYPE}" if STT_BACKEND == "local" else f"transcribe={OPENAI_TRANSCRIBE_MODEL} (cloud)"
    print(f"[stt] ready in {total_ms}ms (load={load_ms}ms, warm={warm_ms}ms): wake={WAKE_MODEL_NAME}, {tail}")


async def start_warmup():
    """Kick off model load + warmup in a background task. Safe to call
    multiple times — second call is a no-op. Returns immediately; caller
    can await wait_ready() if they need to block until done."""
    global _warmup_started
    if _warmup_started:
        return
    _warmup_started = True
    asyncio.create_task(_warmup_loop())


async def _warmup_loop():
    """The background task that actually loads + warms. Broadcasts
    status transitions so the browser can gate UI on readiness."""
    global _on_status_change
    if _on_status_change is not None:
        try: await _on_status_change("warming", f"loading {WAKE_MODEL_NAME} + STT backend")
        except Exception as e: print(f"[stt] status callback (warming) failed: {e}")
    try:
        await asyncio.to_thread(_load_models_sync)
    except Exception as e:
        print(f"[stt] fatal during warmup: {e!r}")
        if _on_status_change is not None:
            try: await _on_status_change("error", str(e))
            except Exception: pass
        return
    _models_ready.set()
    # Start the Modal Parakeet persistent client now that local models
    # are loaded. The client opens its WebSocket to Modal in the
    # background — by the time the user's first wake fires, the WS is
    # already up, so begin_recording is just a fast "open" json send
    # rather than a full TLS handshake. This is the fix for the
    # interrupt-path latency outlier.
    ensure_parakeet_client()
    if _on_status_change is not None:
        try: await _on_status_change("ready", "")
        except Exception as e: print(f"[stt] status callback (ready) failed: {e}")


# Legacy-compatible alias. If anything still calls _ensure_models()
# synchronously (e.g. on first audio frame before warmup completes),
# it's now a check-only: models either are or aren't ready. Callers
# that want to block should await wait_ready() instead.
def _ensure_models():
    if not _models_ready.is_set():
        raise RuntimeError("laira_stt models not ready — call start_warmup() at startup and check is_ready() before feeding audio")


class SttPipeline:
    """One instance per laira process. Holds the state machine and a
    pre-roll ring buffer. Feeds 16kHz int16 mono audio in 80ms chunks.

    State flow:
      IDLE ─(wake score ≥ threshold)→ WAKE_TAIL_SKIP
      WAKE_TAIL_SKIP ─(skipped WAKE_TAIL_SKIP_MS)→ RECORDING
      RECORDING ─(SILENCE_END_MS silence or COMMAND_MAX_MS)→ IDLE

    WAKE_TAIL_SKIP exists to drop the trailing ~150ms of wake-word audio
    that's still in flight when openWakeWord's confidence crosses
    threshold — without it, Whisper sees the wake-word fragment and
    hallucinates a prepended word ('Just turn around')."""

    IDLE = "idle"
    WAKE_TAIL_SKIP = "wake_tail_skip"
    RECORDING = "recording"

    def __init__(self, on_command, on_wake=None, on_drop=None):
        """on_command: async callback(text: str) invoked when a command
        has been captured + transcribed.

        on_wake: optional async callback (no args) fired the instant
        the wake word triggers, BEFORE recording / transcription. Used
        by the brain to (a) interrupt any active plan immediately so
        the user feels feedback within ~80ms instead of waiting for
        STT to complete, and (b) twitch a paw motor for auditory
        confirmation.

        on_drop: optional async callback(reason: str) fired when a
        recording is dropped before it would have been forwarded to
        the brain — e.g., no speech detected, or transcription looked
        like a hallucination. Used to surface a 'try again' message
        in the browser console so the user knows their wake word fired
        but their command didn't make it through."""
        self.on_command = on_command
        self.on_wake = on_wake
        self.on_drop = on_drop
        self.state = SttPipeline.IDLE
        # Pre-roll ring buffer of recent audio chunks — kept at 0 length
        # by default (see PREROLL_MS comment).
        preroll_chunks = max(1, PREROLL_MS // CHUNK_MS) if PREROLL_MS > 0 else 1
        self._preroll = deque(maxlen=preroll_chunks)
        # Command buffer: list of int16 chunks concatenated on end-of-cmd.
        self._cmd_chunks = []
        self._cmd_ms = 0
        self._silence_ms = 0
        # Current silence-to-end threshold — bumped from SHORT to LONG
        # the first time speech resumes after silence (= user took a
        # mid-phrase pause, so they're giving a longer command).
        self._silence_threshold_ms = SILENCE_END_MS_SHORT
        self._skip_ms_remaining = 0   # for WAKE_TAIL_SKIP state
        # Pending partial chunk: if aiortc hands us audio that doesn't
        # line up with our 1280-sample boundary, we buffer the remainder.
        self._pending = np.empty(0, dtype=np.int16)
        # Rate-limit timestamp for sub-threshold near-miss logging.
        self._last_near_miss_log_t = 0.0
        # Whether VAD has detected any speech since the last wake fired.
        # Used to apply the PRE_SPEECH-vs-SHORT silence threshold rule.
        self._has_speech_after_wake = False
        # Diagnostic counters: how many wake inferences have we run, and
        # what was the peak score in the most recent window? Used to
        # confirm the pipeline is actually running during plan execution
        # (vs. being starved by a blocked stt_worker).
        self._wake_inferences_count = 0
        self._wake_max_score_window = 0.0
        self._last_wake_diag_t = 0.0
        # Whether a Modal Parakeet recording session is currently active
        # on the persistent client. True between begin_recording and
        # end_recording_and_get_text. Used to skip push_chunk when the
        # backend isn't Parakeet, and to know when to call
        # end_recording_and_get_text in _finish_recording.
        self._parakeet_active = False
        # Calibration gain derived from the preroll on each wake fire.
        # Applied to every chunk pushed via the persistent client so
        # Modal sees audio at the level the model was trained on.
        self._stream_gain = 1.0
        # Stereo recording buffer: parallel L and R channel chunks
        # accumulated during RECORDING. Used at finish-recording time
        # to compute direction-of-arrival via interaural level
        # difference. Cleared on each new wake fire.
        self._cmd_chunks_L = []
        self._cmd_chunks_R = []
        # Stereo preroll. Parallel L and R deques sized identically to
        # self._preroll. Lets us preserve stereo info across the
        # wake-word audio, in case we ever want to compute DOA from
        # the wake utterance itself.
        preroll_chunks = max(1, PREROLL_MS // CHUNK_MS) if PREROLL_MS > 0 else 1
        self._preroll_L = deque(maxlen=preroll_chunks)
        self._preroll_R = deque(maxlen=preroll_chunks)
        # Pending partial stereo chunks for the boundary aligner. The
        # mono _pending exists for size-misalignment (chunks not
        # divisible by CHUNK_SAMPLES); add per-channel equivalents.
        self._pending_L = np.empty(0, dtype=np.int16)
        self._pending_R = np.empty(0, dtype=np.int16)

    async def _safe_on_wake(self):
        try:
            await self.on_wake()
        except Exception as e:
            print(f"[stt] on_wake callback failed: {e!r}")

    async def feed_stereo(self, pcm_L, pcm_R):
        """Feed 16kHz int16 stereo audio (L and R channels separately,
        same length). Caller has resampled and de-interleaved. We buffer
        per-channel until we have CHUNK_SAMPLES (1280, 80ms), then
        derive a mono signal for wake/VAD/STT and accumulate the
        stereo pair into the recording buffer for DOA computation at
        finish-recording time."""
        if not _models_ready.is_set():
            return
        if pcm_L.size == 0 or pcm_L.size != pcm_R.size:
            return
        if self._pending_L.size:
            pcm_L = np.concatenate([self._pending_L, pcm_L])
            pcm_R = np.concatenate([self._pending_R, pcm_R])
            self._pending_L = np.empty(0, dtype=np.int16)
            self._pending_R = np.empty(0, dtype=np.int16)
        n_chunks, tail = divmod(pcm_L.size, CHUNK_SAMPLES)
        if tail:
            self._pending_L = pcm_L[-tail:].copy()
            self._pending_R = pcm_R[-tail:].copy()
            pcm_L = pcm_L[:-tail]
            pcm_R = pcm_R[:-tail]
        for i in range(n_chunks):
            chunk_L = pcm_L[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
            chunk_R = pcm_R[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
            chunk_mono = self._stereo_to_mono(chunk_L, chunk_R)
            await self._on_chunk(chunk_mono, chunk_L=chunk_L, chunk_R=chunk_R)

    @staticmethod
    def _stereo_to_mono(chunk_L, chunk_R):
        """Combine the two channels into a single mono signal that
        wake/VAD/STT will consume. Two modes:

          - STEREO_BEAMFORM=False (default): plain (L+R)/2 average.
            Same content as the old mono capture, no SNR change.
          - STEREO_BEAMFORM=True: simple delay-and-sum beamform pointed
            straight forward. Sounds arriving in-phase from front/back
            constructively add (~+6dB), sounds from sides partially
            cancel because of their interaural time difference. Crude
            but free; emphasizes the "user is in front of LaiRA" case.
            Output scaled to keep peak in int16 range.

        Switch between them via LAIRA_STEREO_BEAMFORM env var. Easy
        to revert if real-world testing shows it hurts more than helps."""
        if STEREO_BEAMFORM:
            # Coherent sum, then scale by 0.55 so the +6dB front-axis
            # boost doesn't routinely clip on already-loud speech.
            mono = (chunk_L.astype(np.int32) + chunk_R.astype(np.int32))
            mono = (mono * 0.55).astype(np.int32)
            return np.clip(mono, -32768, 32767).astype(np.int16)
        else:
            mono = (chunk_L.astype(np.int32) + chunk_R.astype(np.int32)) // 2
            return mono.astype(np.int16)

    # Legacy mono-only entry point. If anything still calls .feed(pcm)
    # with a single mono buffer, route it through the stereo path by
    # duplicating the channel so DOA reports "center" (no spatial info).
    async def feed(self, pcm_int16):
        if pcm_int16.size == 0:
            return
        await self.feed_stereo(pcm_int16, pcm_int16)

    async def _on_chunk(self, chunk, chunk_L=None, chunk_R=None):
        """Handle one 80ms chunk. chunk is the mono signal that wake/
        VAD/STT will see; chunk_L and chunk_R are the original stereo
        channels (same length as chunk) used to populate the stereo
        recording buffer for DOA. Mono-only callers leave them None."""
        if self.state == SttPipeline.IDLE:
            # Always cache into pre-roll so we can prepend to the command
            # buffer if wake fires on THIS chunk.
            self._preroll.append(chunk.copy())
            if chunk_L is not None and chunk_R is not None:
                self._preroll_L.append(chunk_L.copy())
                self._preroll_R.append(chunk_R.copy())
            # Run wake-word inference.
            scores = _wake_model.predict(chunk)
            best_model, best_score = max(scores.items(), key=lambda kv: kv[1])
            # Diagnostic: count + track max score AND track RMS of this
            # chunk so we can characterize the mic's noise floor (when
            # idle = nobody talking) vs speech level. Used to pick a
            # sane RMS_SPEECH_THRESHOLD.
            self._wake_inferences_count += 1
            if best_score > self._wake_max_score_window:
                self._wake_max_score_window = best_score
            chunk_rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))) if chunk.size else 0.0
            if not hasattr(self, "_idle_rms_samples"):
                self._idle_rms_samples = []
            self._idle_rms_samples.append(chunk_rms)
            now = time.time()
            if now - self._last_wake_diag_t >= 5.0:
                if self._last_wake_diag_t > 0:
                    elapsed = now - self._last_wake_diag_t
                    rate = self._wake_inferences_count / elapsed if elapsed > 0 else 0
                    rms_arr = self._idle_rms_samples
                    rms_min = int(min(rms_arr)) if rms_arr else 0
                    rms_max = int(max(rms_arr)) if rms_arr else 0
                    rms_med = int(sorted(rms_arr)[len(rms_arr)//2]) if rms_arr else 0
                    print(f"[stt:diag] wake inference {rate:.1f}/s, max_score_window={self._wake_max_score_window:.3f}, idle_rms min={rms_min} med={rms_med} max={rms_max}")
                self._last_wake_diag_t = now
                self._wake_inferences_count = 0
                self._wake_max_score_window = 0.0
                self._idle_rms_samples = []
            if best_score >= WAKE_THRESHOLD:
                print(f"[stt] wake triggered: {best_model} score={best_score:.3f}")
                # Fire-and-forget the wake callback. Awaiting it would
                # block the stt_worker → which blocks the asyncio loop's
                # progress → which starves the audio pump → which causes
                # a brief audible drop on the browser's WebRTC playback
                # AND chops the start of the user's command audio. The
                # interrupt + paw twitch still happen with the same
                # latency (they're just scheduled rather than awaited).
                if self.on_wake is not None:
                    asyncio.create_task(self._safe_on_wake())
                self._start_recording()
            elif best_score >= WAKE_NEAR_MISS_LOG_THRESHOLD:
                # Sub-threshold peak — log occasionally so we can see what
                # the model thinks of failed wake attempts. Rate-limited
                # to once per WAKE_NEAR_MISS_MIN_INTERVAL_S seconds because
                # peaks tend to cluster (a single utterance produces ~5-10
                # frames above threshold).
                now = time.time()
                if (now - self._last_near_miss_log_t) >= WAKE_NEAR_MISS_MIN_INTERVAL_S:
                    self._last_near_miss_log_t = now
                    print(f"[stt] wake near-miss: {best_model} score={best_score:.3f} (threshold={WAKE_THRESHOLD})")
        elif self.state == SttPipeline.WAKE_TAIL_SKIP:
            # Discarding audio — we're past the wake detection but the
            # trailing audio of the wake word itself is still in flight.
            # Skipping avoids polluting Whisper with wake-word fragments.
            self._skip_ms_remaining -= CHUNK_MS
            if self._skip_ms_remaining <= 0:
                self.state = SttPipeline.RECORDING
        else:  # RECORDING
            self._cmd_chunks.append(chunk.copy())
            # Also accumulate the per-channel stereo audio for DOA
            # computation at finish-recording time. Skipped if caller
            # didn't provide stereo (mono fallback path).
            if chunk_L is not None and chunk_R is not None:
                self._cmd_chunks_L.append(chunk_L.copy())
                self._cmd_chunks_R.append(chunk_R.copy())
            # Also tee to the Modal Parakeet persistent client if
            # active — ships the chunk live over the always-open WS,
            # so Modal can accumulate chunks during recording rather
            # than receiving them all in a burst at close time.
            if self._parakeet_active:
                client = ensure_parakeet_client()
                if client is not None:
                    client.push_chunk(chunk)
            self._cmd_ms += CHUNK_MS
            # Speech detection: take the OR of Silero VAD and a simple
            # per-chunk RMS energy floor. Silero is precise when it
            # fires but routinely misses real speech on this mic.
            # Energy is a dumb-but-honest fallback: if the audio is
            # measurably louder than ambient noise, it's speech.
            # VAD on a gain-boosted copy of the chunk: Silero is trained
            # on broadcast-level audio and silently misses quiet/distant
            # speech (max_prob stays under threshold even when the user
            # IS clearly talking — confirmed in real-world recordings
            # where transcripts came back empty because VAD never fired
            # and silence-detection terminated the recording at the
            # first valley between syllables). _stream_gain was already
            # computed from the preroll's "lyra" loudness for exactly
            # this reason — apply it here so VAD sees audio at the
            # level it was trained on. RMS stays on the raw chunk so
            # background noise doesn't get amplified into false-positive
            # speech (which would re-trigger the silence-stretch bug).
            if (self.state == SttPipeline.RECORDING
                    and self._stream_gain > 1.0
                    and chunk.size):
                chunk_for_vad = (
                    chunk.astype(np.float32) * self._stream_gain
                ).clip(-32768, 32767).astype(np.int16)
            else:
                chunk_for_vad = chunk
            speech_prob = _vad.predict(chunk_for_vad)
            chunk_rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))) if chunk.size else 0.0
            # Tracking per-chunk stats so we can characterize what
            # speech-during-recording actually looks like vs. silence.
            if not hasattr(self, "_rec_chunk_rms_samples"):
                self._rec_chunk_rms_samples = []
                self._rec_vad_max = 0.0
                self._rec_vad_speech_count = 0
                self._rec_rms_speech_count = 0
            self._rec_chunk_rms_samples.append(int(chunk_rms))
            if speech_prob > self._rec_vad_max:
                self._rec_vad_max = speech_prob
            vad_says_speech = speech_prob >= VAD_SPEECH_THRESHOLD
            rms_says_speech = chunk_rms >= RMS_SPEECH_THRESHOLD
            if vad_says_speech:
                self._rec_vad_speech_count += 1
            if rms_says_speech:
                self._rec_rms_speech_count += 1
            is_speech = vad_says_speech or rms_says_speech
            if is_speech:
                # First speech since wake → flip out of pre-speech mode
                # (no longer wait the long 4s for "the actual command").
                self._has_speech_after_wake = True
                # Adaptive silence rule: if the user just broke a silence
                # (speech resumed after accumulated quiet), they're
                # giving a multi-clause command. Bump the threshold so
                # we don't clip the next clause.
                #
                # Only bump on VAD-confirmed speech. RMS-only triggers
                # are too noise-prone — a single RMS spike during ambient
                # noise would otherwise stretch the recording from 600ms
                # to 1200ms silence-tolerance, ballooning latency on
                # simple commands. With VAD agreement, the bump fires
                # only when actual speech resumed.
                if (vad_says_speech
                        and self._silence_ms > 0
                        and self._silence_threshold_ms == SILENCE_END_MS_SHORT):
                    self._silence_threshold_ms = SILENCE_END_MS_LONG
                self._silence_ms = 0
            else:
                self._silence_ms += CHUNK_MS
            # Effective silence threshold for THIS chunk's end-check:
            # before the user has uttered any actual command syllable
            # (only "lyra" + silence so far), we wait the longer
            # PRE_SPEECH window so a thinking pause doesn't truncate
            # to wake-word-only audio. After first speech, we drop back
            # to the regular SHORT/LONG state machine.
            effective_thresh = (self._silence_threshold_ms
                                if self._has_speech_after_wake
                                else SILENCE_END_MS_PRE_SPEECH)
            # End conditions: enough trailing silence, OR max-duration hit.
            end = False
            if self._cmd_ms >= COMMAND_MIN_MS and self._silence_ms >= effective_thresh:
                end = True
                pre = "pre-speech" if not self._has_speech_after_wake else "thresh"
                reason = f"silence ({self._silence_ms}ms, {pre}={effective_thresh})"
            elif self._cmd_ms >= COMMAND_MAX_MS:
                end = True
                reason = f"max duration ({self._cmd_ms}ms)"
            if end:
                await self._finish_recording(reason)

    def _start_recording(self):
        """Entering the post-wake lifecycle. If WAKE_TAIL_SKIP_MS > 0 we
        enter WAKE_TAIL_SKIP first and discard that many ms of audio
        (the tail of the wake-word utterance), then transition to
        RECORDING. If it's 0 we go straight to RECORDING."""
        if WAKE_TAIL_SKIP_MS > 0:
            self.state = SttPipeline.WAKE_TAIL_SKIP
            self._skip_ms_remaining = WAKE_TAIL_SKIP_MS
        else:
            self.state = SttPipeline.RECORDING
        # Seed command buffer from pre-roll (typically empty since
        # PREROLL_MS=0; kept for flexibility if we ever re-enable it).
        self._cmd_chunks = list(self._preroll)
        # Seed stereo buffers identically. self._preroll_L / _R have
        # been kept in lock-step with self._preroll by feed_stereo.
        self._cmd_chunks_L = list(self._preroll_L)
        self._cmd_chunks_R = list(self._preroll_R)
        self._cmd_ms = len(self._cmd_chunks) * CHUNK_MS
        self._silence_ms = 0
        # Fresh command starts with SHORT silence threshold; bumps to
        # LONG on first mid-phrase pause detected (see _on_chunk).
        # Until any speech is heard, we override with PRE_SPEECH (4s).
        self._silence_threshold_ms = SILENCE_END_MS_SHORT
        self._has_speech_after_wake = False

        # Preroll-calibrated gain. The preroll buffer contains the user
        # saying "lyra" + a slice of ambient before that — a real-time
        # measurement of voice level RIGHT NOW under THIS room's
        # conditions. Compute peak, derive gain to bring it to a target
        # level Whisper/Parakeet was trained on, and apply that ONE
        # gain to every chunk for the rest of the command. No pumping
        # artifacts (constant gain), no per-chunk normalization, no
        # spectral processing — just one multiplier that adapts per
        # wake fire.
        self._stream_gain = self._compute_preroll_gain(self._preroll)

        self._preroll.clear()
        self._preroll_L.clear()
        self._preroll_R.clear()
        # Reset openwakeword's internal state so the same wake-word
        # activation doesn't re-fire on adjacent chunks. The model keeps
        # a buffer internally that can score >threshold for a few
        # consecutive predictions after detection.
        try:
            _wake_model.reset()
        except AttributeError:
            # Older API: clear prediction_buffer dicts manually.
            if hasattr(_wake_model, "prediction_buffer"):
                for k in _wake_model.prediction_buffer:
                    _wake_model.prediction_buffer[k].clear()

        # If using Modal Parakeet, open a streaming session on the
        # PERSISTENT WebSocket. begin_recording is just two operations:
        # send a small "open" json + initialize per-recording state.
        # No TLS handshake at wake-time anymore — that was the cause of
        # the ~3.5s outlier on interrupt-path recordings (asyncio loop
        # contention during interrupt cleanup stalled the handshake).
        # If the WS is currently disconnected (cellular switch in
        # progress, etc.), begin_recording raises and we fall through
        # to the OpenAI batch path on the recording-end side.
        client = ensure_parakeet_client() if STT_BACKEND == "modal_parakeet" else None
        if client is not None:
            try:
                # Schedule begin_recording as a fire-and-forget task so we
                # don't block here. The "open" json gets sent ASAP. If
                # WS is disconnected, this raises inside the task — we'll
                # see the failure when end_recording_and_get_text() is
                # called and fall back at that point.
                asyncio.create_task(client.begin_recording(gain=self._stream_gain))
                self._parakeet_active = True
                # Push preroll into the persistent stream so Parakeet
                # sees the whole utterance including the wake word.
                for c in self._cmd_chunks:
                    client.push_chunk(c)
            except Exception as e:
                print(f"[stt] parakeet begin_recording scheduling failed: {e!r}")
                self._parakeet_active = False
        else:
            self._parakeet_active = False

    @staticmethod
    def _compute_preroll_gain(preroll_chunks):
        """Compute calibration gain from preroll audio. The preroll holds
        ~PREROLL_MS of audio captured BEFORE the wake-word inference
        triggered, so it contains the user's "lyra" utterance plus some
        ambient. Peak of this is a representative current voice-level
        measurement. Returns float gain factor."""
        if not preroll_chunks:
            return 1.0
        try:
            joined = np.concatenate(list(preroll_chunks))
        except Exception:
            return 1.0
        if joined.size == 0:
            return 1.0
        peak = int(np.max(np.abs(joined)))
        if peak == 0:
            return 1.0
        target_peak = int(PREROLL_CALIBRATION_PEAK * 32767)
        gain = target_peak / peak
        # Cap at 30x — beyond that the audio is mostly noise anyway and
        # huge gain just amplifies it into fake-speech-shaped noise.
        return max(1.0, min(30.0, gain))

    async def _finish_recording(self, reason):
        """Transcribe accumulated command audio and hand off to callback.
        Routes to cloud or local based on STT_BACKEND."""
        chunks = self._cmd_chunks
        had_speech = self._has_speech_after_wake
        # Compute direction-of-arrival from the per-channel stereo buffers
        # before we clear them. ILD-based — compare L vs R RMS over the
        # speech portion. Always done regardless of STEREO_BEAMFORM
        # because the brain wants the spatial hint either way.
        doa = self._compute_doa(self._cmd_chunks_L, self._cmd_chunks_R) if DOA_ENABLED else None
        self._cmd_chunks = []
        self._cmd_chunks_L = []
        self._cmd_chunks_R = []
        self._cmd_ms = 0
        self._silence_ms = 0
        self.state = SttPipeline.IDLE
        # Reset wake model so the same utterance doesn't re-fire on the
        # next background chunk.
        try:
            _wake_model.reset()
        except AttributeError:
            if hasattr(_wake_model, "prediction_buffer"):
                for k in _wake_model.prediction_buffer:
                    _wake_model.prediction_buffer[k].clear()
        if not chunks:
            return
        # Anti-hallucination guard: if VAD never detected speech during the
        # whole recording window, the audio is functionally silence + room
        # noise. Whisper-family models (including gpt-4o-mini-transcribe)
        # are prone to inventing content from such input — we've seen them
        # output Korean ("바이러스 상태"), random English words ("Lam",
        # "now"), or echo back the prompt. Just drop the recording and
        # log it; the user can say their command again.
        # Always transcribe so the user can see what was actually heard,
        # even when we're going to drop. Costs a few extra API calls on
        # silent / quiet audio but gives full visibility — was confusing
        # before to see "wake heard, no usable command" with no clue why.
        # Log per-chunk speech-detection stats from this recording so
        # we can see whether VAD or RMS was carrying speech detection,
        # and what RMS values speech chunks actually produced. Reset
        # per-recording state.
        if hasattr(self, "_rec_chunk_rms_samples") and self._rec_chunk_rms_samples:
            samples = self._rec_chunk_rms_samples
            samples_sorted = sorted(samples)
            n = len(samples)
            rms_min = samples_sorted[0]
            rms_med = samples_sorted[n // 2]
            rms_max = samples_sorted[-1]
            rms_p90 = samples_sorted[min(n - 1, int(n * 0.9))]
            print(
                f"[stt:rec_diag] {n} chunks, rms min={rms_min} med={rms_med} p90={rms_p90} max={rms_max}, "
                f"vad_speech_chunks={self._rec_vad_speech_count} (max_prob={self._rec_vad_max:.3f}), "
                f"rms_speech_chunks={self._rec_rms_speech_count} (>={RMS_SPEECH_THRESHOLD})"
            )
            self._rec_chunk_rms_samples = []
            self._rec_vad_max = 0.0
            self._rec_vad_speech_count = 0
            self._rec_rms_speech_count = 0
        if not chunks:
            return
        pcm_int16 = np.concatenate(chunks)
        duration_s = pcm_int16.size / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(pcm_int16.astype(np.float32) ** 2))) if pcm_int16.size else 0.0
        # Pre-flight quality flags — used to decide whether to forward to
        # the brain or just log+show. We always still transcribe so the
        # user sees what was heard.
        # Pre-speech VAD check removed: in practice, Silero VAD on this
        # mic's audio produces a lot of false negatives — it'd say
        # "no speech" while Whisper transcribed clean English. Trust
        # the non-Latin filter downstream as the hallucination guard
        # instead. The RMS check stays as a hard floor (truly silent
        # audio = silence + mic self-noise = guaranteed hallucination).
        pre_drop_reason = None
        if rms < 80:
            pre_drop_reason = f"truly silent (rms={rms:.0f})"
        # Loudness normalization. The Pi's webcam mic captures speech at
        # roughly -30 to -40 dBFS — about 25dB quieter than the audio
        # Whisper was trained on (broadcast/podcast levels around -20
        # dBFS). At those levels Whisper's confidence collapses and it
        # falls back to language-model hallucinations of plausible
        # commands. The audio IS intelligible (a human listener can
        # hear it fine), it just doesn't match Whisper's training
        # distribution. Solution: scale the recording so RMS hits a
        # target level Whisper expects, capped so peaks never clip.
        # No spectral processing, no per-chunk artifacts, no time-
        # varying dynamics — just one multiplication. Different from
        # the AGC we tried before (per-chunk, pumping artifacts) and
        # different from noisereduce (subtraction, spectral artifacts).
        pcm_int16_norm, gain = _normalize_loudness(pcm_int16)
        print(f"[stt] cmd captured: {duration_s:.2f}s, rms={rms:.0f}, gain={gain:.1f}x, end={reason} — transcribing via {STT_BACKEND}...")
        t0 = time.time()
        text = None
        try:
            if STT_BACKEND == "modal_parakeet" and self._parakeet_active:
                # All chunks already on the wire via the persistent WS.
                # end_recording_and_get_text just sends "close" and
                # awaits Modal's "final" message. Total latency is just
                # the close+inference roundtrip (~150-300ms typical).
                client = ensure_parakeet_client()
                self._parakeet_active = False
                try:
                    if client is None:
                        raise RuntimeError("parakeet client unavailable")
                    text = await client.end_recording_and_get_text(timeout=10.0)
                except Exception as e:
                    # Modal hiccup, WS dropped (cellular switch), etc.
                    # Fall back to OpenAI batch on the locally-buffered
                    # audio. Slower but reliable.
                    print(f"[stt] parakeet stream failed ({e!r}), falling back to openai batch")
                    text = await _transcribe_openai(pcm_int16_norm)
                # Empty-transcript fallback. Parakeet's noise-suppression
                # is aggressive — it returns "" rather than hallucinate
                # from quiet/borderline audio. But sometimes the user
                # was speaking and Parakeet just didn't catch it. Try
                # gpt-4o-transcribe on the same audio buffer; Whisper-
                # family models are tuned to extract speech from quieter
                # input. If Whisper also returns nothing, the audio
                # really was unintelligible and we drop normally.
                if not (text or "").strip():
                    print("[stt] === parakeet empty → trying gpt-4o-transcribe BACKUP ===")
                    backup_t0 = time.time()
                    try:
                        text = await _transcribe_openai(pcm_int16_norm)
                        backup_ms = int((time.time() - backup_t0) * 1000)
                        print(f"[stt] === gpt-4o BACKUP returned ({backup_ms}ms): {text!r} ===")
                    except Exception as e:
                        backup_ms = int((time.time() - backup_t0) * 1000)
                        print(f"[stt] === gpt-4o BACKUP failed ({backup_ms}ms): {e!r} ===")
                        text = ""
            elif STT_BACKEND == "local":
                pcm_f32 = pcm_int16_norm.astype(np.float32) / 32768.0
                text = await asyncio.to_thread(_transcribe_local, pcm_f32)
            else:
                text = await _transcribe_openai(pcm_int16_norm)
        except Exception as e:
            print(f"[stt] transcribe failed: {e!r}")
            self._parakeet_active = False
            return
        elapsed_ms = int((time.time() - t0) * 1000)
        text = (text or "").strip()
        print(f"[stt] transcribed ({elapsed_ms}ms): {text!r}")
        # Now decide drop vs forward. All drops include the actual heard
        # text so the user can see what the mic + Whisper got from this
        # attempt — useful for debugging "she keeps mishearing me".
        non_latin = [c for c in text if ord(c) > 0x024F]
        post_drop_reason = None
        if pre_drop_reason is not None:
            post_drop_reason = pre_drop_reason
        elif not text:
            post_drop_reason = "empty transcription"
        elif non_latin:
            sample = "".join(non_latin[:10])
            post_drop_reason = f"non-English chars (forced language=en) sample={sample!r}"
        if post_drop_reason is not None:
            heard = f" heard: {text!r}" if text else ""
            print(f"[stt] DROPPED ({post_drop_reason}){heard}")
            self._fire_drop(f"{post_drop_reason} — heard {text!r}" if text else f"{post_drop_reason}")
            return
        # Fire-and-forget the on_command. Awaiting it would block the
        # stt_worker chain — submit_text can take 1-3s when it does an
        # Opus call, which chops the audio pump and prevents wake
        # inference from running during that window. The brain has its
        # own _submit_lock to serialize. We pass `doa` (left/right/
        # center) so the brain can prepend a spatial hint to the
        # user_text before sending to Opus.
        # Phonetic correction: catches near-homophone mishearings like
        # "why down" → "lie down" without LLM help. Strict threshold
        # (≤1 phoneme), substring-gated so legit phrases pass through.
        # See laira_phonetic.py for vocabulary + rationale.
        try:
            corrected, reason = laira_phonetic.correct(text)
            if reason is not None:
                print(f"[stt:phonetic] {reason}")
                text = corrected
        except Exception as _pe:
            print(f"[stt:phonetic] correction failed (passing through): {_pe!r}")
        print(f"[stt] dispatching: text={text!r} doa={doa}")
        asyncio.create_task(self._safe_on_command(text, doa))


    def _fire_drop(self, reason):
        """Schedule on_drop without awaiting. Prevents the broadcast →
        WS write blocking the stt_worker (which would chop audio)."""
        if self.on_drop is None:
            return
        async def _go():
            try:
                await self.on_drop(reason)
            except Exception as e:
                print(f"[stt] on_drop callback failed: {e!r}")
        asyncio.create_task(_go())

    async def _safe_on_command(self, text, doa=None):
        try:
            # Try the new signature with doa first; fall back to the
            # old single-arg signature so legacy callers still work.
            try:
                await self.on_command(text, doa)
            except TypeError:
                await self.on_command(text)
        except Exception as e:
            print(f"[stt] on_command callback failed: {e!r}")

    @staticmethod
    def _compute_doa(chunks_L, chunks_R):
        """DOA via interaural time difference (ITD), measured by
        GCC-PHAT cross-correlation between the L and R channels.

        With mics ~5cm apart on the C960's front face and no head
        between them, ILD is bounded at <1dB even for hard-side
        sources (geometry / no acoustic shadow). ITD is the only
        usable cue: max ~146μs = ~2.3 samples at 16kHz. We compute
        GCC-PHAT (whitens the spectrum, robust to room reverb),
        find the peak lag, parabolically interpolate for sub-sample
        resolution, and threshold on the fractional lag.

        Sign convention: positive lag = L leads R = source on LEFT
        (sound hit L mic first). Negative lag = source on RIGHT.

        Returns "left", "right", "center", or None on degenerate input.
        """
        if not chunks_L or not chunks_R:
            return None
        try:
            L = np.concatenate(chunks_L).astype(np.float32)
            R = np.concatenate(chunks_R).astype(np.float32)
        except Exception:
            return None
        if L.size < 1024 or R.size < 1024:
            return None
        rms_L = float(np.sqrt(np.mean(L * L)))
        rms_R = float(np.sqrt(np.mean(R * R)))
        # Skip on near-silence — cross-correlation of mostly-noise
        # gives a meaningless peak.
        if max(rms_L, rms_R) < 80.0:
            print(f"[stt:doa] silent (rms_L={rms_L:.0f} rms_R={rms_R:.0f})")
            return "center"
        # GCC-PHAT: whitened cross-spectrum, then inverse FFT gives
        # a sharp peak at the time-difference of arrival.
        n = max(L.size, R.size)
        nfft = 1 << (2 * n - 1).bit_length()
        SL = np.fft.rfft(L, nfft)
        SR = np.fft.rfft(R, nfft)
        W = SL * np.conj(SR)
        W /= np.abs(W) + 1e-9
        cc = np.fft.irfft(W, nfft)
        # Reorder so lag 0 is at index nfft//2.
        cc = np.concatenate([cc[-(nfft // 2):], cc[:nfft // 2]])
        center_idx = nfft // 2
        # Search ±MAX_LAG_SAMPLES around 0. Physical ITD bound at
        # 16kHz is ~2.3 samples; window slightly wider for noise.
        MAX_LAG_SAMPLES = 4
        win_lo = center_idx - MAX_LAG_SAMPLES
        win_hi = center_idx + MAX_LAG_SAMPLES + 1
        win = cc[win_lo:win_hi]
        peak_local = int(np.argmax(win))
        peak_lag_int = peak_local - MAX_LAG_SAMPLES  # samples
        # Parabolic interpolation around peak for sub-sample precision.
        if 0 < peak_local < len(win) - 1:
            y0, y1, y2 = win[peak_local - 1], win[peak_local], win[peak_local + 1]
            denom = (y0 - 2 * y1 + y2)
            if abs(denom) > 1e-9:
                offset = 0.5 * (y0 - y2) / denom
                peak_lag = peak_lag_int + float(offset)
            else:
                peak_lag = float(peak_lag_int)
        else:
            peak_lag = float(peak_lag_int)
        # Threshold (samples). LAIRA_DOA_LAG_THRESHOLD env var lets us
        # tune in the field. Default 0.4 samples = ~25μs ITD = source
        # ~6° off-axis at this baseline (well above measurement noise
        # for clean GCC-PHAT but small enough to catch realistic angles).
        thresh = float(os.environ.get("LAIRA_DOA_LAG_THRESHOLD", "0.4"))
        ild_db = 20.0 * np.log10(rms_L / max(1.0, rms_R))
        diag = (f"lag={peak_lag:+.2f} samp rms_L={rms_L:.0f} "
                f"rms_R={rms_R:.0f} ild={ild_db:+.2f}dB")
        # Sign convention (verified empirically with R-lags-L test):
        # numpy FFT-based xcorr puts the peak at NEGATIVE lag when R
        # lags L (i.e., source on LEFT). So negative peak_lag = LEFT.
        if peak_lag <= -thresh:
            print(f"[stt:doa] left ({diag})")
            return "left"
        if peak_lag >= thresh:
            print(f"[stt:doa] right ({diag})")
            return "right"
        print(f"[stt:doa] center ({diag})")
        return "center"


def _normalize_loudness(pcm_int16, target_rms_frac=0.10, max_peak_frac=0.92):
    """Scale a recording so its RMS hits target_rms_frac of full-scale,
    while ensuring peaks stay below max_peak_frac. Returns (audio, gain).

    target_rms=0.10 corresponds to ~-20 dBFS RMS — typical broadcast /
    podcast level, comfortably inside Whisper's training distribution.
    max_peak=0.92 keeps headroom against peak clipping (which Whisper
    handles fine but is still avoidable).

    For very quiet recordings (e.g. user across the room from the Pi
    mic), this can apply 10-30x gain — making the recording sound
    "loud" to Whisper without changing the speech content. Single
    multiplicative gain, no spectral processing, no artifacts."""
    if pcm_int16.size == 0:
        return pcm_int16, 1.0
    pcm_f32 = pcm_int16.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(pcm_f32 ** 2)))
    peak = float(np.max(np.abs(pcm_f32)))
    if rms == 0 or peak == 0:
        return pcm_int16, 1.0
    gain_for_rms = target_rms_frac / rms
    max_gain_for_peak = max_peak_frac / peak
    gain = min(gain_for_rms, max_gain_for_peak)
    if gain <= 1.0:
        return pcm_int16, 1.0
    pcm_f32 = np.clip(pcm_f32 * gain, -1.0, 1.0)
    return (pcm_f32 * 32767.0).astype(np.int16), gain


def _preprocess_audio(pcm_int16):
    """Pre-transcription audio cleanup. Returns int16 PCM at the same
    sample rate.

    Two stages:
      1. High-pass filter at 80Hz. Voice is mostly 200-3000Hz; below
         80Hz is AC hum, low rumble, mic mount thumps, breath plosives.
         Stripping it doesn't lose any speech information and gives
         the noise reducer a cleaner signal to work with.
      2. Spectral noise gate via noisereduce. Estimates the stationary
         noise floor from the whole clip and subtracts it. Effective on
         HVAC, fan whine, ambient room hiss — anything constant. Less
         effective on transient noise (motor whine when LaiRA is
         moving) but doesn't make those clips worse.

    Both run in float32 internally; we round-trip to int16 for the
    callers (Whisper API and faster-whisper both accept int16/float32
    interchangeably, but the rest of the pipeline is int16).

    Imports inline so module load doesn't pay for these unless we
    actually run STT. scipy is fast (real C); noisereduce is pure
    Python wrapping scipy and is the slower of the two (~50ms on a
    3s clip on Pi 5)."""
    if pcm_int16.size == 0:
        return pcm_int16
    from scipy.signal import butter, sosfiltfilt
    import noisereduce as nr
    pcm_f32 = pcm_int16.astype(np.float32) / 32768.0
    # 80Hz high-pass, 4th order butterworth, zero-phase (filtfilt).
    sos = butter(4, 80.0, btype="highpass", fs=SAMPLE_RATE, output="sos")
    pcm_f32 = sosfiltfilt(sos, pcm_f32).astype(np.float32)
    # Stationary noise reduction. With stationary=True, noisereduce
    # estimates a single noise profile from the whole clip — works
    # well when the noise is roughly constant across the recording
    # (which is what we have: room hum, fan, mic self-noise).
    pcm_f32 = nr.reduce_noise(
        y=pcm_f32,
        sr=SAMPLE_RATE,
        stationary=True,
        prop_decrease=0.85,  # how aggressively to attenuate noise
    ).astype(np.float32)
    pcm_int16_out = np.clip(pcm_f32 * 32767.0, -32768, 32767).astype(np.int16)
    return pcm_int16_out


def _transcribe_local(pcm_float32):
    """Blocking faster-whisper call — runs in a thread."""
    segments, _info = _whisper.transcribe(
        pcm_float32,
        language="en",
        beam_size=1,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    return " ".join(seg.text for seg in segments)


# Short-audio hallucination mitigation: Whisper-family models (including
# gpt-4o-transcribe) are known to hallucinate content — sometimes outputting
# the wrong language entirely — when handed very short clips. Classic
# symptom: a 500ms "sit" comes back as "是" (Chinese yes) or "S".
# Fix: pad short clips to at least MIN_CLIP_MS with silence on both sides.
# Preserves content, gives the model enough time-context to not invent.
MIN_CLIP_MS = 1500

# Note: we used to pass a context-prompt hint to the transcription API,
# but that introduced a worse failure mode than it fixed — for too-short
# or too-quiet audio, gpt-4o-mini-transcribe would ECHO THE PROMPT back
# verbatim as the transcription, resulting in phantom commands like
# "LaiRA is a robot dog. The user gives short English commands..." being
# forwarded into T1. This is a documented Whisper-family behavior.
#
# Going prompt-free. Padding + language=en does the anti-hallucination
# work without the echo risk.


def _pcm_to_wav_bytes(pcm_int16, pad_to_ms=None):
    """Wrap raw int16 mono PCM in a minimal RIFF/WAV container for upload
    to OpenAI's /v1/audio/transcriptions endpoint. Optionally pads with
    silence on both sides to reach pad_to_ms, which fixes short-audio
    hallucination."""
    if pad_to_ms is not None:
        current_ms = 1000 * pcm_int16.size // SAMPLE_RATE
        if current_ms < pad_to_ms:
            need_samples = SAMPLE_RATE * (pad_to_ms - current_ms) // 1000
            # Symmetric padding so the utterance sits in the middle.
            pre = need_samples // 2
            post = need_samples - pre
            pcm_int16 = np.concatenate([
                np.zeros(pre, dtype=np.int16),
                pcm_int16,
                np.zeros(post, dtype=np.int16),
            ])
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def _dump_wav_debug(pcm_int16, dest="/tmp/laira_last_command.wav"):
    """Save the un-padded raw command audio to /tmp for debug. Listen to
    it after a bad transcription to confirm whether the audio itself is
    good (mic issue) or whether the model is hallucinating."""
    try:
        with wave.open(dest, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_int16.tobytes())
    except Exception as e:
        print(f"[stt] debug wav dump failed: {e}")


async def _transcribe_openai(pcm_int16):
    """POST the recorded command audio to OpenAI's transcription
    endpoint and return the transcribed text. Uses aiohttp for async I/O
    so the event loop stays unblocked during the network round-trip.

    Applies short-audio padding and a command-vocabulary prompt hint to
    reduce hallucination on commands like 'sit' / 'stand' that are too
    short for Whisper to reliably ground."""
    import aiohttp
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("T1_PROMPT_KEY_OPENAI")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set — can't use cloud STT backend")
    # Snapshot the raw audio to disk for post-mortem debugging, then send
    # the padded version to OpenAI.
    _dump_wav_debug(pcm_int16)
    wav_bytes = _pcm_to_wav_bytes(pcm_int16, pad_to_ms=MIN_CLIP_MS)
    timeout = aiohttp.ClientTimeout(total=20)
    form = aiohttp.FormData()
    form.add_field("model", OPENAI_TRANSCRIBE_MODEL)
    form.add_field("language", "en")
    # No prompt hint — see OPENAI_PROMPT_HINT comment at top of module.
    # response_format=text returns a plain-text body (no JSON wrapping) —
    # cheapest to parse. gpt-4o-mini-transcribe supports this.
    form.add_field("response_format", "text")
    form.add_field("file", wav_bytes,
                   filename="cmd.wav",
                   content_type="audio/wav")
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.post(
            OPENAI_TRANSCRIBE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            data=form,
        ) as r:
            raw = await r.text()
            if r.status != 200:
                raise RuntimeError(f"openai transcribe {r.status}: {raw[:300]}")
            return raw.strip()


class ParakeetClient:
    """Persistent WebSocket client to the Modal Parakeet worker. ONE
    connection is held open and reused across recordings; each recording
    is just an open/chunks/close exchange on the same socket. This
    eliminates the per-wake handshake latency that was bottlenecking
    interrupt-path recordings (the ~3.5s outlier we traced to
    asyncio-loop-contended TLS handshakes during interrupt cleanup).

    Architecture:
      - One background task (_connect_loop) owns the WS lifecycle and
        keeps it open. On any disconnect (server close, network blip,
        cellular IP change), it tears down and reconnects with backoff.
      - One reader task (_reader_loop) drains incoming TEXT frames and
        dispatches `final` / `error` messages to whichever recording is
        currently in progress (single-recording-at-a-time invariant —
        the Pi only records one command at once).
      - Per-recording API: begin_recording(gain) → push_chunk(pcm) ...
        → end_recording_and_get_text() → returns transcript string.
      - If WS is down when begin_recording is called, raises immediately
        so caller can fall back to OpenAI batch.
      - If WS drops mid-recording, the in-flight Future fails so caller's
        end_recording_and_get_text() raises and falls back.

    Cellular switching specifically: when LaiRA's network swaps cellular
    cell or moves between WiFi and cellular, the existing TCP connection
    breaks. aiohttp's WS heartbeat (heartbeat=20s) detects the broken
    pong, raises, and the connect loop reconnects on the new IP. By
    the next wake fire, the WS is back up and the recording proceeds
    normally."""

    def __init__(self, ws_url):
        self._ws_url = ws_url
        self._aiohttp_session = None
        self._ws = None
        self._connected = asyncio.Event()
        self._connect_task = None
        # Per-recording state — only one recording is ever active at a
        # time (mic captures sequentially). Reset on each begin/end.
        self._current_sid = None
        self._final_future = None
        self._gain = 1.0

    def start(self):
        """Idempotent: starts the connect loop if not already running."""
        if self._connect_task is None or self._connect_task.done():
            self._connect_task = asyncio.create_task(self._connect_loop())

    @property
    def connected(self):
        return self._connected.is_set()

    async def _connect_loop(self):
        """Maintains the WS connection. Reconnects on drop with
        exponential backoff (capped at 30s). Runs forever."""
        import aiohttp
        backoff = 1.0
        while True:
            try:
                self._aiohttp_session = aiohttp.ClientSession()
                # heartbeat=20: aiohttp sends ping every 20s, expects pong.
                # If pong doesn't come (network broken, cellular switch),
                # the WS raises and we land in the except branch below.
                self._ws = await self._aiohttp_session.ws_connect(
                    self._ws_url, heartbeat=20.0,
                )
                print(f"[parakeet] WS connected to {self._ws_url}")
                self._connected.set()
                backoff = 1.0
                # Drain incoming messages until ws closes.
                async for msg in self._ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            data = msg.json()
                        except Exception:
                            continue
                        typ = data.get("type")
                        if typ == "final":
                            fut = self._final_future
                            if fut is not None and not fut.done():
                                fut.set_result(data.get("text", ""))
                        elif typ == "error":
                            fut = self._final_future
                            if fut is not None and not fut.done():
                                fut.set_exception(RuntimeError(
                                    f"parakeet server error: {data.get('error')}"
                                ))
                        # open_ok and others: ignored. We don't await
                        # open_ok because TCP-level message order is
                        # preserved — chunks sent right after "open"
                        # arrive at the server in order, after the
                        # server has processed the open.
                    elif msg.type in (aiohttp.WSMsgType.CLOSE,
                                      aiohttp.WSMsgType.CLOSED,
                                      aiohttp.WSMsgType.ERROR):
                        break
                print("[parakeet] WS disconnected, will reconnect")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[parakeet] WS connect/run failed: {e!r}")
            finally:
                self._connected.clear()
                # If a recording is in flight, fail it so caller can
                # immediately fall back to OpenAI batch instead of
                # waiting for the timeout.
                fut = self._final_future
                if fut is not None and not fut.done():
                    fut.set_exception(RuntimeError("parakeet WS dropped mid-recording"))
                self._current_sid = None
                self._final_future = None
                if self._ws is not None:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass
                    self._ws = None
                if self._aiohttp_session is not None:
                    try:
                        await self._aiohttp_session.close()
                    except Exception:
                        pass
                    self._aiohttp_session = None
            # Backoff before reconnect attempt. Cap at 30s so a network
            # outage doesn't make us wait forever; also short enough
            # that the user's next wake within a minute usually finds
            # us reconnected.
            await asyncio.sleep(min(backoff, 30.0))
            backoff = min(backoff * 2.0, 30.0)

    async def begin_recording(self, gain=1.0):
        """Open a new recording session on the persistent WS. Sends the
        "open" json. Caller can immediately push_chunk after this returns.
        Raises if WS is not currently connected — caller should fall back
        to OpenAI batch."""
        if not self._connected.is_set() or self._ws is None:
            raise RuntimeError("parakeet WS not connected")
        import uuid
        sid = str(uuid.uuid4())
        self._current_sid = sid
        self._final_future = asyncio.get_event_loop().create_future()
        self._gain = float(gain)
        try:
            await self._ws.send_json({
                "type": "open",
                "session_id": sid,
                "sample_rate": SAMPLE_RATE,
            })
        except Exception as e:
            self._current_sid = None
            self._final_future = None
            raise RuntimeError(f"parakeet open send failed: {e!r}")
        return sid

    def push_chunk(self, pcm_int16):
        """Non-blocking. Schedules a chunk send. Silently drops if WS
        isn't connected — chunks lost in that window will just be
        missing from the recording, recovery is the caller's problem."""
        if not self._connected.is_set() or self._ws is None:
            return
        if self._current_sid is None:
            # No active recording session; drop.
            return
        gained = self._gained(pcm_int16)
        try:
            asyncio.get_event_loop().create_task(self._send_bytes(gained.tobytes()))
        except Exception:
            pass

    async def _send_bytes(self, data):
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.send_bytes(data)
        except Exception as e:
            # Log but don't propagate — best-effort delivery, the
            # server will work with whatever chunks made it through.
            print(f"[parakeet] send_bytes failed: {e!r}")

    async def end_recording_and_get_text(self, timeout=10.0):
        """Send "close" json and await the final transcript. Returns
        the transcript string. Raises on disconnect or timeout, in
        which case caller should fall back to OpenAI batch."""
        if self._current_sid is None or self._final_future is None:
            return ""
        sid = self._current_sid
        fut = self._final_future
        # Reset state BEFORE sending close so a parallel push_chunk
        # (shouldn't happen but defense) doesn't conflict.
        self._current_sid = None
        ws = self._ws
        if ws is None or not self._connected.is_set():
            self._final_future = None
            raise RuntimeError("parakeet WS lost before close could be sent")
        try:
            await ws.send_json({"type": "close", "session_id": sid})
        except Exception as e:
            self._final_future = None
            raise RuntimeError(f"parakeet close send failed: {e!r}")
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._final_future = None

    def _gained(self, pcm_int16):
        if self._gain == 1.0:
            return pcm_int16
        scaled = pcm_int16.astype(np.int32) * self._gain
        return np.clip(scaled, -32768, 32767).astype(np.int16)


# Module-level singleton client. Initialized lazily by ensure_parakeet_client().
# None until laira_stt is loaded with LAIRA_STT_BACKEND=modal_parakeet AND
# LAIRA_PARAKEET_WS_URL set.
_parakeet_client = None


def ensure_parakeet_client():
    """Idempotent: returns the singleton ParakeetClient, creating and
    starting it on first call. Returns None if Parakeet isn't configured."""
    global _parakeet_client
    if STT_BACKEND != "modal_parakeet" or not PARAKEET_WS_URL:
        return None
    if _parakeet_client is None:
        _parakeet_client = ParakeetClient(PARAKEET_WS_URL)
        _parakeet_client.start()
    return _parakeet_client
