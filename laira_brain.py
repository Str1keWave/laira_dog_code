"""
laira_brain.py — Pi-resident orchestrator (T1 routing, T2 vision, T3 motion).

Runs alongside laira_go_wsprotected.py in the same asyncio event loop.
Browsers on /smart_laira_test send text commands wrapped in
{type:'chat-message', message:'{"user_text":"..."}'}; the main script's
handle_message forwards them to brain.submit_text(). The brain calls
Anthropic directly, executes tools (via main-script motor primitives in
Phase 3d+), and publishes logs/state/reasoning back to the browser via
the outbound /laiRA WebSocket.

Current scope: Phase 3c.
  - Ported: tool schemas, system prompts, two-tier routing (Sonnet→Opus),
            session store with rolling 10-frame history + sticky escalation,
            Anthropic call, tool-call broadcast.
  - NOT yet: actual tool execution (poses, drive, wait, follow, go_to).
            All tool calls currently log as "would execute" and no motion
            happens. Phase 3d wires execution.
"""

import asyncio
import json
import os
import time

import aiohttp
import websockets

import laira_frames
import laira_tier1  # local keyword classifier: fast path for sit/stand/lie_down/shake/stop


# ═══════════════════════════════════════════════════════════════════════
# 1. Outbound broadcast channel (Pi → browser)
# ═══════════════════════════════════════════════════════════════════════

_outbound_ws = None


def set_outbound_ws(ws):
    """Called by laira_go_wsprotected when the /laiRA ws opens/closes."""
    global _outbound_ws
    _outbound_ws = ws


async def _send(payload):
    """Fire-and-forget. A failure here must not take down the brain, but
    it ALSO must not vanish silently — we lost a whole test to a quietly-
    closed ws where every broadcast was dropped and the journal gave no
    clue. Log failures to stdout so journalctl still reveals them."""
    ws = _outbound_ws
    if ws is None:
        try:
            print(f"[brain] _send: NO WS (dropped {payload.get('type')})")
        except Exception:
            pass
        return
    try:
        await ws.send(json.dumps(payload))
    except Exception as e:
        try:
            print(f"[brain] _send FAILED ({payload.get('type')}): {e}")
        except Exception:
            pass


async def broadcast_log(level, text):
    await _send({"type": "laira-log", "level": level, "text": text})


async def broadcast_plan_state(state, action=""):
    await _send({"type": "laira-plan-state", "state": state, "action": action})


async def broadcast_reasoning(source, text):
    await _send({"type": "laira-reasoning", "source": source, "text": text})


async def broadcast_tool_call(tool, args):
    await _send({"type": "laira-tool-call", "tool": tool, "args": args})


async def broadcast_tracker_stats(mode, target, depth_cm, action, status):
    await _send({
        "type": "laira-tracker-stats",
        "mode": mode, "target": target, "depthCm": depth_cm,
        "action": action, "status": status,
    })


async def broadcast_bbox(bbox):
    await _send({"type": "laira-bbox", "bbox": bbox})


async def broadcast_modal_status(status, detail=""):
    await _send({"type": "laira-modal-status", "status": status, "detail": detail})


async def broadcast_hello():
    await _send({"type": "laira-hello", "version": "brain-v1"})


async def broadcast_stt(phrase, forwarded_to_brain):
    """Emit a heard-voice-command to the browser console. `phrase` is the
    post-wake-word-stripped command (what the user meant for LaiRA to do).
    `forwarded_to_brain` reflects whether the brain also processed this
    phrase as a user_text (the stt_to_brain toggle)."""
    await _send({
        "type": "laira-stt",
        "phrase": phrase,
        "forwarded_to_brain": bool(forwarded_to_brain),
    })


# Tracks last known STT status so we can rebroadcast to a late-joining
# browser via on_hello_request (same pattern as _modal_status).
_stt_status = "cold"
_stt_status_detail = ""


async def broadcast_stt_status(status, detail=""):
    """Tell the browser whether STT is cold / warming / ready / error."""
    global _stt_status, _stt_status_detail
    _stt_status = status
    _stt_status_detail = detail
    await _send({"type": "laira-stt-status", "status": status, "detail": detail})


def stt_status():
    return _stt_status


# ═══════════════════════════════════════════════════════════════════════
# 2. Bbox streaming toggle (controlled by browser checkbox)
# ═══════════════════════════════════════════════════════════════════════

_bbox_stream = False


def set_bbox_stream(enabled):
    global _bbox_stream
    _bbox_stream = bool(enabled)
    print(f"[brain] bbox_stream -> {_bbox_stream}")


def bbox_stream_enabled():
    return _bbox_stream


# ── Voice-to-brain toggle (controlled by browser) ───────────────────────
# When True (default), wake-word-stripped STT phrases are submitted to the
# brain's T1 pipeline (i.e., LaiRA acts on spoken commands). The browser
# toggle in /smart_laira_test can flip this OFF — useful for dialing in
# wake-word recognition without LaiRA acting on every false-fire while
# you tune. The browser re-syncs its current value on every laira-hello,
# so a fresh service restart inherits whatever the UI is showing.
_stt_to_brain = True


def set_stt_to_brain(enabled):
    global _stt_to_brain
    _stt_to_brain = bool(enabled)
    print(f"[brain] stt_to_brain -> {_stt_to_brain}")


def stt_to_brain_enabled():
    return _stt_to_brain


# ── Wake-word safety-net stripper ──────────────────────────────────────
# The ACTUAL wake-word detection now happens upstream in laira_stt.py via
# a dedicated openWakeWord neural model — this list is only a safety net
# for extract_wake_word_command() in case whisper happens to transcribe
# the wake word as part of the command (e.g., the user ran wake-word +
# command together and Whisper captured both).
#
# Entries cover:
#   - "hey_jarvis" placeholder wake word variants (initial deployment)
#   - "laira" variants for when the custom-trained laira wake model lands
#   - Legacy Vosk-era mishears preserved as a fallback
WAKE_ALIASES = (
    # Current openwakeword default:
    "hey jarvis", "hey jervis", "hey jarv", "heyjarvis",
    # Target (custom model to come):
    "laira", "layra", "lyra", "lira",
    # Legacy Vosk mishears — harmless to keep; stripping is post-hoc:
    "liar", "lie r", "hi rest", "virus", "byron",
    "why run", "why around", "hi rez", "why i", "wheras", "iris",
    "why rob", "hi there", "higher", "i love",
)


def extract_wake_word_command(text):
    """Returns the command portion of a transcribed phrase — everything
    AFTER the earliest wake-word match — or None if no wake word was
    heard.

    Rules (per user spec):
      - The EARLIEST wake-word occurrence triggers extraction; leading
        filler before the wake word is dropped.
      - Only the FIRST wake-word instance is stripped. If the user says
        "laira" a second time within the same phrase it stays put (the
        command might legitimately reference her by name).
      - Empty command ("just heard laira alone") returns None — not
        worth invoking anything on an empty instruction.
      - Case-insensitive match; returned command preserves original case.
    """
    if not text:
        return None
    lower = text.lower()
    best_start = None
    best_end = None
    for w in WAKE_ALIASES:
        idx = lower.find(w)
        if idx < 0:
            continue
        if best_start is None or idx < best_start:
            best_start, best_end = idx, idx + len(w)
    if best_end is None:
        return None
    command = text[best_end:].lstrip(" ,:.!?-\t").rstrip()
    return command or None


async def handle_stt_dropped(reason):
    """Called by laira_stt when a recording was captured but discarded
    before transcription would be useful — either no speech detected
    in the window (Whisper would hallucinate from silence) or the
    transcription contained non-Latin characters (hallucination).

    Posting this to the browser console is essential UX: without it,
    a user who said the wake word and then a command sees the paw
    twitch (acknowledging wake fired) and then absolutely nothing
    happens — looks like LaiRA stopped responding. Now they at least
    get told 'I heard you but didn't catch the command'."""
    await broadcast_log("stt", f"⚠ wake heard, no usable command: {reason}")


async def handle_wake_detected():
    """Called by laira_stt the instant the wake word fires, BEFORE the
    command is recorded or transcribed. Two responsibilities:

      1. INTERRUPT any active plan immediately so the user feels feedback
         within ~80ms of saying 'lyra' instead of waiting 1-3s for STT
         to finish. Without this, mid-plan wake fires only take effect
         AFTER the new command lands via submit_text → interrupt_active,
         which makes LaiRA feel unresponsive ("she keeps walking past me
         even after I called her").

      2. PAW TWITCH for auditory confirmation. The Pi plays no TTS and
         has no LEDs the user can see across a room — but a quick
         servo-driven motion of the front-leg hip produces a clearly
         audible servo whine + tiny visible jiggle. User immediately
         knows 'she heard me' even if STT hasn't transcribed yet.
    """
    # 1. Halt motion FIRST — submit_text will also call interrupt_active
    #    once the transcription lands, but we want the halt to register
    #    immediately on wake so LaiRA stops walking past you.
    try:
        await interrupt_active()
    except Exception as e:
        print(f"[brain] wake interrupt failed: {e!r}")
    # 2. Paw twitch in the background — non-blocking so the STT pipeline
    #    can immediately move on to recording. The motor commands DO
    #    share a USB bus with the webcam mic, so each twitch costs a
    #    measurable audio gap. But the bigger USB offender (the
    #    unconditional brain_set_keystates({}) above) is now gated on
    #    actual plan-interrupts, so a clean wake-with-nothing-running
    #    is now down to just the twitch's three short serial calls.
    asyncio.create_task(_paw_twitch_safe())


async def _paw_twitch_safe():
    """Wraps the main module's paw_twitch in an exception handler so a
    motor-bus hiccup never crashes the wake-handler task."""
    try:
        main = _main()
        twitch = getattr(main, "brain_paw_twitch", None)
        if twitch is None:
            return
        await twitch()
    except Exception as e:
        print(f"[brain] paw_twitch failed: {e!r}")


async def handle_stt_phrase(text, doa=None):
    """Called by laira_go_wsprotected's stt_action_router with a fully
    transcribed command + optional direction-of-arrival hint. Upstream
    has already done wake-word detection + command recording + ILD-
    based DOA from the stereo mic. This function applies safety-net
    filters and then broadcasts + (optionally) submits to T1.

    `doa` is "left", "right", "center", or None. "left"/"right" mean
    the user's voice was meaningfully louder on that side of LaiRA's
    mics; "center" means front or behind (ambiguous along that axis,
    use vision to disambiguate); None means no stereo data was captured.

    Safety nets:
      1. Wake-word re-strip in case Whisper picked up both wake + command.
      2. Hallucinated-prompt-echo filter — Whisper-family models
         occasionally echo the API prompt back as the transcription.
         Drop any transcription that contains tell-tale prompt-echo
         phrases so we don't forward garbage into T1."""
    text = (text or "").strip()
    if not text:
        return
    # (2) Prompt-echo filter. These phrases don't belong in any real voice
    # command; they only appear when the transcription model has
    # hallucinated or leaked a context prompt.
    lower = text.lower()
    PROMPT_ECHO_TELLS = (
        "laira is a robot dog",
        "the user gives",
        "voice commands",
        "common commands:",
        "following are voice",
    )
    if any(tell in lower for tell in PROMPT_ECHO_TELLS):
        print(f"[brain] STT dropped (prompt echo detected): {text!r}")
        return
    # (1) Best-effort wake-word strip — only kicks in if the wake word is
    # present in the command text.
    stripped = extract_wake_word_command(text)
    command = stripped if stripped is not None else text
    forward = stt_to_brain_enabled()
    print(f"[brain] STT command: {command!r} (to_brain={forward}, doa={doa})")
    await broadcast_stt(command, forward)
    if forward:
        await submit_text(command, doa=doa)


# ═══════════════════════════════════════════════════════════════════════
# 3. Modal readiness tracking
# ═══════════════════════════════════════════════════════════════════════

_modal_status = "cold"  # cold | warming | ready | error


async def _modal_probe_once():
    """Single GET /healthz. Returns (success_bool, detail_str). detail is
    elapsed ms on success, error description on failure. We READ THE BODY
    on failure because Modal piggybacks billing/quota errors on HTTP 429
    (e.g. 'workspace billing cycle spend limit reached') and we want that
    surfaced to the user, not buried as a generic rate-limit message."""
    url = os.environ.get("MODAL_TRACKER_URL", "").rstrip("/")
    if not url:
        return False, "MODAL_TRACKER_URL not set"
    try:
        timeout = aiohttp.ClientTimeout(total=90)
        t0 = time.time()
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(url + "/healthz") as r:
                elapsed_ms = int((time.time() - t0) * 1000)
                if r.status == 200:
                    return True, f"{elapsed_ms}ms"
                # Read body to extract the real reason. Modal's 429s carry
                # text like "modal-http: Webhook failed: workspace billing
                # cycle spend limit reached" which is NOT a transient
                # rate-limit and won't fix itself with backoff.
                body = ""
                try:
                    body = (await r.text())[:200].strip()
                except Exception:
                    pass
                if body:
                    return False, f"http {r.status}: {body}"
                return False, f"healthz http {r.status}"
    except asyncio.TimeoutError:
        return False, "healthz timeout (90s)"
    except Exception as e:
        return False, str(e)


def _modal_error_is_billing(detail):
    """Detect the Modal billing-cap failure mode specifically. Returns
    True if the error body mentions billing/spend/quota — these need
    user action (raise cap or add credit), no amount of retrying helps."""
    if not detail:
        return False
    d = detail.lower()
    return ("billing" in d or "spend limit" in d or "quota" in d
            or "spend cap" in d)


def _modal_error_is_transient(detail):
    """Worth retrying: generic 429 (real rate limit), 5xx, timeouts.
    NOT worth retrying: billing cap, 4xx config errors. Billing must be
    detected explicitly because Modal returns 429 for it."""
    if not detail:
        return True
    if _modal_error_is_billing(detail):
        return False  # billing won't auto-fix
    d = detail.lower()
    if "429" in d or "http 429" in d or "timeout" in d or "connection" in d:
        return True
    if "http 5" in d:
        return True
    return False


async def warmup_modal():
    """Initial Modal readiness probe with backoff. If all retries fail,
    schedules a background re-probe loop so the system recovers
    automatically when Modal comes back online — no Pi restart needed."""
    global _modal_status
    if not os.environ.get("MODAL_TRACKER_URL", "").rstrip("/"):
        _modal_status = "error"
        msg = "MODAL_TRACKER_URL not set in ~/laira_code/.env"
        print(f"[brain] {msg}")
        await broadcast_modal_status("error", msg)
        await broadcast_log("err", msg)
        return
    _modal_status = "warming"
    print("[brain] warming Modal")
    await broadcast_modal_status("warming")
    await broadcast_log("evt", "warming up Modal (SAMURAI tracker)…")

    # Backoff retries on transient errors. Most commonly: 429 right after
    # a burst of restarts (Modal rate-limits per-account/IP).
    last_detail = ""
    for attempt, delay in enumerate([0.0, 3.0, 8.0, 20.0]):
        if delay > 0:
            print(f"[brain] Modal warmup retry {attempt} (after {delay}s)")
            await asyncio.sleep(delay)
        ok, detail = await _modal_probe_once()
        last_detail = detail
        if ok:
            _modal_status = "ready"
            print(f"[brain] Modal ready ({detail})")
            await broadcast_modal_status("ready", detail)
            await broadcast_log("ok", f"Modal ready ({detail})")
            return
        print(f"[brain] Modal warmup attempt {attempt} failed: {detail}")
        if not _modal_error_is_transient(detail):
            break  # 4xx-style failures aren't going to fix themselves

    # All initial retries exhausted. If it's a billing cap, broadcast a
    # specific actionable message and slow the reprobe WAY down (10
    # minutes) — billing won't fix itself and we shouldn't burn cycles.
    # If it's something transient that just hasn't recovered yet, normal
    # 30s reprobe.
    _modal_status = "error"
    is_billing = _modal_error_is_billing(last_detail)
    if is_billing:
        msg = ("Modal workspace spend cap reached — raise the cap or add "
               "credit in the Modal dashboard. Nothing the brain does will "
               "fix this. Reprobing every 10 minutes in case you fix it.")
        print(f"[brain] {msg}")
        await broadcast_modal_status("error", "billing cap reached — raise cap in Modal dashboard")
        await broadcast_log("err", msg)
        asyncio.create_task(_modal_reprobe_loop(interval=600))
    else:
        print(f"[brain] Modal warmup gave up: {last_detail}; starting reprobe loop")
        await broadcast_modal_status("error", last_detail)
        await broadcast_log("err", f"Modal unavailable: {last_detail} — retrying every 30s")
        asyncio.create_task(_modal_reprobe_loop(interval=30))


async def _modal_reprobe_loop(interval=30):
    """While modal_status != 'ready', probe healthz every `interval`s and
    flip to ready the moment Modal answers 200. Exits as soon as ready
    (a successful SAMURAI ws connect elsewhere can also flip status).
    interval is short (30s) for transient errors and long (600s) for
    billing-cap failures — those need user action, no point spamming."""
    global _modal_status
    print(f"[brain] modal reprobe loop started (interval={interval}s)")
    while _modal_status != "ready":
        await asyncio.sleep(interval)
        if _modal_status == "ready":
            break
        ok, detail = await _modal_probe_once()
        if ok:
            _modal_status = "ready"
            print(f"[brain] Modal recovered via reprobe ({detail})")
            await broadcast_modal_status("ready", f"recovered ({detail})")
            await broadcast_log("ok", f"Modal recovered ({detail})")
            return
        # Stay in error. If the failure mode flipped (e.g. billing was
        # fixed but now genuinely rate-limited), adjust the interval on
        # the fly.
        new_interval = 600 if _modal_error_is_billing(detail) else 30
        if new_interval != interval:
            print(f"[brain] modal reprobe interval {interval}s -> {new_interval}s ({detail[:80]})")
            interval = new_interval
        print(f"[brain] modal reprobe still failing: {detail[:120]}")
    print("[brain] modal reprobe loop exiting (status=ready)")


async def _mark_modal_ready_from_evidence(reason):
    """Called when something OTHER than the healthz probe proves Modal is
    alive (e.g. a SAMURAI ws successfully opened). Promotes status to
    ready and broadcasts so the textbox un-disables on the browser."""
    global _modal_status
    if _modal_status == "ready":
        return
    _modal_status = "ready"
    print(f"[brain] Modal marked ready ({reason})")
    await broadcast_modal_status("ready", reason)
    await broadcast_log("ok", f"Modal ready ({reason})")


def modal_status():
    return _modal_status


# ═══════════════════════════════════════════════════════════════════════
# 4. Tool schemas (verbatim port from server.js SHARED_TOOLS/SONNET_TOOLS/OPUS_TOOLS)
# ═══════════════════════════════════════════════════════════════════════

SHARED_TOOLS = [
    {
        "name": "sit",
        "description": "Make LaiRA sit on her haunches.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "stand",
        "description": "Make LaiRA stand up from any pose. Also use this when the user wants her to stop sitting / lying / shaking.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "lie_down",
        "description": "Make LaiRA lie down on her belly. Use for: lie down, sleep, rest, take a nap.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "shake",
        "description": "Make LaiRA offer her paw for a handshake.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "follow",
        "description": "Continuously track and follow a target object until told to stop. Holds the requested following distance, does NOT terminate on its own. Use for: follow X, chase X, stay on X. For \"follow me\" the target is the visible person.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Short natural-language description of what to follow (e.g. \"the person in the red shirt\", \"me\", \"the green ball\")."},
                "target_distance_cm": {
                    "type": "integer", "minimum": 5, "maximum": 300,
                    "description": "How far back LaiRA should hold while following, in centimetres. Reference points: 30-50 close-follow on a small toy or pet, 60-100 a polite person-following distance, 150+ \"stay in the area but give space\".",
                },
            },
            "required": ["target", "target_distance_cm"],
        },
    },
    {
        "name": "crawl",
        "description": "Same semantics as follow() — continuous tracking of a target, does NOT auto-terminate — but at a noticeably SLOW gait. Use when the user's intent is crawling / sneaking / creeping / stalking movement. Typical triggers: \"crawl to me\", \"sneak up on X\", \"creep over\", \"come slowly\". The speed parameter is hard-capped at 1-4; do NOT try to use crawl with a higher speed to approximate a normal follow — the gait isn't designed for that. If the user wants regular-paced locomotion use follow() instead.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Short natural-language description of what to crawl toward (same conventions as follow)."},
                "target_distance_cm": {
                    "type": "integer", "minimum": 5, "maximum": 300,
                    "description": "How close to hold while crawling, in centimetres. Same scale as follow.",
                },
                "speed": {
                    "type": "integer", "minimum": 1, "maximum": 4,
                    "description": "Crawl gait speed, 1-4. Default 2. 1 = barely-moving full-stealth; 2 = slow stalking (default); 3 = cautious approach; 4 = the top of the crawl range, still clearly slow. The 4 cap is a hard limit; crawl cannot be used as a fast-follow shortcut.",
                },
            },
            "required": ["target", "target_distance_cm"],
        },
    },
    {
        "name": "go_to",
        "description": "Walk to a specific target and STOP once reached. Differs from follow: terminates when LaiRA has arrived. Use for: go to X, come to X, walk over to X.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Short natural-language description of what to walk to."},
                "target_distance_cm": {
                    "type": "integer", "minimum": 5, "maximum": 300,
                    "description": "How close LaiRA should be to the target when \"arrived\", in centimetres. Reference points: 10-20 nose-to-target (a ball she'd sniff), 30-60 body-distance (a person, another dog, a piece of furniture she greets), 80-150 room-scale (\"just be in the area\").",
                },
            },
            "required": ["target", "target_distance_cm"],
        },
    },
    {
        "name": "stop",
        "description": "Cancel any active task, stand still, and END THE CURRENT SESSION. After stop(), no further reasoning happens — the next user message starts a fresh conversation with no memory of what came before. Use ONLY when: (a) the user explicitly cancels (\"stop\", \"halt\", \"freeze\", \"never mind\", \"cancel that\"), or (b) the user's task is fully complete AND they want LaiRA to disengage entirely with no further response (e.g. \"go to your spot and chill\"). DO NOT call stop() if you intend to keep watching for a condition or react to a future event — that's wait()'s job. stop() is the kill switch, not a pause.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "wait",
        "description": "Pause for duration_ms milliseconds, then the session continues — you'll receive a fresh camera frame and decide the next step. Hard cap per call is 30000ms, BUT you can chain successive wait() calls indefinitely; session memory persists across them. Use cases: (1) Pacing within a plan (\"sit, then stand 2s later\" -> sit, wait(2000), stand). (2) Watch-and-react (\"shake when I put my hand out\") — sit, wait(1500), then on the next round inspect the frame for the trigger; if present, do the conditional action; if not, wait(1500) again. (3) Wait for a scene change (\"approach when the cat leaves\") — same polling pattern. Wait is interrupted by user input or stop().",
        "input_schema": {
            "type": "object",
            "properties": {
                "duration_ms": {
                    "type": "integer", "minimum": 100, "maximum": 30000,
                    "description": "Milliseconds to pause before getting a fresh frame and deciding again. Reference: 1500-3000 for an active polling loop (watch-and-react), 5000-10000 for a relaxed monitor, 10000+ for low-frequency check-ins.",
                },
            },
            "required": ["duration_ms"],
        },
    },
]

OPUS_TOOLS = SHARED_TOOLS + [
    {
        "name": "drive",
        "description": "Send a single low-level movement command. Used for non-tracking-based movement like \"back up a bit\", \"turn around\", \"spin in place\". Direction: forward / back (translation), or rotate_left / rotate_right (turn in place). LaiRA is a quadruped — she does not strafe; sideways movement is achieved by rotating then driving forward. \n\nDURATION: forward / back are hard-capped at 3000ms (longer blind translation is unreliable — use the tracker via go_to / follow). rotate_left / rotate_right may go up to 15000ms (full 360° at default speed = ~15s; 7.5s = 180°; 4s ≈ 90°; 2s ≈ 45°).\n\nIN-BETWEEN SHOTS (rotate only): the optional in_between_shots parameter (default 0) requests N intermediate camera snapshots taken at evenly-spaced moments DURING the rotation, in addition to the post-rotation frame you'd see normally. Use it to peek at intermediate angles instead of jumping straight to the end pose. Example: drive(rotate_right, 15000, in_between_shots=2) rotates a full 360° and captures snapshots at ~5s and ~10s in addition to the post-rotation frame. Each in-between snapshot will appear in your NEXT turn's input prefixed with a label like \"[in-between shot K/N from rotate_X at elapsed=Yms — NOT the current frame]\" so you don't confuse it with the post-rotation frame. in_between_shots is ONLY honored for rotate_left / rotate_right; for forward / back it must be 0 (and will be silently coerced to 0 if you set it).",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["forward", "back", "rotate_left", "rotate_right"]},
                "duration_ms": {"type": "integer", "minimum": 100, "maximum": 15000},
                "in_between_shots": {"type": "integer", "minimum": 0, "maximum": 10, "default": 0},
            },
            "required": ["direction", "duration_ms"],
        },
    },
]

VALID_OPUS_TOOL_NAMES = {t["name"] for t in OPUS_TOOLS}


# ═══════════════════════════════════════════════════════════════════════
# 5. System prompts (verbatim port)
# ═══════════════════════════════════════════════════════════════════════

NO_TALKBACK_CLAUSE = " ".join([
    "CRITICAL CONSTRAINT — READ TWICE:",
    "You are LaiRA, a quadruped robot dog. You communicate ONLY through tool calls. You do NOT speak to the user.",
    "Dogs do not talk. You are a dog.",
    "If you cannot perform the requested action, do nothing — do not apologize, do not explain to the user, do not ask for clarification.",
    "Never call a tool that \"tells the user\" something. There is no such tool. There never will be.",
    "You may include brief reasoning text BEFORE your tool calls. This reasoning is logged for human debugging only and is never shown to the user. Keep it under 30 words. Skip it if the action is obvious.",
])

OPUS_SYSTEM = "\n".join([
    NO_TALKBACK_CLAUSE,
    "",
    "You are the orchestrator for LaiRA. Plan the entire sequence of tool calls UPFRONT in one response. You will NOT get a chance to react to results — tools execute fire-and-forget, and any failure aborts the rest of the plan.",
    "Be conservative. If a step is risky or you're uncertain it will work, omit it. Better to do less correctly than more half-done.",
    "You can issue multiple tool calls in your single response. They execute in order. Each tool blocks until complete (e.g. go_to() blocks until LaiRA arrives or gives up). Plan accordingly: after a successful go_to(green ball, 15), you can chain a sit() because you know LaiRA will be standing right next to it. For pacing (\"sit, then stand a moment later\"), use wait(duration_ms) between the two actions.",
    "For follow / go_to / crawl you must pick target_distance_cm (5-300). Reference: 10-20 nose-to-target, 30-60 body-distance to a person/dog, 80-150 room-scale. Choose what fits the user's actual intent.",
    "For crawl: the speed arg defaults to 2. Use 1 for max-stealth (\"sneak in without her noticing\"), 2 for generic slow stalking, 3 for cautious approach, 4 for the top of the slow range. Never above 4 — that's a hard cap, not a suggestion.",
    "You have a drive() tool the first-line router does not. Use it for \"turn around\", \"back up\", \"explore\" type intents. Note: LaiRA cannot strafe — there is no left/right translation, only forward/back and rotate_left/rotate_right. To \"move sideways\" you rotate then drive forward.",
    "",
    "Camera frames in your message history are labeled \"[frame X of N — ...]\". The CURRENT frame (highest index, marked CURRENT) is the just-captured one. Earlier frames in the same session are kept so you remember spatial context from when the task started — for example, if the user pointed somewhere in the original frame, that gesture won't be in the current frame but you can refer back to it. Older frames beyond the rolling window are omitted to save tokens.",
    "",
    "STOP vs WAIT — these have very different semantics, get this right:",
    "  - stop() ENDS the current session entirely. No further reasoning, session memory dropped. The next user message starts a fresh conversation.",
    "  - wait(ms) PAUSES briefly within an active session, after which you get a fresh frame and decide again. Session memory persists across waits.",
    "Decide based on user intent after the action is done:",
    "  - PARK (user wants disengagement, no further response): \"go to your spot\", \"go wait by the door\", \"chill there\". After arriving / settling: stop(). Clean disengagement.",
    "  - CHAIN (user wants follow-up actions): \"come here, then sit\", \"go to the ball and lie down\". After the prior step: chain the next tool call directly.",
    "  - MONITOR (user wants you to react to a future condition): \"shake when I put my hand out\", \"approach when the cat leaves\", \"follow me until I stop\". Use the watch-and-react pattern below — NEVER stop() for these. stop() kills the watching task and your memory of what to look for.",
    "",
    "WATCH-AND-REACT — for any conditional intent (\"do X when Y happens\"):",
    "  1. Position yourself appropriately (go_to / sit / stand / wherever you need to observe from).",
    "  2. Call wait(1500-3000). The session continues; you'll get a fresh frame and re-enter planning.",
    "  3. Inspect the new frame for the trigger condition (hand extended, cat moved, person sat down, etc.).",
    "  4. If the condition is MET: emit the conditional action (shake, follow, approach, sit, etc.). Plan continues normally from there.",
    "  5. If the condition is NOT met: emit wait(1500-3000) again — you'll loop back to step 3.",
    "Each wait+continuation pair is one round-trip. Loops can run indefinitely; session memory carries the user's original instruction the whole time. Only break the loop with the conditional action, an alternate user-driven action, or stop() (only if user genuinely cancels). Picking the wait duration: 1500ms for time-sensitive triggers (gesture), 3000-5000ms for slow events (someone leaving the room), 10000+ for low-priority idle checks.",
    "",
    "ROTATION TIMING — LaiRA rotates SLOWLY. Real numbers at default speed: a full 360° takes ~15 seconds of actual rotation; 180° ≈ 7.5s; 90° ≈ 4s; 45° ≈ 2s. So a 500ms drive(rotate_*) turns her only ~12° — barely a glance, usually not enough to bring a new region into view. Use chunk sizes appropriate to intent:",
    "  - SEARCH rotations (target out of frame, scanning the room): you have two equally-good options. (a) Issue 5-7 chunks of 2000-3000ms each, evaluating after each. (b) Issue ONE long rotate (up to 15000ms = full 360°) with in_between_shots set to 4-6, getting all the search frames in a single drive call. Option (b) is faster end-to-end since you skip orchestrator round-trips between chunks. Use (a) when you want the option to abort early on spotting the target; use (b) for committed full-circle scans.",
    "  - FINE ADJUSTMENTS (target already spotted, want to center it before go_to): 200-500ms is fine — small nudge to align.",
    "  - QUICK GLANCES (\"is the door behind me?\"): one 1500-2500ms chunk to look that direction, then decide.",
    "",
    "SPATIAL LANGUAGE — interpret directional words from LaiRA's perspective by default:",
    "  - \"your left\" / \"on your left\" / \"to your left\" = LaiRA's left = the LEFT side of the camera frame (lower x pixel values).",
    "  - \"your right\" / \"on your right\" = LaiRA's right = the RIGHT side of the frame (higher x pixel values).",
    "  - \"in front of you\" = the center / forward of the frame.",
    "  - \"behind you\" = not in the current frame; rotate ~180° to bring it into view.",
    "  - \"my left\" / \"my right\" = the user's perspective. If the user is visible in the frame and you can infer their facing direction, translate from there (their left is usually opposite to yours when they face you). If you can't infer their facing, default to LaiRA's perspective and proceed.",
    "If the user says \"the X on your left\" and you do NOT see X on the left side of the current frame, do NOT reinterpret as their left — instead, rotate left to look (the target may simply be out of frame to LaiRA's left). Trust the user's directional cue; rotate first, recheck.",
    "",
    "TARGET SELECTION — LaiRA's camera is mounted low (roughly knee-height on a standing person). As she walks toward a tall target, the upper portion rises OUT of the top of the frame; once SAMURAI's pixel-region leaves the frame, the tracker emits 'lost'. To avoid premature loss when calling follow() / go_to(), prefer a target that lives near LaiRA's vertical plane and will stay in frame on approach:",
    "  - \"the gaming chair\" instead of \"the person in the gaming chair\" — the chair persists in frame; the person rises out as you close in.",
    "  - \"the green ball\" instead of \"the person holding the ball\".",
    "  - \"the bottom of the bookshelf\" instead of \"the books on the top shelf\".",
    "  - \"the dog bed\" instead of \"the dog standing on the couch\" if there's a couch in the way.",
    "If the target itself is low (toy, pet, low furniture), name it directly. If the user's intent is to reach a person, find a SURFACE adjacent to them at ground level (chair, couch, doorway base) and target that instead — you'll arrive at the same place. Only fall back to naming a tall/elevated target when there is no adjacent ground-plane landmark.",
    "",
    "SEARCH BEHAVIOR — when the user asks you to find / go to / follow something that is NOT visible in the current frame, rotate in place. Two equivalent strategies (pick one per intent):",
    "  Strategy A (chunked): use 2000-3000ms chunks. After each, you see a new frame and can switch to follow() / go_to() the moment the target appears. Best when you want early-exit on spotting the target.",
    "  Strategy B (single long rotate with in_between_shots): drive(rotate_right, 15000, in_between_shots=4-6) does a full 360° and returns ALL the intermediate frames in one shot. Faster total wall clock, but you commit to the whole rotation before deciding. Use when you want a complete scan no matter what.",
    "DO NOT give up after one or two rotation chunks just because the target still isn't visible — it may be directly behind you, which takes ~7.5s (half the 15s full rotation) to bring into view. Keep rotating until ONE of:",
    "  (a) you spot the target — immediately switch to follow() or go_to() (with a TARGET SELECTION-aware target string) without finishing the scan;",
    "  (b) you have completed a full 360°, meaning the current frame approximately matches the \"oldest in window\" frame from your history (same walls, same furniture, same general composition / angle). At that point the target is not reachable from this spot; call stop().",
    "How to recognize a full 360° from the frame history: compare the CURRENT frame to the OLDEST frame in the window (the one labeled \"from when this part of the conversation started\"). If they look clearly different — different walls, different scene — you have NOT completed the rotation yet, keep going. If they look roughly the same perspective, you have come back around. Judge generously; frames won't match pixel-perfectly and small parallax is normal. Only stop() after a genuine full scan (~5-7 chunks of 2000-3000ms each, or one 15000ms with in_between_shots).",
    "",
    "DIRECTION-OF-ARRIVAL HINT — voice commands may be prefixed with [voice came from your left] / [voice came from your right] / [voice came from your center]. This comes from LaiRA's stereo mics measuring which side of her the user's voice was loudest on. Use it as a strong hint when the user's location matters but they're not visible:",
    "  - \"left\" / \"right\": the user's voice was meaningfully louder on that side. Trust it. If you need to find them (e.g. \"come over to me\" and the user isn't in frame), rotate THAT direction first instead of guessing.",
    "  - \"center\": means the voice was roughly equal volume on both mics. This is AMBIGUOUS along the front-back axis — the user is either directly in front of LaiRA or directly behind her. Use the camera frame to disambiguate: if a likely user is visible in the current frame, they're in front; if not, they're behind. Don't guess randomly.",
    "  - No hint at all: stereo data unavailable, treat the command as if no spatial info was given.",
    "Always strip the hint from your interpretation of the user's words — they didn't actually say \"voice came from your left\", that's a mic-derived annotation. The hint is information-only; it doesn't change what to do, just informs HOW (which way to rotate, etc.). Don't acknowledge the hint in any response.",
])


# ═══════════════════════════════════════════════════════════════════════
# 6. Session store (rolling frame history + sticky escalation)
# ═══════════════════════════════════════════════════════════════════════

MAX_FRAMES_IN_HISTORY = 10

# Single active session. server.js has a Map because one server serves many
# LaiRAs; here the Pi *is* one LaiRA. We only need one conversation state.
_session = None


def _new_session():
    return {"messages": [], "last_touch": time.time()}


def _clear_session():
    global _session
    _session = None


def _strip_frames(content):
    """Replace image blocks with text markers."""
    out = []
    for b in content:
        if b.get("type") == "image":
            out.append({"type": "text", "text": "[prior camera frame omitted]"})
        else:
            out.append(b)
    return out


def _build_call_messages(prior, current_user_content):
    """Port of server.js buildCallMessages. Walks user turns; in the last
    MAX_FRAMES_IN_HISTORY image-bearing turns the images are kept and each
    gets a position label ([frame X of N — ...]); older turns have images
    stripped to text markers."""
    all_msgs = list(prior) + [{"role": "user", "content": current_user_content}]
    image_turn_idx = []
    for i, m in enumerate(all_msgs):
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            if any(b.get("type") == "image" for b in m["content"]):
                image_turn_idx.append(i)
    keep_count = min(len(image_turn_idx), MAX_FRAMES_IN_HISTORY)
    keep_start = len(image_turn_idx) - keep_count
    kept_indices_ordered = image_turn_idx[keep_start:]
    kept_indices = set(kept_indices_ordered)

    out = []
    for i, m in enumerate(all_msgs):
        if m.get("role") != "user" or not isinstance(m.get("content"), list):
            out.append(m)
            continue
        has_image = any(b.get("type") == "image" for b in m["content"])
        if not has_image:
            out.append(m)
            continue
        if i not in kept_indices:
            out.append({"role": "user", "content": _strip_frames(m["content"])})
            continue
        position = kept_indices_ordered.index(i) + 1  # 1-based
        is_current = position == keep_count
        is_oldest = position == 1 and keep_count > 1
        if is_current:
            label = f"[frame {position} of {keep_count} — CURRENT (just captured)]"
        elif is_oldest:
            label = f"[frame {position} of {keep_count} — oldest in window, from when this part of the conversation started]"
        else:
            label = f"[frame {position} of {keep_count}]"
        labeled = []
        for b in m["content"]:
            if b.get("type") == "image":
                labeled.append({"type": "text", "text": label})
            labeled.append(b)
        out.append({"role": "user", "content": labeled})
    return out


def _summarize_assistant_turn(tool_list, reasoning):
    calls = "; ".join(
        f"{t['tool']}({json.dumps(t.get('args', {}))})" for t in (tool_list or [])
    )
    text_body = ((reasoning.strip() + "\n") if reasoning else "") + (
        f"Plan: {calls}" if calls else "Plan: (no tools)"
    )
    return {"role": "assistant", "content": text_body}


# ═══════════════════════════════════════════════════════════════════════
# 7. Anthropic HTTP client
# ═══════════════════════════════════════════════════════════════════════

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# Single-tier routing: every user turn (and every continuation / reengage
# round-trip) hits the orchestrator directly. Previously there was a
# Sonnet router layer that decided whether to escalate to Opus — ripped
# out 2026-04-23 because Opus is actually faster per-token, and routing
# through Sonnet just added a full round-trip of latency.
#
# OpenAI fallback: gpt-5.4 is the current API-available flagship (GPT-5.5
# was ChatGPT/Codex-only as of 2026-04-23; swap the string when 5.5 lands
# on the API). Costs ~1.5× Anthropic Opus — fine.
ANTHROPIC_MODEL = "claude-opus-4-7"
OPENAI_MODEL = "gpt-5.4"


class _RetryableUpstreamError(Exception):
    """Raised by _call_anthropic on 5xx / 429 / network / timeout errors —
    things that might succeed on a fallback provider. Non-retryable
    failures (4xx auth/bug, parse errors) raise plain RuntimeError instead
    so we don't mask programming errors behind a fallback."""


_ANTHROPIC_FALLBACK_STATUS = {408, 429, 500, 502, 503, 504, 529}


async def _call_anthropic(model, system, tools, messages):
    api_key = os.environ.get("T1_PROMPT_KEY", "")
    if not api_key:
        raise RuntimeError("T1_PROMPT_KEY not set in environment")
    timeout = aiohttp.ClientTimeout(total=30)
    body = {
        "model": model,
        "max_tokens": 1024,
        "system": system,
        "tools": tools,
        "messages": messages,
    }
    try:
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(
                ANTHROPIC_URL,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
            ) as r:
                raw = await r.text()
                if r.status in _ANTHROPIC_FALLBACK_STATUS:
                    raise _RetryableUpstreamError(f"anthropic {r.status}: {raw[:200]}")
                if r.status != 200:
                    raise RuntimeError(f"anthropic {r.status}: {raw[:400]}")
                return json.loads(raw)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        raise _RetryableUpstreamError(f"anthropic network/timeout: {e}")


def _extract_tool_calls(payload, valid_names):
    """Port of server.js extractToolCalls. Returns (tools, reasoning)."""
    blocks = payload.get("content") or []
    tools = []
    reasoning_parts = []
    for b in blocks:
        if b.get("type") == "text" and b.get("text"):
            reasoning_parts.append(b["text"])
        if b.get("type") == "tool_use" and b.get("name") in valid_names:
            tools.append({"tool": b["name"], "args": b.get("input") or {}})
    return tools, "".join(reasoning_parts).strip()


# ─── OpenAI shim (fallback provider) ─────────────────────────────────────
# Translates our Anthropic-shaped tool schemas + messages into OpenAI's
# chat-completions format, calls the API, parses the response back. The
# Pi brain's canonical representation stays Anthropic-shaped; translation
# happens only at call-time.

def _openai_tools(tools):
    """Anthropic tool schema → OpenAI function-calling schema.
    Anthropic: {name, description, input_schema}
    OpenAI:    {type:'function', function:{name, description, parameters}}"""
    out = []
    for t in tools:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return out


def _openai_messages(system, anthropic_messages):
    """Prepend system as a role=system message, translate each user turn's
    image+text content blocks into OpenAI format. Assistant turns in our
    session are stored as plain strings already (we don't re-use Anthropic
    tool_use blocks as assistant content), so they translate unchanged."""
    out = [{"role": "system", "content": system}]
    for m in anthropic_messages:
        role = m.get("role", "user")
        content = m.get("content")
        if role == "assistant" or isinstance(content, str):
            # Plain-text assistant turn or simple user string — pass through.
            out.append({"role": role, "content": content if isinstance(content, str) else ""})
            continue
        if not isinstance(content, list):
            out.append({"role": role, "content": ""})
            continue
        new_blocks = []
        for b in content:
            if not isinstance(b, dict):
                continue
            btype = b.get("type")
            if btype == "text":
                new_blocks.append({"type": "text", "text": b.get("text", "")})
            elif btype == "image":
                # Anthropic: {source: {type:'base64', media_type, data}}
                # OpenAI:    {type:'image_url', image_url:{url:'data:...;base64,...'}}
                src = b.get("source") or {}
                media = src.get("media_type", "image/jpeg")
                data = src.get("data", "")
                new_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media};base64,{data}"},
                })
            # Drop tool_use / tool_result if they somehow made it here —
            # session only stores summarized strings, so shouldn't happen.
        out.append({"role": role, "content": new_blocks})
    return out


async def _call_openai(model, system, tools, messages):
    """Chat-completions call. Returns the raw response JSON (OpenAI shape)."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set — can't fall back")
    timeout = aiohttp.ClientTimeout(total=30)
    # gpt-5.x uses `max_completion_tokens` (older `max_tokens` is a 400
    # error on the new models). This is the newer chat-completions name.
    body = {
        "model": model,
        "max_completion_tokens": 1024,
        "messages": _openai_messages(system, messages),
        "tools": _openai_tools(tools),
    }
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.post(
            OPENAI_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            json=body,
        ) as r:
            raw = await r.text()
            if r.status != 200:
                raise RuntimeError(f"openai {r.status}: {raw[:400]}")
            return json.loads(raw)


def _extract_tool_calls_openai(payload, valid_names):
    """Returns (tools, reasoning) in the SAME shape as _extract_tool_calls.
    OpenAI puts tool calls on assistant.message.tool_calls; arguments come
    back as JSON strings (not parsed objects), so we json.loads them."""
    choices = payload.get("choices") or []
    if not choices:
        return [], ""
    msg = choices[0].get("message") or {}
    reasoning = msg.get("content") or ""
    tools = []
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        name = fn.get("name")
        if name not in valid_names:
            continue
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except json.JSONDecodeError:
            args = {}
        tools.append({"tool": name, "args": args})
    return tools, reasoning.strip() if reasoning else ""


# ─── Fallback routing: Anthropic preferred, OpenAI on failure ────────────

# Sticky cooldown: when Anthropic errors (retriable) we avoid hitting it
# again for this window. Prevents slow failure-retry loops during outages.
ANTHROPIC_COOLDOWN_SEC = 120
_anthropic_cooldown_until = 0.0


async def _route_to_llm(system, tools, messages):
    """Single LLM call: Anthropic-preferred, OpenAI-fallback. Returns
    (tools_list, reasoning, source_label) where source is 'anthropic' or
    'openai' for diagnostics."""
    global _anthropic_cooldown_until
    valid_names = {t["name"] for t in tools}
    now = time.time()

    # During an Anthropic cooldown window, skip straight to OpenAI.
    if now < _anthropic_cooldown_until:
        print(f"[brain] anthropic cooldown active ({_anthropic_cooldown_until - now:.0f}s left); OpenAI direct")
        payload = await _call_openai(OPENAI_MODEL, system, tools, messages)
        calls, reasoning = _extract_tool_calls_openai(payload, valid_names)
        return calls, reasoning, "openai"

    try:
        payload = await _call_anthropic(ANTHROPIC_MODEL, system, tools, messages)
        calls, reasoning = _extract_tool_calls(payload, valid_names)
        return calls, reasoning, "anthropic"
    except _RetryableUpstreamError as e:
        _anthropic_cooldown_until = now + ANTHROPIC_COOLDOWN_SEC
        print(f"[brain] Anthropic failed ({e}); falling back to OpenAI, cooldown {ANTHROPIC_COOLDOWN_SEC}s")
        await broadcast_log("err", f"Anthropic down; falling back to OpenAI ({OPENAI_MODEL})")
        payload = await _call_openai(OPENAI_MODEL, system, tools, messages)
        calls, reasoning = _extract_tool_calls_openai(payload, valid_names)
        return calls, reasoning, "openai"
    # Non-retriable Anthropic errors (auth, parse, etc.) propagate up.


# ═══════════════════════════════════════════════════════════════════════
# 8. Tool execution (Phase 3d — poses / drive / wait / stop)
# ═══════════════════════════════════════════════════════════════════════

# Pose angle arrays copied verbatim from user.html / t1_test.html so the Pi
# receives exactly what it already knows how to execute. 'stand' is zero-
# keystates (matches user.html's stand command; the Pi's key_handling
# resets leg_targets when all keys are off).
POSE_PAYLOADS = {
    "sit":      [[90, 172, 142], [90, 200, 50], [90, 170, 144], [90, 200, 50]],
    "lie_down": [[90, 200, 135], [90, 200, 135], [90, 200, 135], [90, 200, 135]],
    "shake":    [[90, 200, 113], [90, 200, 50], [90, 170, 144], [120, 260, 0]],
}

# drive direction -> keystate map. LaiRA is a quadruped, no strafe —
# rotate_left/right map to a/d since q/e are unused on the Pi side.
DRIVE_KEYSTATES = {
    "forward":      {"w": True},
    "back":         {"s": True},
    "rotate_left":  {"a": True},
    "rotate_right": {"d": True},
}


def _main():
    """Return the running laira_go_wsprotected module object.

    We CANNOT use `import laira_go_wsprotected` here: the script is
    launched as `python3 laira_go_wsprotected.py`, so Python loads it
    under the module name `__main__`. An import-by-name lookup would then
    find the .py file on disk and load it AGAIN as a second module
    (`laira_go_wsprotected`), re-running .env loading, serial init, vosk
    model init (500MB!), and giving us a zombie second instance whose
    keystates dict nobody reads. The brain's writes would go to that
    dead dict and LaiRA would never move.

    sys.modules['__main__'] is the actual running script — same globals
    the control_loop reads from. This is the load-bearing fix; don't
    change it to an import() even if it looks cleaner."""
    import sys
    return sys.modules['__main__']


async def _execute_pose(pose_name):
    """Play a pose on the Pi. Mirrors /user.html's runCustomAnimations()
    timing: main angle → (for lie_down/shake) follow-up → motor_off. The
    client-side executePose was doing ~1.1s of setTimeout chain; we await
    equivalent asyncio.sleeps here."""
    main = _main()
    if pose_name == "stand":
        # `stand` means "get her upright from whatever pose she's in".
        # brain_stand_up() does motors_on + reset_function + settle wait.
        # (The old implementation was `brain_set_keystates({})` which is
        # a no-op when she's already at zero keystates — that's why stand
        # from lying silently did nothing.)
        await main.brain_stand_up()
        return

    arr = POSE_PAYLOADS.get(pose_name)
    if arr is None:
        print(f"[brain] _execute_pose: unknown {pose_name}")
        return
    # Pose commands need motors on but NOT a full stand-up reset — running
    # reset_function before sit would bring her to standing first and then
    # sit, which looks weird. Just ensure motors are live so angle
    # commands actually drive the servos.
    await main.brain_motors_on()
    # run_animation mutates angle_buffer + sends motor commands.
    await main.run_animation(arr, "angle")

    if pose_name == "lie_down":
        await asyncio.sleep(0.5)
        # Equivalent to user.html's follow-up {action:'rest'} → ctrl.motor_off.
        if main.ctrl is not None:
            try:
                main.ctrl.motor_off(main.SERVO_ID_ALL)
                main.motors_resting = True
            except Exception as e:
                print(f"[brain] motor_off(all) failed: {e}")
    elif pose_name == "shake":
        # user.html's shake is 3-step: pose1 → pose2 at +500ms → motor_off on
        # hip servos at +1000ms. Replicating.
        await asyncio.sleep(0.5)
        await main.run_animation(
            [[90, 200, 113], [90, 240, 100], [90, 170, 144], [120, 260, 0]],
            "angle",
        )
        await asyncio.sleep(0.5)
        if main.ctrl is not None:
            try:
                main.ctrl.motor_off(14)
                main.ctrl.motor_off(15)
                main.motors_resting = True
            except Exception as e:
                print(f"[brain] motor_off(14/15) failed: {e}")


# In-between snapshots captured during a rotate-with-in_between_shots call.
# Each entry: {index, total, direction, elapsed_ms, frame_b64}. Drained
# (consumed and cleared) by _grab_user_content on the next orchestrator
# round-trip so the frames flow into Opus's next input alongside the
# post-rotation current frame.
_pending_in_between_frames = []


async def _execute_drive(direction, duration_ms, in_between_shots=0):
    """Hold a direction keystate for duration_ms, then clear. Clear in a
    finally so a mid-drive cancel also stops the motion.

    For rotate_*, optionally captures `in_between_shots` evenly-spaced
    snapshots during the rotation. Snapshots land in the module-level
    _pending_in_between_frames buffer to be consumed by the next
    user-content build. For forward/back, in_between_shots is silently
    coerced to 0 — translation snapshots aren't useful since the camera
    is already pointed forward."""
    main = _main()
    ks = DRIVE_KEYSTATES.get(direction)
    if ks is None:
        await broadcast_log("err", f"drive: unknown direction {direction!r}")
        return
    is_rotate = direction in ("rotate_left", "rotate_right")
    if is_rotate:
        dur = max(100, min(15000, int(duration_ms or 500)))
    else:
        dur = max(100, min(3000, int(duration_ms or 500)))
        if in_between_shots:
            print(f"[brain] drive: ignoring in_between_shots={in_between_shots} on {direction} (rotate-only feature)")
        in_between_shots = 0
    n_in = max(0, min(10, int(in_between_shots or 0)))
    # Wake her up first — drive from lying/sitting used to do nothing
    # because motors were off and control_loop was in angle-mode.
    await main.brain_stand_up()
    try:
        await main.brain_set_keystates(ks)
        if n_in == 0:
            await asyncio.sleep(dur / 1000.0)
        else:
            # Capture at fractions 1/(N+1), 2/(N+1), ..., N/(N+1) of dur.
            # The post-rotation frame is captured separately by the
            # next orchestrator round-trip's _grab_user_content.
            start_t = time.time() * 1000.0
            for i in range(1, n_in + 1):
                target_ms = dur * i / (n_in + 1)
                wait_ms = max(0.0, target_ms - (time.time() * 1000.0 - start_t))
                if wait_ms > 0:
                    await asyncio.sleep(wait_ms / 1000.0)
                frame_b64, _wh = laira_frames.cache.grab_jpeg_b64(quality=85)
                actual_ms = int(time.time() * 1000.0 - start_t)
                if frame_b64 is not None:
                    _pending_in_between_frames.append({
                        "index": i,
                        "total": n_in,
                        "direction": direction,
                        "elapsed_ms": actual_ms,
                        "frame_b64": frame_b64,
                    })
                    print(f"[brain] drive in-between {i}/{n_in} ({direction}) at elapsed={actual_ms}ms — captured")
                else:
                    print(f"[brain] drive in-between {i}/{n_in} ({direction}) at elapsed={actual_ms}ms — NO FRAME")
            # Sleep any remaining time so total duration is honored.
            remaining_ms = dur - (time.time() * 1000.0 - start_t)
            if remaining_ms > 0:
                await asyncio.sleep(remaining_ms / 1000.0)
    finally:
        try:
            await main.brain_set_keystates({})
        except Exception:
            pass


async def _execute_wait(duration_ms):
    ms = max(100, min(30000, int(duration_ms or 1000)))
    await asyncio.sleep(ms / 1000.0)


async def _execute_stop():
    """Halt motion + clear session so the next user_text is a fresh
    conversation. Mirrors browser-side stop tool behavior."""
    main = _main()
    try:
        await main.brain_set_keystates({})
    except Exception:
        pass
    _clear_session()


async def _run_plan_loop(initial_tools):
    """Execute a plan, then continue asking the orchestrator what to do
    next until the session is stopped/canceled or no further tools come
    back. Handles two continuation-style round-trips:
      - continuePlan: fires after a plan runs to completion (Sonnet often
        returns one tool at a time even for compound requests, so we
        ask "what next" with the session history intact).
      - reengageOnStuck: fires when go_to plateaus above target — replaces
        the remaining plan with whatever Sonnet/Opus decides (switch to
        follow, settle, retry, or stop).
    """
    print(f"[brain] plan_loop entry: {len(initial_tools or [])} tool(s)")
    await broadcast_plan_state("EXECUTING")
    tools = list(initial_tools or [])
    last_step_desc = None
    # Budget to prevent a pathological re-engage loop where each re-engage
    # returns the same failing tool. Re-engages are useful but must
    # terminate. Reset on every successful non-reengage step so a long
    # multi-step plan isn't penalized by early recoveries.
    reengage_budget = 6
    try:
        while tools:
            planned = tools
            tools = []  # becomes non-empty only on reengage / continuation
            stopped = False
            replaced = False

            for t in planned:
                name = t.get("tool")
                args = t.get("args") or {}
                print(f"[brain] > {name}({json.dumps(args)})")
                await broadcast_log("tool", f"> {name}({json.dumps(args)})")
                try:
                    if name in ("sit", "stand", "lie_down", "shake"):
                        await _execute_pose(name)
                        last_step_desc = f"{name}() completed"
                    elif name == "drive":
                        _ib = int(args.get("in_between_shots") or 0)
                        await _execute_drive(args.get("direction"), args.get("duration_ms"), _ib)
                        if _ib:
                            last_step_desc = f"drive({args.get('direction')}, {args.get('duration_ms')}ms, in_between_shots={_ib}) completed — {_ib} in-between snapshots attached to your next input"
                        else:
                            last_step_desc = f"drive({args.get('direction')}, {args.get('duration_ms')}ms) completed"
                    elif name == "wait":
                        await _execute_wait(args.get("duration_ms"))
                        last_step_desc = f"wait({args.get('duration_ms')}ms) completed"
                    elif name == "stop":
                        await _execute_stop()
                        await broadcast_log("evt", "plan terminated by stop()")
                        stopped = True
                        break
                    elif name in ("follow", "go_to", "crawl"):
                        dist = args.get("target_distance_cm")
                        tgt = args.get("target")
                        if not tgt or not isinstance(dist, int) or not (5 <= dist <= 300):
                            await broadcast_log("err", f"{name}: bad args {args}")
                            stopped = True
                            break
                        # crawl carries an extra optional `speed` param, 1-4.
                        # Hard-clamp server-side (defense in depth on top of
                        # the schema minimum/maximum) so Opus literally can
                        # not use crawl as a speed-up hack.
                        speed_override = None
                        if name == "crawl":
                            raw_speed = args.get("speed", 2)
                            try:
                                speed_override = max(1, min(4, int(raw_speed)))
                            except (TypeError, ValueError):
                                speed_override = 2
                        outcome, depth, aux = await _execute_track(
                            name, tgt, dist, speed_override=speed_override
                        )
                        reengage_fn = None
                        reengage_label = None
                        if outcome == "stuck":
                            reengage_fn = _reengage_on_stuck(tgt, dist, depth)
                            reengage_label = "stuck"
                        elif outcome == "lost":
                            reengage_fn = _reengage_on_lost(tgt, dist, name)
                            reengage_label = "lost"
                        elif outcome == "vision_fail":
                            # Vision gave us a reason in `aux` — pass it
                            # through so Opus can refine the target
                            # description instead of repeating the same
                            # one that just failed.
                            reengage_fn = _reengage_on_vision_fail(tgt, dist, name, aux)
                            reengage_label = "vision_fail"
                        elif outcome == "init_fail":
                            # Transient tracker-side failure. One retry is
                            # worth trying; the budget prevents loops.
                            reengage_fn = _reengage_on_init_fail(tgt, dist, name, aux)
                            reengage_label = "init_fail"

                        if reengage_fn is not None:
                            if reengage_budget <= 0:
                                await broadcast_log("err", f"{name} {outcome}; reengage budget exhausted, giving up")
                                stopped = True
                                break
                            reengage_budget -= 1
                            new_tools = await reengage_fn
                            if not new_tools:
                                # Opus may have described its plan in text
                                # without emitting an actual tool_use block
                                # (known LLM failure mode). Retry ONCE with
                                # a stricter prompt that forces a tool call.
                                # If still nothing, force a clean stop() so
                                # the agent doesn't silently hang and the
                                # user gets a clear "I gave up" signal.
                                await broadcast_log("evt", f"reengage-{reengage_label} returned no tool — retrying with stricter prompt")
                                new_tools = await _force_tool_retry(reengage_label)
                                if not new_tools:
                                    await broadcast_log("err", f"reengage-{reengage_label} retry also returned no tool — forcing stop()")
                                    await _execute_stop()
                                    await broadcast_log("evt", "plan terminated by forced stop (Opus refused to choose a tool)")
                                    stopped = True
                                    break
                                await broadcast_log("evt", f"reengage-{reengage_label} retry succeeded: {len(new_tools)} new tool(s) (budget left: {reengage_budget})")
                            else:
                                await broadcast_log("evt", f"reengage-{reengage_label}: {len(new_tools)} new tool(s) (budget left: {reengage_budget})")
                            tools = new_tools
                            replaced = True
                            break
                        else:
                            # Clean outcome (arrived, canceled). Reset the
                            # reengage budget — a successful step means
                            # we're making progress, not looping.
                            reengage_budget = 6
                            last_step_desc = f"{name}({tgt!r}) {outcome}" + (f" at depth={depth}cm" if depth is not None else "")
                    else:
                        await broadcast_log("err", f"unknown tool {name!r}")
                        stopped = True
                        break
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    await broadcast_log("err", f"{name} failed: {e}")
                    stopped = True
                    break

            if stopped:
                print("[brain] plan_loop exit: stopped")
                return
            if replaced:
                print(f"[brain] plan_loop: replaced with {len(tools)} tool(s)")
                continue  # loop with replacement tools

            # Plan exhausted cleanly. If the session is still alive (i.e.
            # stop() wasn't called, cancel hasn't fired), ask Sonnet what's
            # next — the user's original request may still have uncovered
            # steps (e.g. "come to me, sit" with a Sonnet that only emitted
            # go_to initially).
            if _session is None:
                print("[brain] plan_loop exit: session cleared")
                return
            print(f"[brain] plan_loop continue: last={last_step_desc!r}")
            new_tools = await _continue_plan(last_step_desc or "previous step completed")
            if not new_tools:
                print("[brain] plan_loop exit: continue returned 0 tools")
                await broadcast_log("ok", "plan complete")
                return
            print(f"[brain] plan_loop: continuation gave {len(new_tools)} tool(s)")
            tools = new_tools
    except asyncio.CancelledError:
        print("[brain] plan_loop canceled")
        raise
    except Exception as e:
        print(f"[brain] plan_loop CRASHED: {e!r}")
        import traceback; traceback.print_exc()
        raise
    finally:
        await broadcast_plan_state("IDLE")


# ═══════════════════════════════════════════════════════════════════════
# 9. Entry points
# ═══════════════════════════════════════════════════════════════════════

# Protect against overlapping submit_text calls (user double-tapping send)
_submit_lock = asyncio.Lock()

# The currently-executing plan task. cancel_active() cancels it; a new
# submit_text cancels + replaces it.
_active_plan_task = None


async def _route_once(user_content):
    """Single T1 round-trip: call the orchestrator (Opus, falling back to
    GPT-5.4 on Anthropic failure), get tools, persist the turn. Used by
    both submit_text (fresh) and the continuation helpers. Returns [] on
    error — caller decides what to do with no-tools."""
    global _session
    if _session is None:
        _session = _new_session()
    prior = _session["messages"]
    messages_for_call = _build_call_messages(prior, user_content)

    await broadcast_plan_state("PLANNING")

    try:
        print(f"[brain] route: -> orchestrator")
        final_tools, final_reasoning, final_source = await _route_to_llm(
            OPUS_SYSTEM, OPUS_TOOLS, messages_for_call
        )
        print(f"[brain] route done: {final_source} -> {len(final_tools)} tool(s)")
    except Exception as e:
        print(f"[brain] T1 error: {e!r}")
        import traceback; traceback.print_exc()
        await broadcast_log("err", f"T1 error: {e}")
        return []

    if final_reasoning:
        await broadcast_reasoning(final_source, final_reasoning)
    for t in final_tools:
        await broadcast_tool_call(t["tool"], t.get("args", {}))

    # Persist turn (frames kept intact; rolling-buffer labeler decides
    # which survive into the NEXT call).
    stored_user = {"role": "user", "content": user_content}
    stored_asst = _summarize_assistant_turn(final_tools, final_reasoning)
    _session["messages"] = prior + [stored_user, stored_asst]
    _session["last_touch"] = time.time()

    return final_tools


def _grab_user_content(text, doa=None):
    """Bundle a fresh camera frame + text into the Anthropic user content
    block shape. Returns None if no fresh frame is available.

    If doa is "left"/"right"/"center" (computed from LaiRA's stereo
    mic), we prepend a small spatial hint to the text so Opus can
    incorporate it into navigation decisions. The hint format and
    interpretation contract are documented in OPUS_SYSTEM.

    If a recent drive(rotate_*, in_between_shots=N) populated the
    in-between snapshot buffer, those frames are inserted FIRST (each
    with a labeling text block immediately preceding it) so Opus knows
    which is the current frame vs an in-between. The buffer is drained
    on consume to avoid double-feeding."""
    frame_b64, _wh = laira_frames.cache.grab_jpeg_b64(quality=85)
    if frame_b64 is None or laira_frames.cache.age_seconds() > 5:
        return None
    if doa in ("left", "right", "center"):
        annotated = f"[voice came from your {doa}] {text}"
    else:
        annotated = text
    content = []
    global _pending_in_between_frames
    if _pending_in_between_frames:
        in_betweens = _pending_in_between_frames
        _pending_in_between_frames = []
        for f in in_betweens:
            content.append({
                "type": "text",
                "text": (
                    f"[in-between shot {f['index']}/{f['total']} from "
                    f"{f['direction']} at elapsed={f['elapsed_ms']}ms — "
                    f"NOT the current frame, this is a snapshot taken "
                    f"DURING that rotation]"
                ),
            })
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": f["frame_b64"]},
            })
    content.append({
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": frame_b64},
    })
    content.append({"type": "text", "text": annotated})
    return content


async def _continue_plan(last_step_desc):
    """Called after a plan exhausts cleanly. Asks Sonnet what to do next
    given the completed step. Returns the new tool list (possibly empty)."""
    await broadcast_log("evt", f"continue: {last_step_desc}")
    text = (
        f"STATUS: previous step finished — {last_step_desc}. Looking at "
        f"the current frame and the original user request, decide your next "
        f"tool to continue, or call stop() if the user's request is now "
        f"fully satisfied. One tool call."
    )
    user_content = _grab_user_content(text)
    if user_content is None:
        await broadcast_log("err", "continue: no frame; aborting")
        return []
    return await _route_once(user_content)


async def _force_tool_retry(prior_label):
    """Re-prompt Opus when its previous response was text-only (no
    tool_use block). This is a known LLM failure mode: Opus describes
    its planned action in text but forgets to actually call the tool.
    The retry prompt is short, blunt, and forbids text-only responses
    so the next round either yields a tool or a clean stop()."""
    text = (
        f"STATUS: your previous response on the {prior_label!r} reengage "
        f"contained reasoning text but NO tool_use block — meaning "
        f"nothing actually executed and the system is hanging. "
        f"You MUST emit exactly one tool_use call this turn. "
        f"Do NOT describe a plan in text — call the tool. "
        f"If you genuinely cannot decide what action would help, call "
        f"stop() to end the session cleanly. One tool call, no "
        f"deliberation. Pick anything actionable."
    )
    user_content = _grab_user_content(text)
    if user_content is None:
        await broadcast_log("err", "force-retry: no frame; aborting")
        return []
    return await _route_once(user_content)


async def _reengage_on_stuck(target, target_distance_cm, last_depth_cm):
    """Called when go_to plateaus above its target. Asks Sonnet to decide:
    switch to follow, retry go_to at a different distance, drive forward
    manually, or stop. Returns the new tool list (possibly empty)."""
    await broadcast_log("evt", f"reengage stuck at {target!r} depth={last_depth_cm}")
    text = (
        f"STATUS: go_to(\"{target}\", target_distance_cm={target_distance_cm}) "
        f"FAILED to reach the target. The tracker plateaued at "
        f"{last_depth_cm}cm and could not close the remaining "
        f"{max(0, last_depth_cm - target_distance_cm)}cm. THIS IS A FAILURE, "
        f"NOT A SUCCESSFUL APPROACH. Do not interpret \"got stuck at "
        f"{last_depth_cm}cm\" as \"arrived at {last_depth_cm}cm\".\n\n"
        f"Reread the user's original request and look at the current frame. "
        f"Ask yourself honestly: \"Does being {last_depth_cm}cm away from "
        f"{target} actually satisfy what the user asked for?\" For requests "
        f"like \"come over to me\" / \"come here\" / \"sit next to me\", "
        f"being 80cm+ away does NOT satisfy that intent — the user expects "
        f"you within ~30-50cm.\n\n"
        f"Decide ONE next tool:\n"
        f"(a) drive(forward, 1000-2000ms) to manually close the gap if the "
        f"current frame shows clear path forward to the target;\n"
        f"(b) go_to with a DIFFERENT target description — pick a more "
        f"trackable landmark closer to where you actually need to be (e.g. "
        f"\"the floor next to the chair\" instead of \"the chair\");\n"
        f"(c) follow() if the target is moving (e.g., the user walking around);\n"
        f"(d) stop() ONLY if you've concluded the gap genuinely cannot be "
        f"closed AND the user's request is satisfied at the current distance.\n\n"
        f"DO NOT call sit/lie_down/shake just because you're somewhat close — "
        f"unless the user explicitly asked for those AT a non-arrival "
        f"distance (rare). One tool call."
    )
    user_content = _grab_user_content(text)
    if user_content is None:
        await broadcast_log("err", "reengage: no frame; aborting")
        return []
    return await _route_once(user_content)


async def _reengage_on_vision_fail(target, target_distance_cm, mode, vision_reason):
    """Called when the vision model couldn't identify the target. Crucially
    different from 'lost' — here the frame was inspected and the target
    description didn't match what was visible. The vision model usually
    explains why ("chair is there but no person in it", "not visible",
    etc.), which lets Opus refine the target description, rotate a bit,
    or conclude the target isn't reachable. Returns new tool list."""
    await broadcast_log("evt", f"reengage vision_fail: {target!r} ({vision_reason[:80]})")
    text = (
        f"STATUS: I just tried to {mode}(\"{target}\", {target_distance_cm}cm) "
        f"but the vision module couldn't confidently identify that target in "
        f"the current frame. Vision's reasoning: \"{vision_reason}\". "
        f"Looking at the current frame, decide what to do. Options: "
        f"(a) call {mode} again with a REFINED target description — e.g., "
        f"describe a visible landmark instead (\"the gaming chair\" instead "
        f"of \"the person in the gaming chair\", or a specific color / "
        f"object that IS in the frame); "
        f"(b) rotate with drive(rotate_left/right, 400-600) to adjust the "
        f"viewing angle, then decide again on the next round; "
        f"(c) stop() if you're satisfied the target isn't findable from "
        f"this position. One tool call."
    )
    user_content = _grab_user_content(text)
    if user_content is None:
        await broadcast_log("err", "reengage: no frame; aborting")
        return []
    return await _route_once(user_content)


async def _reengage_on_init_fail(target, target_distance_cm, mode, err_detail):
    """Called when the SAMURAI tracker failed to initialize (ws open or
    session init failed — usually transient Modal issue). Brain can retry
    the same call (often fixes it), pick a different tool, or stop."""
    await broadcast_log("evt", f"reengage init_fail: {target!r} ({err_detail[:80]})")
    text = (
        f"STATUS: the SAMURAI tracker failed to start for {mode}(\"{target}\"). "
        f"Error: \"{err_detail}\". This is usually a transient issue on the "
        f"tracker service. Looking at the current frame, decide: "
        f"(a) retry the same tool call (often succeeds on second attempt); "
        f"(b) pick a different tool entirely; "
        f"(c) stop() if the user's intent is already satisfied. One tool call."
    )
    user_content = _grab_user_content(text)
    if user_content is None:
        await broadcast_log("err", "reengage: no frame; aborting")
        return []
    return await _route_once(user_content)


async def _reengage_on_lost(target, target_distance_cm, mode):
    """Called when SAMURAI returns null bbox for >2s during go_to/follow.
    Formerly the plan just terminated here with 'go_to ended: lost' and
    did nothing — really we should ask the brain to decide what to do
    (rotate to re-find, retry with a different target description, or
    stop). Returns the new tool list (possibly empty)."""
    await broadcast_log("evt", f"reengage lost: {target!r}")
    text = (
        f"STATUS: while running {mode}(\"{target}\"), I lost sight of the "
        f"target — SAMURAI returned no bbox for 2+ seconds. That usually "
        f"means the target moved out of frame faster than I could follow, "
        f"or something occluded them. Looking at the current frame, decide "
        f"what to do next. If you can SEE the target in the current frame, "
        f"just call {mode}/go_to/follow again (possibly with a broader "
        f"description). If not, rotate with drive(rotate_left/right) to "
        f"search — apply the same 360° search rule as on an initial search. "
        f"If you've concluded the target is gone for good, call stop(). "
        f"One tool call."
    )
    user_content = _grab_user_content(text)
    if user_content is None:
        await broadcast_log("err", "reengage: no frame; aborting")
        return []
    return await _route_once(user_content)


async def submit_text(text, continuation=False, doa=None):
    """Browser submitted a user-text command. Two-tier routing:

    Tier 1 (local, fast): laira_tier1.classify() runs a keyword/regex
    match on the phrase. If it unambiguously means sit/stand/lie_down/
    shake/stop, execute that action directly — zero cloud round-trip.
    ~300ms saved vs going through Opus for a bare "sit".

    Tier 2 (cloud, capable): any phrase that isn't an obvious single-
    action match (navigation, compound plans, questions, anything with
    a target object) escalates to Opus via _route_once. Same flow as
    before the tier-1 layer was added.

    continuation: unused externally now (continue/reengage are internal
    helpers). Kept for backward compat with future callers.
    doa: optional direction-of-arrival from the stereo mic — "left",
    "right", or "center" (front/behind ambiguous). When present, we
    prepend a [from your X] hint to the text Opus sees so it can
    factor that into navigation decisions. Tier-1 actions don't need
    spatial info so the hint is appended only to the tier-2 path."""
    global _active_plan_task, _session
    text = (text or "").strip()
    if not text:
        return
    async with _submit_lock:
        print(f"[brain] user-text: {text!r} doa={doa}")
        await broadcast_log("evt", f"user: {text}")

        if not continuation:
            # Mid-plan interrupt: halt motion + kill the current plan task
            # but KEEP the session's message history so prior context
            # carries forward (e.g. "laira, a little faster" mid-crawl
            # needs to know what she was crawling toward). Full session
            # reset only happens on explicit stop() or user-issued cancel.
            await interrupt_active()

        # ── Tier 1: local keyword classifier ────────────────────────────
        tier1_action = laira_tier1.classify(text)
        if tier1_action is not None:
            print(f"[brain] tier-1 direct: {tier1_action}")
            await broadcast_reasoning("tier1", f"local: {tier1_action}")
            await broadcast_tool_call(tier1_action, {})
            # Persist this turn to the session so a later escalated turn
            # sees the simple-action context. Opus's rolling-buffer
            # message-builder handles string-content user turns fine
            # (see _build_call_messages).
            if _session is None:
                _session = _new_session()
            turn_user = {"role": "user", "content": text}
            turn_asst = _summarize_assistant_turn(
                [{"tool": tier1_action, "args": {}}], ""
            )
            _session["messages"] = _session["messages"] + [turn_user, turn_asst]
            _session["last_touch"] = time.time()
            # Run the one-tool plan through the existing executor so
            # _execute_pose / _execute_stop logic applies — including the
            # dedupe + motors-on + session-cleanup semantics. If the
            # direct action is "stop", _execute_stop will _clear_session
            # correctly.
            tools = [{"tool": tier1_action, "args": {}}]
            _active_plan_task = asyncio.create_task(_run_plan_loop(tools))
            return

        # ── Tier 2: cloud Opus path (original flow) ─────────────────────
        user_content = _grab_user_content(text, doa=doa)
        if user_content is None:
            await broadcast_log("err", "no fresh camera frame — waiting for WebRTC handshake")
            return

        tools = await _route_once(user_content)

        if not tools:
            await broadcast_plan_state("IDLE")
            await broadcast_log("evt", "no action chosen")
            return

        # Spawn the plan-loop task. Handles continuation + reengage
        # internally. A subsequent submit_text will cancel this task via
        # cancel_active.
        _active_plan_task = asyncio.create_task(_run_plan_loop(tools))


async def _stop_active_motion():
    """Internal: kill the currently-running plan task + halt motion.
    Does NOT touch _session. Shared core for both interrupt_active (keeps
    session) and cancel_active (clears it)."""
    global _active_plan_task
    t = _active_plan_task
    _active_plan_task = None
    had_running_task = (t is not None and not t.done())
    if had_running_task:
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    # Only clear keystates if we actually had a plan running. The lewansoul
    # motor controller and the webcam mic share a USB bus on the Pi; every
    # serial roundtrip to the motors causes a measurable audio gap on the
    # outbound stream. handle_wake_detected calls interrupt_active on EVERY
    # wake fire, so without this gate we'd burn a USB transaction every
    # time the user said "lyra" — even when nothing was running and the
    # motors were already idle. Audio cuts on every wake event was the
    # symptom; this is the root-cause fix.
    if had_running_task:
        try:
            await _main().brain_set_keystates({})
        except Exception:
            pass


async def interrupt_active():
    """Soft cancel for mid-plan interrupts. Kills the current plan task
    and halts motion, but PRESERVES _session so the next user_text carries
    prior conversation context. Used when a new user_text arrives while a
    plan is running — usually a course correction ('a little faster',
    'also lie down') that needs to reason about what was already happening.

    NOT used by the stop() tool or the browser's explicit Stop button —
    those mean full disengage, so they call cancel_active() which also
    wipes the session."""
    await _stop_active_motion()
    await broadcast_plan_state("IDLE")


async def cancel_active():
    """Hard cancel: kill the plan, halt motion, AND clear session. Called
    by the stop() tool (Opus explicitly ending the session), by the
    browser's Stop button / manual WASD override ({cancel:true} WS
    message — user taking over from the AI), and by any other explicit
    full-reset path. After cancel_active the next user_text opens a fresh
    conversation with no memory of what came before."""
    await _stop_active_motion()
    _clear_session()
    await broadcast_plan_state("IDLE")


# ═══════════════════════════════════════════════════════════════════════
# 9. Lifecycle
# ═══════════════════════════════════════════════════════════════════════

async def start():
    """Background task spawned from main(). Kicks off Modal warmup + STT
    warmup concurrently so the user isn't waiting on cold inference
    latency at first use. Idle loop after."""
    print("[brain] start")
    await broadcast_log("ok", "brain started")
    await broadcast_hello()
    asyncio.create_task(warmup_modal())
    asyncio.create_task(_modal_keepalive_loop())
    asyncio.create_task(_parakeet_keepalive_loop())
    # STT warmup: load openwakeword + silero VAD (+ faster-whisper if
    # local backend) in a thread and run a warmup inference to eat the
    # ONNX JIT cost. Previous lazy-load behavior meant the first ~20-30s
    # after a restart, voice commands silently did nothing while models
    # loaded on-demand on the first audio frame.
    import laira_stt
    laira_stt.set_status_callback(broadcast_stt_status)
    asyncio.create_task(laira_stt.start_warmup())
    while True:
        await asyncio.sleep(60)


async def _modal_keepalive_loop():
    """Ping Modal /healthz periodically so the SAMURAI container stays
    warm. Without this, Modal's autoscaler scales the GPU container to
    zero after a couple minutes of idle, and the next go_to/follow/crawl
    pays a 25-30s cold-start penalty (visible as a delay between the
    `vision: ...` log line and LaiRA actually moving).

    Runs forever — when Modal is ready, ping; when warming, wait; when
    error, the existing reprobe loop is already polling so we step out.
    Interval is 90s — short enough to stay inside Modal's typical
    container-idle window (~120s default), long enough to keep request
    cost negligible (each /healthz is ~$0.0001)."""
    global _modal_status
    INTERVAL = 90
    # Wait for the initial warmup to finish before we start pinging,
    # otherwise we double up with the warmup_modal() task.
    while _modal_status not in ("ready", "error"):
        await asyncio.sleep(2)
    print(f"[brain] modal keepalive loop online (every {INTERVAL}s)")
    while True:
        await asyncio.sleep(INTERVAL)
        # Only ping when Modal is currently ready. If status went to
        # error, _modal_reprobe_loop is already polling — don't pile on.
        if _modal_status != "ready":
            continue
        try:
            ok, detail = await _modal_probe_once()
            if not ok:
                # Healthz failed — most likely Modal scaled to zero
                # despite our pings (rare, but possible under load).
                # Hand off to the reprobe loop and let it warm us back.
                _modal_status = "warming"
                print(f"[brain] modal keepalive saw failure: {detail} — kicking warmup")
                asyncio.create_task(_modal_reprobe_loop(interval=15))
        except Exception as e:
            print(f"[brain] modal keepalive errored: {e!r}")


async def _parakeet_keepalive_loop():
    """Ping Modal Parakeet /healthz periodically so the streaming-STT
    container stays warm. Without this, the next "lyra <cmd>" after
    a quiet period would pay a 25-30s cold-start penalty before the
    transcription even starts. Same architecture as
    _modal_keepalive_loop but for the Parakeet container.

    Skipped entirely if LAIRA_PARAKEET_WS_URL isn't set (i.e. user
    hasn't migrated to the Parakeet backend yet — they're on
    openai/local). No-op rather than warning so the brain still
    boots cleanly without the env var."""
    INTERVAL = 90
    ws_url = os.environ.get("LAIRA_PARAKEET_WS_URL", "").rstrip("/")
    if not ws_url:
        return
    # Derive HTTPS healthz URL from the WS URL. Modal serves both the
    # WS endpoint and the FastAPI /healthz off the same hostname.
    if ws_url.startswith("wss://"):
        healthz_url = "https://" + ws_url[len("wss://"):]
    elif ws_url.startswith("ws://"):
        healthz_url = "http://" + ws_url[len("ws://"):]
    else:
        healthz_url = ws_url
    # Strip any /ws path to get the FastAPI root.
    if healthz_url.endswith("/ws"):
        healthz_url = healthz_url[:-3]
    healthz_url = healthz_url.rstrip("/") + "/healthz"
    print(f"[brain] parakeet keepalive loop online (every {INTERVAL}s, healthz={healthz_url})")
    # Initial warm-up call so the first wake-fire doesn't cold-start.
    while True:
        try:
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                t0 = time.time()
                async with s.get(healthz_url) as r:
                    elapsed_ms = int((time.time() - t0) * 1000)
                    if r.status == 200:
                        if elapsed_ms > 3000:
                            print(f"[brain] parakeet ping took {elapsed_ms}ms (cold start?)")
                    else:
                        body = ""
                        try:
                            body = (await r.text())[:200]
                        except Exception:
                            pass
                        print(f"[brain] parakeet healthz {r.status}: {body}")
        except Exception as e:
            print(f"[brain] parakeet keepalive errored: {e!r}")
        await asyncio.sleep(INTERVAL)


async def on_ws_opened():
    """Called each time the main script's /laiRA outbound ws comes up.
    Re-sends hello + current modal+stt status so late-joining browsers
    can sync their UI."""
    await broadcast_hello()
    await broadcast_modal_status(_modal_status)
    await broadcast_stt_status(_stt_status, _stt_status_detail)


async def on_hello_request():
    """Browser asked for the current brain state. This fixes the startup
    race where the Pi-ws-open hello is emitted before any browser is
    connected to /user — the server-side relay drops those messages
    because userSocket is null at the time. Browser sends hello_request
    whenever it receives laiRA-connected to pull fresh state."""
    await broadcast_hello()
    await broadcast_modal_status(_modal_status)
    await broadcast_stt_status(_stt_status, _stt_status_detail)


# ═══════════════════════════════════════════════════════════════════════
# 10. Vision lookup (Phase 3e — T2)
#
# Port of server.js /api/vision. Calls Anthropic directly from the Pi —
# the /api/vision Render endpoint is bypassed entirely in the new
# architecture. Returns the bbox that fully contains the target, or None
# if the vision model can't find it with confidence.
# ═══════════════════════════════════════════════════════════════════════

async def call_vision(target, purpose, frame_b64, width, height):
    """Returns (bbox, reasoning, frame_b64). bbox is [x1, y1, x2, y2] or
    None. frame_b64 is passed through so the caller has a coherent
    frame+bbox pair to init SAMURAI with (prevents locking onto the wrong
    region because the frame moved during the LLM call)."""
    system = "\n".join([
        "You are LaiRA's vision module. You receive a single video frame and a target",
        "description. Identify the target and return a single JSON object — no",
        "surrounding text — of the form:",
        '  {"bbox": [x1, y1, x2, y2], "reasoning": "brief"}',
        f"Coordinates are integer pixels in the image's native resolution (image width={width or 'unknown'}, height={height or 'unknown'}).",
        "The bbox should fully contain the target with a small margin (~10px).",
        "",
        "If you cannot identify the target with confidence, return:",
        '  {"bbox": null, "reasoning": "why not"}',
        "",
        f"Caller's purpose: {purpose or 'unspecified'}.",
    ])

    api_key = os.environ.get("T1_PROMPT_KEY", "")
    if not api_key:
        raise RuntimeError("T1_PROMPT_KEY not set")

    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-opus-4-7",
                "max_tokens": 512,
                "system": system,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": frame_b64}},
                        {"type": "text", "text": target},
                    ],
                }],
            },
        ) as r:
            raw = await r.text()
            if r.status != 200:
                raise RuntimeError(f"vision {r.status}: {raw[:300]}")
            payload = json.loads(raw)
    text_blocks = [b.get("text", "") for b in (payload.get("content") or []) if b.get("type") == "text"]
    text = "".join(text_blocks).strip()
    # Extract the first JSON object in the text.
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        parsed = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        raise RuntimeError(f"vision returned no parseable JSON: {text[:200]}")
    return parsed.get("bbox"), parsed.get("reasoning") or "", frame_b64


# ═══════════════════════════════════════════════════════════════════════
# 11. SAMURAI WebSocket client (Phase 3e)
#
# Port of the browser-side SAMURAI client in smart_laira_test.html's
# SECTION 5. Key differences: runs as an asyncio task, pulls frames from
# the shared laira_frames.cache instead of a canvas capture, and race-
# guards against cancellation via a session_id counter.
# ═══════════════════════════════════════════════════════════════════════

SAMURAI_FRAME_HZ = 10                 # track-frame send rate
SAMURAI_FRAME_QUALITY = 70            # JPEG quality for track frames (lower = faster)
SAMURAI_FRAME_INTERVAL = 1.0 / SAMURAI_FRAME_HZ

# ── Persistent SAMURAI WebSocket ─────────────────────────────────────────
# Modal rate-limits new ws connections (HTTP 429 if you open too many in
# a window). Original design opened a fresh ws per follow/go_to call,
# which hit the limit after ~5 sessions in quick succession. The browser
# version (smart_laira_test.html) already used a long-lived ws — this
# brings the Pi port to parity. One ws stays open; sessions become
# init/close message pairs on it, not separate connections.

_sam_ws = None
_sam_ws_lock = asyncio.Lock()
_sam_ws_reader_task = None
# session_id -> SamuraiClient  (single active in practice; map for safety)
_sam_active_sessions = {}


def _ws_is_alive(ws):
    """websockets has shifted APIs across versions; probe both."""
    if ws is None:
        return False
    try:
        state = getattr(ws, 'state', None)
        if state is not None and getattr(state, 'name', '').upper() == 'OPEN':
            return True
        if hasattr(ws, 'closed'):
            return not ws.closed
    except Exception:
        pass
    return False


async def _ensure_sam_ws():
    """Get/create the persistent SAMURAI ws. Reconnects if Modal closed
    it. Handles 429 with linear backoff in-place so a brief rate-limit
    doesn't bubble up to a full reengage round-trip."""
    global _sam_ws, _sam_ws_reader_task
    async with _sam_ws_lock:
        if _ws_is_alive(_sam_ws):
            return _sam_ws
        _sam_ws = None  # whatever was there is dead

        modal_url = os.environ.get("MODAL_TRACKER_URL", "").rstrip("/")
        if not modal_url:
            raise RuntimeError("MODAL_TRACKER_URL not set")
        ws_url = modal_url.replace("http", "ws", 1) + "/ws"

        last_err = None
        for attempt, delay in enumerate([0.0, 2.0, 5.0]):
            if delay > 0:
                print(f"[brain] SAMURAI 429 backoff: sleeping {delay}s before retry {attempt}")
                await asyncio.sleep(delay)
            try:
                _sam_ws = await websockets.connect(ws_url, max_size=None, open_timeout=30)
                if _sam_ws_reader_task and not _sam_ws_reader_task.done():
                    _sam_ws_reader_task.cancel()
                _sam_ws_reader_task = asyncio.create_task(_sam_ws_reader(_sam_ws))
                print("[brain] SAMURAI ws established")
                # Successful ws connect proves Modal is alive — promote
                # status to ready (in case warmup_modal had marked it
                # error and the reprobe loop hadn't caught up yet).
                asyncio.create_task(_mark_modal_ready_from_evidence("ws connect ok"))
                return _sam_ws
            except Exception as e:
                last_err = e
                err_text = str(e)
                # Only retry on 429; everything else is something
                # backoff won't help with (auth, DNS, Modal genuinely down).
                if "429" not in err_text:
                    raise
                print(f"[brain] SAMURAI rate-limited (429): {err_text[:120]}")
        raise last_err


async def _sam_ws_reader(ws):
    """Single reader for the persistent ws. Routes each message to the
    client registered under that session_id. Falls through silently on
    stale/unknown session ids (covers the race after a session close)."""
    global _sam_ws
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            sid = msg.get("session_id")
            client = _sam_active_sessions.get(sid)
            if client is None:
                continue
            try:
                await client._handle_message(msg)
            except Exception as e:
                print(f"[brain] SAMURAI client handler error: {e}")
    except (asyncio.CancelledError, websockets.ConnectionClosed):
        pass
    except Exception as e:
        print(f"[brain] SAMURAI shared reader error: {e}")
    finally:
        # Mark global as dead so the next caller re-opens.
        if _sam_ws is ws:
            _sam_ws = None
        print("[brain] SAMURAI ws closed; will reconnect on next session")


class SamuraiClient:
    """Per-track session. Now uses the persistent _sam_ws shared across
    all sessions (was a fresh ws per session — that hit Modal's 429).
    open() ensures the shared ws is up; close() unregisters this session
    but leaves the ws open for the next track call."""

    def __init__(self):
        self.session_id = None
        self.on_bbox = None              # async callback(bbox, depth_cm)
        self.on_init_ok = None           # async callback(bbox)
        self.on_init_fail = None         # async callback(reason)
        self.on_lost = None              # async callback()
        self._frame_sender_task = None
        self._sent_in_flight = False
        self._closed = False

    async def open(self):
        # Ensures the shared ws is alive. Doesn't open per-instance.
        await _ensure_sam_ws()

    async def init_session(self, frame_b64, bbox_xyxy):
        """bbox_xyxy: [x1, y1, x2, y2]. Sends init on the shared ws and
        registers this client as the handler for the new session_id."""
        await _ensure_sam_ws()  # in case ws died between open() and here
        self.session_id = "sam-" + str(int(time.time() * 1000))
        _sam_active_sessions[self.session_id] = self
        await _sam_ws.send(json.dumps({
            "type": "init",
            "session_id": self.session_id,
            "frame": frame_b64,
            "bbox": list(bbox_xyxy),
        }))

    async def _handle_message(self, msg):
        """Called by the shared reader for each message tagged with our
        session_id. Was previously inline in the per-instance _reader."""
        if self._closed:
            return
        t = msg.get("type")
        if t == "init_ok" and self.on_init_ok:
            await self.on_init_ok(msg.get("bbox"))
            if self._frame_sender_task is None or self._frame_sender_task.done():
                self._frame_sender_task = asyncio.create_task(self._frame_sender())
        elif t == "init_fail" and self.on_init_fail:
            await self.on_init_fail(msg.get("error") or "")
        elif t == "bbox":
            self._sent_in_flight = False
            bbox = msg.get("bbox")
            depth = None
            debug = msg.get("debug") or {}
            if debug.get("depth_cm") is not None:
                depth = debug["depth_cm"]
            if bbox and self.on_bbox:
                await self.on_bbox(bbox, depth)
            elif bbox is None and self.on_lost:
                await self.on_lost()
        elif t == "error":
            self._sent_in_flight = False

    async def _frame_sender(self):
        """Send track frames on the shared ws at SAMURAI_FRAME_HZ.
        Back-pressure via _sent_in_flight."""
        try:
            while not self._closed:
                if not _ws_is_alive(_sam_ws):
                    return  # ws went away; let session terminate
                if not self._sent_in_flight:
                    frame_b64, wh = laira_frames.cache.grab_jpeg_b64(
                        quality=SAMURAI_FRAME_QUALITY
                    )
                    if frame_b64 is not None:
                        self._sent_in_flight = True
                        try:
                            await _sam_ws.send(json.dumps({
                                "type": "track",
                                "session_id": self.session_id,
                                "frame": frame_b64,
                            }))
                        except Exception:
                            return
                await asyncio.sleep(SAMURAI_FRAME_INTERVAL)
        except asyncio.CancelledError:
            return

    async def close(self):
        """End THIS session — sends close to Modal and unregisters from
        the active-sessions map. Does NOT close the shared ws (which
        will be reused for the next track call)."""
        self._closed = True
        if self._frame_sender_task and not self._frame_sender_task.done():
            self._frame_sender_task.cancel()
        if self.session_id is not None:
            _sam_active_sessions.pop(self.session_id, None)
            if _ws_is_alive(_sam_ws):
                try:
                    await _sam_ws.send(json.dumps({"type": "close", "session_id": self.session_id}))
                except Exception:
                    pass


# ═══════════════════════════════════════════════════════════════════════
# 12. Motion controller (Phase 3e — T3)
#
# Port of smart_laira_test.html's SECTION 6. Converts bbox + depth into
# keystates, detects arrived/stuck/lost. Sequential turn-then-walk active
# (because the Pi's key_handling drops w+a/d — see laira_mvp_pi_verify
# memory).
# ═══════════════════════════════════════════════════════════════════════

HISTORY_LEN = 12
DISTANCE_STABILITY_CM = 5
STABILITY_FRAMES_ARRIVED = 3
DISTANCE_TOL_CM = 8
FOLLOW_BAND_CM = 20
LOST_GRACE_MS = 2000
# Two thresholds for hysteresis on turning. "Start" is the
# off-centeredness she needs to begin a turn; "stop" is the (much
# tighter) centeredness she has to reach before she'll stop turning.
# Without hysteresis a single threshold causes oscillation: she
# turns just enough to clear the threshold, walks, falls back outside
# it, repeats. Two-band logic makes her commit to a real recentering
# once she starts turning. Frac is fraction-of-frame-width of the
# absolute pixel offset (cx - frame_width/2).
HORIZ_TURN_START_FRAC = 0.156   # ~must exceed to begin turning
HORIZ_TURN_STOP_FRAC = 0.05     # ~must drop below to stop turning
STUCK_MIN_ELAPSED_MS = 6000
STUCK_REGRESSION_CM = 20
STUCK_REGRESSION_DUR_MS = 3000
PROGRESS_TOL_CM = 5

# Control-panel slider state from browser. Defaulting to matches the
# current /smart_laira_test sliders (speed=5 baseline, closeness=3).
# 3g will forward the browser's actual slider values here.
_ui_speed = 5
_ui_closeness = 3


def set_ui_speed(v):
    global _ui_speed
    try:
        _ui_speed = max(1, min(10, int(v)))
    except (TypeError, ValueError):
        pass


def set_ui_closeness(v):
    global _ui_closeness
    try:
        _ui_closeness = max(1, min(8, int(v)))
    except (TypeError, ValueError):
        pass


def _now_ms():
    return time.time() * 1000.0


class TrackingSession:
    """Per-follow/go_to/crawl state. Mirrors state.currentSession on the
    browser today. Holds history, best-depth tracker, timestamps, and
    an optional per-session speed override (used by crawl to force a
    slow gait independent of the browser speed slider)."""

    def __init__(self, mode, target, target_distance_cm, speed_override=None):
        self.mode = mode                          # 'follow' | 'go_to' | 'crawl'
        self.target = target
        self.target_distance_cm = target_distance_cm
        # When set, this int takes precedence over the global _ui_speed in
        # _compute_keystates — that's how crawl enforces its slow gait
        # regardless of where the browser's speed slider is parked.
        self.speed_override = speed_override      # None = use _ui_speed
        self.started_at = _now_ms()
        self.last_bbox_at = self.started_at
        self.best_depth_cm = None
        self.best_depth_at = 0
        self.regressed_since_ms = None
        self.history = []                         # list of {depthCm, cx, t}
        self.last_sent_keystate_sig = None        # per-tracking dedupe
        # Hysteresis state for turn control. Either None (=she's
        # within the tight stop-band, considered centered) or
        # "left"/"right" (=she's actively turning that way and
        # will keep turning until well-centered, regardless of
        # whether the box has crossed back inside the wider start
        # threshold).
        self.turning_dir = None


def _push_history(sess, bbox, depth_cm):
    x, y, w, h = bbox
    sess.history.append({
        "x": x, "y": y, "w": w, "h": h, "cx": x + w / 2,
        "depthCm": None if depth_cm is None else depth_cm,
        "t": _now_ms(),
    })
    while len(sess.history) > HISTORY_LEN:
        sess.history.pop(0)


def _distance_stable(sess, n):
    if len(sess.history) < n:
        return False
    tail = sess.history[-n:]
    depths = [e["depthCm"] for e in tail if e["depthCm"] is not None]
    if len(depths) < n:
        return False
    return (max(depths) - min(depths)) <= DISTANCE_STABILITY_CM


def _compute_keystates(sess, bbox, depth_cm, frame_width):
    """Returns dict {keystates, action, arrived, stuck, status, effective_speed}.
    Sequential turn-then-walk because the Pi drops simultaneous w+a/d — see
    memory: laira_mvp_pi_verify.md item 1."""
    x, _, w, _ = bbox
    cx = x + w / 2
    offset = cx - frame_width / 2
    abs_off = abs(offset)
    start_thresh = frame_width * HORIZ_TURN_START_FRAC
    stop_thresh = frame_width * HORIZ_TURN_STOP_FRAC
    target = sess.target_distance_cm

    # Hysteresis: if not currently turning, only start turning once
    # the offset exceeds the wider start threshold. If already
    # turning, only stop once it drops inside the tight stop band
    # — this is what makes her commit to recentering on the actual
    # center instead of oscillating around the start-threshold edge.
    if sess.turning_dir is None:
        if abs_off > start_thresh:
            sess.turning_dir = "right" if offset > 0 else "left"
    else:
        if abs_off < stop_thresh:
            sess.turning_dir = None
    centered = sess.turning_dir is None

    ks = {"w": False, "a": False, "s": False, "d": False, "q": False, "e": False}
    arrived = False
    stuck = False
    status = "tracking"
    # Base speed: per-session override (set by crawl) trumps the browser's
    # UI slider. Once chosen, the closeness-divisor turn scaling operates
    # on that base so crawl turns remain proportionally slow near center.
    base_speed = float(sess.speed_override if sess.speed_override is not None else _ui_speed)
    effective_speed = base_speed
    turn_part = ""
    dist_part = ""

    if not centered:
        if sess.turning_dir == "right":
            ks["d"] = True; turn_part = "right"
        else:
            ks["a"] = True; turn_part = "left"
        # Turn-speed scaling: slower near center, full at frame edge.
        # Denominator now uses start_thresh (the outer band) so the
        # speed curve still peaks at frame edge and falls off as the
        # box approaches the start-threshold; well inside that, we're
        # in the recentering phase and can keep slowing toward stop_thresh.
        half_w = frame_width / 2
        denom = max(1.0, half_w - start_thresh)
        x_norm = max(0.0, min(1.0, (abs_off - start_thresh) / denom))
        divisor = _ui_closeness - (_ui_closeness - 1) * x_norm
        effective_speed = base_speed / divisor

    if sess.mode == "go_to":
        if depth_cm is None:
            if centered:
                ks["w"] = True
            dist_part = "approaching (no depth)"
            status = "approach"
        elif depth_cm > target + DISTANCE_TOL_CM:
            if centered:
                ks["w"] = True
            dist_part = f"approaching (Δ={int(depth_cm - target)}cm)"
            status = "approach"
        else:
            dist_part = "at distance"
            status = "arrived" if centered else "approach"

        if depth_cm is not None:
            now = _now_ms()
            if sess.best_depth_cm is None:
                sess.best_depth_cm = depth_cm
                sess.best_depth_at = now
                sess.regressed_since_ms = None
            elif depth_cm < sess.best_depth_cm - PROGRESS_TOL_CM:
                sess.best_depth_cm = depth_cm
                sess.best_depth_at = now
                sess.regressed_since_ms = None
            elif depth_cm > sess.best_depth_cm + STUCK_REGRESSION_CM:
                if sess.regressed_since_ms is None:
                    sess.regressed_since_ms = now
            else:
                sess.regressed_since_ms = None

            dist_stable = _distance_stable(sess, STABILITY_FRAMES_ARRIVED)
            if dist_stable and centered and depth_cm <= target + DISTANCE_TOL_CM:
                arrived = True
            elif depth_cm > target + DISTANCE_TOL_CM:
                elapsed = now - sess.started_at
                regressed_for = 0 if sess.regressed_since_ms is None else (now - sess.regressed_since_ms)
                if elapsed >= STUCK_MIN_ELAPSED_MS and regressed_for >= STUCK_REGRESSION_DUR_MS:
                    stuck = True
    else:
        # follow mode. Walk-keys are gated on  for the same
        # reason go_to does it: the Pi's USB serial pipeline drops
        # simultaneous w+a/d (and s+a/d) — see laira_mvp_pi_verify
        # memory. Without this gate, off-center + closing or off-center
        # + backing-off freezes her until she happens to recenter.
        if depth_cm is None:
            dist_part = "following (no depth)"
            status = "hold"
        elif depth_cm < target - FOLLOW_BAND_CM:
            if centered:
                ks["s"] = True
            dist_part = f"backing off (Δ={int(target - depth_cm)}cm)"
            status = "approach"
        elif depth_cm > target + FOLLOW_BAND_CM:
            if centered:
                ks["w"] = True
            dist_part = f"closing (Δ={int(depth_cm - target)}cm)"
            status = "approach"
        else:
            dist_part = "holding distance"
            status = "hold" if centered else "approach"

    if turn_part and dist_part:
        action = f"{dist_part} + turn {turn_part}"
    elif turn_part:
        action = f"centering: turn {turn_part}"
    else:
        action = dist_part or "tracking"

    return {
        "keystates": ks, "action": action, "arrived": arrived, "stuck": stuck,
        "status": status, "effective_speed": effective_speed, "offset": offset,
        "depth_cm": depth_cm,
    }


# ═══════════════════════════════════════════════════════════════════════
# 13. follow / go_to execution (Phase 3e)
# ═══════════════════════════════════════════════════════════════════════

async def _execute_track(mode, target, target_distance_cm, speed_override=None):
    """mode: 'follow' | 'go_to' | 'crawl'. Does vision lookup, opens SAMURAI
    session, runs the motion controller loop.

    speed_override: optional int 1-10. When set (crawl passes it through)
    it replaces _ui_speed as the base speed for this session, independent
    of the browser slider. Closeness-divisor turn scaling still applies
    on top. None = use the current _ui_speed as normal.

    Returns (outcome, depth_cm, aux) where:
      outcome in {'arrived', 'stuck', 'lost', 'canceled', 'init_fail', 'vision_fail'}
      depth_cm: last known depth, or None
      aux: str — extra context (vision reasoning, error text) that the
           plan loop may feed into a reengage prompt. '' if not applicable.
    """
    main = _main()

    # Ensure she's standing before we start firing tracking keystates —
    # otherwise a go_to from a lying state silently does nothing.
    await main.brain_stand_up()

    # ── Vision lookup ──────────────────────────────────────────────────
    frame_b64, wh = laira_frames.cache.grab_jpeg_b64(quality=85)
    if frame_b64 is None or wh is None or laira_frames.cache.age_seconds() > 5:
        await broadcast_log("err", f"{mode}: no fresh camera frame")
        return "vision_fail", None, "no fresh camera frame from WebRTC stream"
    w, h = wh

    await broadcast_plan_state("EXECUTING", f"{mode}: {target} @{target_distance_cm}cm (vision lookup)")
    try:
        bbox_xyxy, reasoning, _ = await call_vision(target, mode, frame_b64, w, h)
    except Exception as e:
        await broadcast_log("err", f"vision failed: {e}")
        return "vision_fail", None, f"vision call exception: {e}"
    if bbox_xyxy is None:
        await broadcast_log("err", f"vision: couldn't find {target!r} — {reasoning}")
        return "vision_fail", None, reasoning or "vision model said: target not found"
    if reasoning:
        await broadcast_reasoning("vision", reasoning)

    # ── Tracking session ───────────────────────────────────────────────
    sess = TrackingSession(mode, target, target_distance_cm, speed_override=speed_override)
    outcome_future = asyncio.get_event_loop().create_future()
    latest_depth = [None]  # mutable box so nested callbacks can set it

    client = SamuraiClient()

    async def on_init_ok(bb):
        await broadcast_log("ok", f"SAMURAI init_ok {bb}")

    async def on_init_fail(err):
        await broadcast_log("err", f"SAMURAI init_fail: {err}")
        if not outcome_future.done():
            outcome_future.set_result("init_fail")

    async def on_bbox(bbox_xywh, depth_cm):
        # Broadcast bbox to browser when the toggle is on.
        if bbox_stream_enabled():
            await broadcast_bbox(bbox_xywh)
        _push_history(sess, bbox_xywh, depth_cm)
        sess.last_bbox_at = _now_ms()
        latest_depth[0] = depth_cm
        info = _compute_keystates(sess, bbox_xywh, depth_cm, w)
        await broadcast_plan_state("TRACKING", f"{mode}: {target} — {info['action']}")
        await broadcast_tracker_stats(
            mode=mode, target=target_distance_cm, depth_cm=depth_cm,
            action=info["action"], status=info["status"],
        )
        # Dedupe within the session loop (matches browser lastTrackingKeystateSig).
        sig = json.dumps({"ks": info["keystates"], "sp": info["effective_speed"]})
        if sig != sess.last_sent_keystate_sig:
            sess.last_sent_keystate_sig = sig
            try:
                await main.brain_set_keystates(info["keystates"], speed=info["effective_speed"])
            except Exception as e:
                await broadcast_log("err", f"keystate write failed: {e}")
        if info["arrived"]:
            await broadcast_log("ok", f"arrived at {target!r} (depth={depth_cm}cm)")
            if not outcome_future.done():
                outcome_future.set_result("arrived")
        elif info["stuck"]:
            await broadcast_log("err", f"stuck approaching {target!r}")
            if not outcome_future.done():
                outcome_future.set_result("stuck")

    async def on_lost():
        sinceMs = _now_ms() - sess.last_bbox_at
        await broadcast_tracker_stats(
            mode=mode, target=target_distance_cm, depth_cm=None,
            action=f"lost ({int(sinceMs)}ms)", status="lost",
        )
        if sinceMs > LOST_GRACE_MS:
            await broadcast_log("err", f"lost {target!r} (no bbox for {int(sinceMs)}ms)")
            if not outcome_future.done():
                outcome_future.set_result("lost")
        try:
            await main.brain_set_keystates({})
        except Exception:
            pass

    client.on_init_ok = on_init_ok
    client.on_init_fail = on_init_fail
    client.on_bbox = on_bbox
    client.on_lost = on_lost

    # Open the ws + init the tracker. Vision bbox is xyxy; SAMURAI init
    # wants xyxy too. Convert to xywh only for our internal history.
    try:
        await client.open()
    except Exception as e:
        await broadcast_log("err", f"SAMURAI open failed: {e}")
        return "init_fail", None, f"SAMURAI ws open failed: {e}"
    x1, y1, x2, y2 = bbox_xyxy
    init_xywh = [x1, y1, x2 - x1, y2 - y1]
    # on_bbox callback uses xywh (what SAMURAI emits on 'bbox' messages).
    try:
        await client.init_session(frame_b64, bbox_xyxy)
    except Exception as e:
        await broadcast_log("err", f"SAMURAI init_session failed: {e}")
        await client.close()
        return "init_fail", None, f"SAMURAI init_session failed: {e}"
    await broadcast_plan_state("EXECUTING", f"{mode}: {target} @{target_distance_cm}cm (tracking)")
    # Seed the first history entry with the vision bbox (approximate).
    _push_history(sess, init_xywh, None)

    try:
        outcome = await outcome_future
    except asyncio.CancelledError:
        await client.close()
        try:
            await main.brain_set_keystates({})
        except Exception:
            pass
        raise
    await client.close()
    try:
        await main.brain_set_keystates({})
    except Exception:
        pass
    return outcome, latest_depth[0], ""
