"""Phonetic correction for STT mishearings.

Problem: ASR confuses near-homophones — "lie down" → "why down",
"sit down" → "kit down", "rest" → "wrist". Whisper/Parakeet
don't know LaiRA's command vocabulary. This module post-processes
transcripts: if no known command appears literally, look for a
phonetic near-match (≤1 phoneme away) and substitute.

Two-stage gate so legit utterances flow through:
  1. SUBSTRING CHECK — if any known command appears as a literal
     substring of the transcript, return as-is. Catches "please
     lie down" / "go ahead and sit" / etc.
  2. PHONETIC EDIT DISTANCE — only runs when (1) didn't fire.
     ARPABET phoneme sequences via CMU dict; Levenshtein with
     equal-cost substitution. Threshold of 1 phoneme catches
     "why down" → "lie down" but rejects anything substantially
     different in length or content (length difference dominates
     edit distance for short commands).

Latency: <2ms worst case. Negligible vs the 150-2000ms STT itself.

Maintenance: COMMANDS list is the contract. Add new commands or
phrasings as needed. Single-word entries are deliberately omitted
from phonetic-correction targets (too easy to false-positive on
random short utterances) — they still match via substring.
"""

import re
import functools
from typing import Optional

import cmudict

_dict = None
def _get_dict():
    global _dict
    if _dict is None:
        _dict = cmudict.dict()
    return _dict


# Each entry maps a CANONICAL command phrase (what the brain will
# receive after correction) to its accepted spoken variations. The
# canonical itself is included in its own variations so substring
# matching works on the canonical too. Single-word variations
# ("down", "up", "sit") are useful for substring matching but
# excluded from phonetic-correction matching to avoid over-eager
# substitutions on misheard short utterances.
# Multi-word variations only. Single-word forms ("down", "up",
# "sit", etc.) are deliberately excluded — they'd trigger the
# substring gate on legitimately-misheard phrases like "why down"
# (preventing the phonetic correction we actually want), AND Opus
# interprets bare one-word commands correctly without our help.
COMMANDS = {
    "lie down":      ["lie down", "lay down"],
    "stand up":      ["stand up", "get up"],
    "sit down":      ["sit down"],
    "come here":     ["come here", "come over"],
    "follow me":     ["follow me"],
    "spin around":   ["spin around", "turn around"],
    "give paw":      ["give paw", "shake paw"],
    "back up":       ["back up", "back away"],
    "go forward":    ["go forward", "walk forward"],
}


def _phonemes(text: str) -> Optional[list]:
    """ARPABET phonemes (without stress digits) for . Returns
    None if any word is missing from the CMU dictionary — we'd
    rather skip phonetic matching than guess, since unknown words
    usually mean the transcript was unusual to begin with."""
    d = _get_dict()
    out = []
    for word in re.findall(r"[a-z']+", text.lower()):
        prons = d.get(word)
        if not prons:
            return None
        # Take the first pronunciation; strip stress digits (1,2,0).
        for p in prons[0]:
            out.append(re.sub(r"\d", "", p))
    return out


def _edit_distance(a: list, b: list) -> int:
    """Standard Levenshtein on phoneme sequences."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,         # delete
                curr[j - 1] + 1,     # insert
                prev[j - 1] + (0 if ca == cb else 1),  # substitute
            )
        prev = curr
    return prev[-1]


# Cache: variation -> phoneme sequence. Computed once at module load.
@functools.lru_cache(maxsize=None)
def _variation_phonemes(variation: str) -> Optional[tuple]:
    p = _phonemes(variation)
    return tuple(p) if p else None


# Phonetic-match candidates: only multi-syllable variations (≥2
# phonemes) so single-word commands like "up" / "down" / "sit"
# don't get substituted in via stretched matches. They still flow
# through the substring gate fine.
_PHONETIC_CANDIDATES = []
for canonical, variations in COMMANDS.items():
    for variation in variations:
        words = variation.split()
        if len(words) < 2:
            continue
        ph = _variation_phonemes(variation)
        if ph and len(ph) >= 4:  # at least 4 phonemes (~2 syllables)
            _PHONETIC_CANDIDATES.append((canonical, variation, ph))


def _has_command_substring(text: str) -> bool:
    """True if any command variation appears as a whole-word
    substring in . Word-boundary anchored so "stop" doesn't
    match "stopwatch"."""
    norm = " " + re.sub(r"[^a-z' ]", " ", text.lower()) + " "
    for variations in COMMANDS.values():
        for variation in variations:
            if f" {variation} " in norm:
                return True
    return False


# Maximum acceptable phoneme edit distance. 1 catches the typical
# single-phoneme ASR substitution ("why" vs "lie") and rejects
# everything else. Length similarity is implicit because the
# Levenshtein distance includes insert/delete penalties — a
# substantially-longer transcript can't match a short command at
# distance 1.
_MAX_DISTANCE = 1


def correct(transcript: str) -> tuple[str, Optional[str]]:
    """Apply phonetic correction.

    Returns (corrected_text, reason). reason is None if the
    transcript was passed through unchanged; otherwise it's a
    short string describing the substitution (logged by caller).

    The transcript is treated case-insensitively but the returned
    corrected text is the canonical command phrase verbatim.
    """
    if not transcript or not transcript.strip():
        return transcript, None
    # Stage 1: substring gate.
    if _has_command_substring(transcript):
        return transcript, None
    # Stage 2: phonetic match.
    target = _phonemes(transcript)
    if target is None:
        return transcript, None
    target_len = len(target)
    best_canonical = None
    best_dist = _MAX_DISTANCE + 1
    best_variation = None
    for canonical, variation, ph in _PHONETIC_CANDIDATES:
        # Skip variations whose length differs by more than the
        # threshold — they can't possibly match within the budget,
        # and skipping them is faster than running edit distance.
        if abs(len(ph) - target_len) > _MAX_DISTANCE:
            continue
        dist = _edit_distance(target, list(ph))
        if dist < best_dist:
            best_dist = dist
            best_canonical = canonical
            best_variation = variation
    if best_canonical is not None and best_dist <= _MAX_DISTANCE:
        reason = f"{transcript!r} → {best_canonical!r} via {best_variation!r} (phoneme dist {best_dist})"
        return best_canonical, reason
    return transcript, None


if __name__ == "__main__":
    # Sanity: run on a battery of expected/unexpected inputs.
    cases = [
        ("why down",                  "lie down"),
        ("why down?",                 "lie down"),
        ("Lay down",                  "Lay down"),       # substring -> passthrough
        ("please lie down",           "please lie down"), # substring -> passthrough
        ("why don't you chill",       "why don't you chill"),
        ("up and at 'em",             "up and at 'em"),
        ("kit down",                  "sit down"),
        ("wrist",                     "wrist"),           # single-word, not a phonetic candidate
        ("come hear",                 "come here"),
        ("stand off",                 "stand off"),
        ("stand up",                  "stand up"),
        ("hello there",               "hello there"),
        ("",                          ""),
    ]
    for inp, expected in cases:
        out, reason = correct(inp)
        ok = "✓" if out.lower() == expected.lower() else "✗"
        print(f"  {ok} {inp!r:40} -> {out!r:30} [{reason or 'passthrough'}]")
