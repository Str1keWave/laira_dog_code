"""
laira_tier1.py — local keyword-based router that bypasses cloud Opus
for simple commands (sit / stand / lie_down / shake / stop).

Rationale: every round-trip to Opus is ~1500ms. For "sit" that's painful —
dogs don't think for 1.5s before sitting. By running a regex match on
the transcribed phrase BEFORE reaching out to Opus, we cut simple-
command latency from ~3s to ~1s (dominated by STT cost alone).

Anything that isn't an unambiguous single-action match — compound
phrases, navigation, conditionals, anything with a target object,
negations, uncertainty — escalates to Opus, same as before.

Design principles:
  - When in doubt, ESCALATE. Opus is smarter than us; false negatives
    in the classifier just mean slightly slower execution, not wrong
    execution.
  - Multi-action phrases ("sit then stand") always escalate; Opus
    handles compound plans properly.
  - Common "conversational" qualifiers (please, can you, would you)
    are stripped before matching, so "can you sit down please" still
    direct-routes.
"""

import re

# Actions this tier can handle without escalation. Must match
# laira_brain's tool names exactly so the brain can dispatch by name.
DIRECT_ACTIONS = ("sit", "stand", "lie_down", "shake", "stop")

# Each action has a list of regex patterns. ANY pattern match = candidate
# for that action. Pattern design rules:
#   - \b word boundaries so "sitting" doesn't match "sit"
#   - Multi-word forms explicit (\b lie\s+down \b)
#   - Single-word ambiguous triggers (like "up" or "down") are AVOIDED
#     because they false-fire too often; user must say "stand up", not
#     bare "up"
ACTION_PATTERNS = {
    "sit": [
        r"\bsit\b",                  # "sit", "sit down", "please sit"
        r"\bhave a seat\b",
        r"\btake a seat\b",
        r"\bseated\b",
    ],
    "stand": [
        r"\bstand\b",                # "stand", "stand up", "please stand"
        r"\bget up\b",
        r"\brise\b",
        r"\bon your feet\b",
    ],
    "lie_down": [
        r"\blie\s+down\b",
        r"\blay\s+down\b",
        r"\bgo to sleep\b",
        r"\btake a nap\b",
        # Not "rest" alone — too ambiguous ("rest here", "the rest is...")
    ],
    "shake": [
        r"\bshake\b",
        r"\bgive me your paw\b",
        r"\bhandshake\b",
    ],
    "stop": [
        r"\bstop\b",
        r"\bhalt\b",
        r"\bfreeze\b",
        r"\bnever mind\b",
        r"\bcancel\b",
        r"\bwait there\b",           # "wait" alone is ambiguous; "wait there" = stop
        r"\bstay\b",
    ],
}

# Compound-action connectors. If the phrase contains ANY of these, we
# escalate — it's probably a multi-step plan that Opus should handle,
# even if a simple action also matches. Examples:
#   "sit and then lie down"   → escalate (compound)
#   "stand up until I say so" → escalate (conditional)
#   "sit while I get my keys" → escalate (conditional)
# A bare "sit" or "please sit down now" avoids all these keywords.
COMPOUND_KEYWORDS = re.compile(
    r"\b(and then|then|and|but|after|before|until|while|whenever|"
    r"if|unless|when|once|as soon as|followed by)\b",
    re.IGNORECASE,
)

# Negation markers. If the phrase has one of these, we escalate — user
# might be saying "don't sit" or "can't sit down right now". Opus can
# disambiguate.
NEGATION_KEYWORDS = re.compile(
    r"\b(don'?t|do not|can'?t|won'?t|not to|shouldn'?t|stop sitting|"
    r"stop standing|stop shaking)\b",
    re.IGNORECASE,
)

# Filler / conversational qualifiers to strip before matching. Helps
# make the "single unambiguous action" check work for polite phrasing.
# Order matters slightly (longer phrases first) so re-applying doesn't
# leave stragglers.
FILLER_PATTERNS = [
    re.compile(r"\b(please|could you|can you|would you|will you|"
               r"how about you|go ahead and|i want you to|i'd like you to|"
               r"now|right now|just|quickly|real quick|for me|"
               r"buddy|girl|girlie|laira|jarvis|ok|okay|alright|hey)\b",
               re.IGNORECASE),
    re.compile(r"[,.!?;:]"),          # punctuation
    re.compile(r"\s+"),               # runs of whitespace → single space
]


def _strip_fillers(text):
    """Remove conversational filler words + punctuation. Returns a
    clean, lowercased, space-normalized phrase."""
    s = text.lower()
    for pat in FILLER_PATTERNS:
        s = pat.sub(" ", s)
    return s.strip()


def classify(text):
    """Return a DIRECT_ACTIONS name if the phrase unambiguously means
    one of those actions; else None (caller should escalate to Opus).

    Rules, in order:
      1. Empty input → None (nothing to route).
      2. Compound/conditional/negation markers → None (escalate).
      3. Strip filler words. Check each action's regexes against the
         stripped phrase.
      4. If EXACTLY ONE action matches, return it.
      5. If zero or multiple matches → None (escalate).
    """
    if not text or not text.strip():
        return None
    # Bail out early on anything that smells like a compound plan or a
    # negation — Opus handles those properly.
    if COMPOUND_KEYWORDS.search(text):
        return None
    if NEGATION_KEYWORDS.search(text):
        return None
    clean = _strip_fillers(text)
    if not clean:
        return None
    # Collect ALL action matches. If >1, the phrase is talking about
    # multiple actions (rare but possible: "sit down and have a seat"
    # which would have already been caught by COMPOUND but not "have a
    # shake and sit"). Escalate on ambiguity.
    matched = []
    for action, patterns in ACTION_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, clean, re.IGNORECASE):
                matched.append(action)
                break  # one match per action is enough
    if len(matched) == 1:
        return matched[0]
    return None
