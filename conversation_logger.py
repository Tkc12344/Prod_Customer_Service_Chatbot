"""
conversation_logger.py
Persists closed live-agent sessions to live_conversations.csv so they can
be used for retraining and policy expansion.

Auto-retrain: after every session is logged, checks if the total number of
new conversation pairs has reached AUTO_RETRAIN_THRESHOLD. If so, triggers
a background retrain automatically — no manual intervention needed.
"""

import csv
import os
import threading
import logging
from datetime import datetime

log = logging.getLogger(__name__)

LIVE_CONV_CSV = os.getenv('LIVE_CONV_CSV', 'live_conversations.csv')

_FIELDNAMES = [
    'session_id', 'timestamp', 'turn', 'speaker',
    'utterance', 'response',
    'channel', 'duration_s',
]

_lock = threading.Lock()

# ── Auto-retrain state ────────────────────────────────────────────────────────
_AUTO_RETRAIN_THRESHOLD = int(os.getenv('AUTO_RETRAIN_THRESHOLD', '10'))
_AUTO_RETRAIN_ENABLED   = os.getenv('AUTO_RETRAIN_ENABLED', 'true').lower() == 'true'
_last_retrain_count     = 0
_retrain_in_progress    = False


def _count_total_pairs() -> int:
    """Count total conversation pairs logged in the CSV (excluding header)."""
    if not os.path.exists(LIVE_CONV_CSV):
        return 0
    try:
        with open(LIVE_CONV_CSV, 'r', encoding='utf-8') as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return 0


def _check_auto_retrain():
    """
    Check if enough new pairs have been logged to trigger an automatic retrain.
    Fires in a background daemon thread so it never blocks the main request.
    """
    global _last_retrain_count, _retrain_in_progress

    if not _AUTO_RETRAIN_ENABLED or _retrain_in_progress:
        return

    total          = _count_total_pairs()
    new_since_last = total - _last_retrain_count

    if new_since_last < _AUTO_RETRAIN_THRESHOLD:
        return

    # Snapshot the count before launching the thread
    _retrain_in_progress = True
    _last_retrain_count  = total
    captured_new         = new_since_last

    def _run():
        global _retrain_in_progress
        try:
            log.info(
                f"[Auto-retrain] Triggered — {captured_new} new pairs "
                f"(threshold: {_AUTO_RETRAIN_THRESHOLD})"
            )
            from retrain_from_conversations import run_retrain
            result = run_retrain(force=False)

            if result['status'] == 'ok':
                log.info(
                    f"[Auto-retrain] Complete — "
                    f"{result.get('new_rows_appended', 0)} rows added, "
                    f"model: {result.get('new_model', 'N/A')}, "
                    f"policies expanded: {result.get('policies_expanded', [])}"
                )
            else:
                log.warning(
                    f"[Auto-retrain] {result['status']}: "
                    f"{result.get('reason', result.get('error', 'unknown'))}"
                )
        except Exception as e:
            log.error(f"[Auto-retrain] Failed: {e}", exc_info=True)
        finally:
            _retrain_in_progress = False

    threading.Thread(target=_run, daemon=True, name="auto-retrain").start()


def _ensure_header():
    """Create the CSV with headers if it doesn't exist yet."""
    if not os.path.exists(LIVE_CONV_CSV):
        with open(LIVE_CONV_CSV, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=_FIELDNAMES).writeheader()


def log_session(
    session_id: str,
    history: list,
    started: datetime,
    channel: str = 'live_chat',
) -> int:
    """
    Persist a completed session's conversation pairs to CSV.

    history items: {'from': 'user'|'agent', 'text': str, 'time': str}

    Pairs consecutive user→agent turns. After writing, checks whether
    the auto-retrain threshold has been reached and fires if so.

    Returns the number of pairs logged.
    """
    if not history:
        return 0

    _ensure_header()

    duration = int((datetime.now() - started).total_seconds())
    pairs    = _extract_pairs(history)
    if not pairs:
        return 0

    rows = [
        {
            'session_id': session_id,
            'timestamp':  datetime.now().isoformat(),
            'turn':       idx,
            'speaker':    'user',
            'utterance':  user_msg,
            'response':   agent_msg,
            'channel':    channel,
            'duration_s': duration,
        }
        for idx, (user_msg, agent_msg) in enumerate(pairs, 1)
    ]

    with _lock:
        with open(LIVE_CONV_CSV, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=_FIELDNAMES).writerows(rows)

    log.info(f"[Logger] Session {session_id} — {len(rows)} pairs logged to {LIVE_CONV_CSV}")

    # Trigger auto-retrain check after every session
    _check_auto_retrain()

    return len(rows)


def _extract_pairs(history: list) -> list:
    """
    Walk the history and pair each user message with the next agent reply.
    Returns list of (user_text, agent_text) tuples.
    """
    pairs = []
    i = 0
    while i < len(history):
        msg = history[i]
        if msg.get('from') == 'user':
            user_text  = msg.get('text', '').strip()
            agent_text = ''
            for j in range(i + 1, len(history)):
                if history[j].get('from') == 'agent':
                    agent_text = history[j].get('text', '').strip()
                    i = j
                    break
            if user_text:
                pairs.append((user_text, agent_text))
        i += 1
    return pairs


def load_conversations(filepath: str = None) -> list:
    """Load all logged conversations as a list of dicts. Returns [] if file missing."""
    path = filepath or LIVE_CONV_CSV
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))
