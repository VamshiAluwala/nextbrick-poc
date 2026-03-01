# backend/app/services/chat_memory.py
# ─────────────────────────────────────────────────────────────────────────────
# In-memory chat history keyed by session_id for correlation across prompts.
# Enables the same session_id to get prior turns so the agent has conversation context.
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from typing import List

from app.config import settings
from app.models.chat import MessageItem

# session_id -> list of MessageItem (role, content); trimmed to max_messages
_store: dict[str, list[MessageItem]] = {}


def get_history(session_id: str) -> List[MessageItem]:
    """Return the last N messages for this session (oldest to newest)."""
    if not session_id:
        return []
    items = _store.get(session_id, [])
    return list(items)


def append_turn(session_id: str, user_content: str, assistant_content: str) -> None:
    """Append one user/assistant turn and trim to max_messages per session."""
    if not session_id:
        return
    max_n = max(1, getattr(settings, "chat_memory_max_messages", 20))
    if session_id not in _store:
        _store[session_id] = []
    _store[session_id].append(MessageItem(role="user", content=user_content))
    _store[session_id].append(MessageItem(role="assistant", content=assistant_content or ""))
    # Keep only the last max_n messages
    _store[session_id] = _store[session_id][-max_n:]
