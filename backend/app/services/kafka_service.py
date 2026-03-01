# backend/app/services/kafka_service.py
# ─────────────────────────────────────────────────────────────────────────────
# Kafka event publisher for both /api/chat and /api/agent responses.
#
# This is a fire-and-forget thin wrapper — it NEVER raises, so a Kafka outage
# cannot break the API response.  All events appear on two topics:
#
#   chat.events   — emitted after every /api/chat response  (existing)
#   agent.events  — emitted after every /api/agent response  (new)
#
# Both topics share the same JSON schema so the Spark consumer can read them
# with a single union stream or two separate streams.
#
# Topic payload schema (chat.events and agent.events):
# {
#   "event":       "chat.response" | "agent.response",
#   "session_id":  "abc12345",
#   "timestamp":   "2026-03-01T05:30:00Z",
#   "message":     "Where is my order?",
#   "reply":       "Your order #12345 is shipped ...",
#   "model":       "qwen2.5-coder:latest",
#   "latency_ms":  320,
#   "tool_calls":  ["salesforce_get_order"],   <- tool names used
#   "citations":   []
# }
#
# Environment variables (backend/.env):
#   KAFKA_BOOTSTRAP_SERVERS  — default "localhost:9092"
#   KAFKA_TOPIC_CHAT_EVENTS  — default "chat.events"
#   KAFKA_TOPIC_AGENT_EVENTS — default "agent.events"
#   KAFKA_ENABLED            — default "true"
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import json
import structlog
from datetime import datetime, timezone
from typing import Any, Optional

from app.config import settings

log = structlog.get_logger(__name__)

# ── Lazy singleton producer ────────────────────────────────────────────────────
_producer = None


def _get_producer():
    """Return a cached KafkaProducer, creating it on first call. Returns None if disabled."""
    global _producer
    if _producer is not None:
        return _producer

    if not settings.kafka_enabled:
        return None

    try:
        from kafka import KafkaProducer  # type: ignore

        _producer = KafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=3,
            request_timeout_ms=2000,   # don't block the API response
            acks=1,
        )
        log.info("kafka.producer.connected", bootstrap=settings.kafka_bootstrap_servers)
    except Exception as exc:
        log.warning("kafka.producer.unavailable", error=str(exc))
        _producer = None  # stays None → all publish calls are no-ops

    return _producer


def _publish(topic: str, payload: dict[str, Any]) -> None:
    """Internal fire-and-forget publish. Never raises."""
    producer = _get_producer()
    if producer is None:
        return
    try:
        producer.send(topic, value=payload)
        log.debug("kafka.event.sent", topic=topic, session=payload.get("session_id"))
    except Exception as exc:
        log.warning("kafka.event.send_failed", topic=topic, error=str(exc))


# ── Public API ─────────────────────────────────────────────────────────────────

def publish_chat_event(
    *,
    session_id: str,
    message: str,
    reply: str,
    model: Optional[str],
    latency_ms: Optional[int],
    tool_calls: list,          # list of ToolCallResult objects
    citations: list[str],
) -> None:
    """
    Publish a chat.events record after every /api/chat response.
    Called from app.routers.chat.
    """
    _publish(
        topic=settings.kafka_topic_chat_events,
        payload={
            "event":       "chat.response",
            "session_id":  session_id,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "message":     message,
            "reply":       reply,
            "model":       model,
            "latency_ms":  latency_ms,
            "tool_calls":  [tc.tool if hasattr(tc, "tool") else str(tc) for tc in tool_calls],
            "citations":   citations,
        },
    )


def publish_agent_event(
    *,
    session_id: str,
    message: str,
    reply: str,
    model: Optional[str],
    latency_ms: Optional[int],
    tool_steps: list,          # list of ToolStep objects from agent_service
) -> None:
    """
    Publish an agent.events record after every /api/agent response.
    Called from app.routers.agent.
    Includes the full tool step trace (which tools were called and why).
    """
    _publish(
        topic=settings.kafka_topic_agent_events,
        payload={
            "event":       "agent.response",
            "session_id":  session_id,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "message":     message,
            "reply":       reply,
            "model":       model,
            "latency_ms":  latency_ms,
            # Tool names used in this agent run
            "tool_calls":  [step.tool if hasattr(step, "tool") else str(step) for step in tool_steps],
            # Full trace — useful for analytics: what did the agent actually do?
            "tool_trace":  [
                {"tool": s.tool, "input": s.input, "output_preview": s.output[:200]}
                for s in tool_steps
                if hasattr(s, "tool")
            ],
            "citations":   [],
        },
    )
