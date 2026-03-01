# backend/app/routers/chat.py
# ─────────────────────────────────────────────────────────────────────────────
# POST /api/chat — main agentic response endpoint.
# Routes through the ReAct agent so retrieval (e.g. next_elastic_test1) and other
# tools are used to extract data and produce the reply.
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import uuid
import structlog

from fastapi import APIRouter, HTTPException, status
from app.config import settings
from app.models.chat import ChatRequest, ChatResponse, ToolCallResult
from app.services.agent_service import invoke_agent
from app.services.chat_memory import get_history as get_memory_history, append_turn as memory_append_turn
from app.services.kafka_service import publish_chat_event
from app.middleware.metrics import metrics_store

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


def _tool_steps_to_citations_and_calls(tool_steps):
    """Build citations and tool_calls for ChatResponse from agent tool_steps.
    Deduplicates by tool name so the UI shows one badge per tool used, not per invocation.
    """
    citations = list({step.tool for step in tool_steps})
    seen: set = set()
    tool_calls = []
    for step in tool_steps:
        if step.tool in seen:
            continue
        seen.add(step.tool)
        count = sum(1 for s in tool_steps if s.tool == step.tool)
        detail = str(step.output)[:200] if step.output else ""
        if count > 1:
            detail = f"used {count}×" + (f" — {detail}" if detail else "")
        tool_calls.append(
            ToolCallResult(tool=step.tool, status="done", detail=detail or "")
        )
    return citations, tool_calls


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def chat(req: ChatRequest) -> ChatResponse:
    """
    Process a user message through the ReAct agent:
    Agent can call elasticsearch_ollama_semantic_search (next_elastic_test1), Confluence,
    Salesforce, etc., and use retrieved data in the reply.
    """
    session_id = req.session_id or str(uuid.uuid4())[:8]
    bound_log = log.bind(session_id=session_id, message_preview=req.message[:50])
    bound_log.info("chat.request")

    # Use in-memory history by session_id so prompts correlate across the conversation
    history = get_memory_history(session_id)
    if not history and req.history:
        history = req.history  # fallback if no stored history yet (e.g. first message)

    model_name = settings.effective_model_name

    try:
        result = invoke_agent(
            message=req.message,
            history=history,
            session_id=session_id,
        )
    except Exception as exc:
        bound_log.exception("chat.agent_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Agent error: {exc}",
        ) from exc

    citations, tool_calls = _tool_steps_to_citations_and_calls(result.tool_steps)
    bound_log.info(
        "chat.response",
        latency_ms=result.latency_ms,
        tools=len(result.tool_steps),
    )

    if result.latency_ms is not None:
        metrics_store.record(
            latency_ms=result.latency_ms,
            tool_calls=len(result.tool_steps),
        )

    # Persist this turn to in-memory history for correlation on next request with same session_id
    memory_append_turn(session_id, req.message, result.reply or "")

    publish_chat_event(
        session_id=session_id,
        message=req.message,
        reply=result.reply or "",
        model=result.model or model_name,
        latency_ms=result.latency_ms,
        tool_calls=tool_calls,
        citations=citations,
    )

    return ChatResponse(
        reply=result.reply or "No response from agent.",
        citations=citations,
        tool_calls=tool_calls,
        latency_ms=result.latency_ms,
        model=result.model or model_name,
        session_id=session_id,
    )
