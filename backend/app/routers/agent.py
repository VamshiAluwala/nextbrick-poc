# backend/app/routers/agent.py
# ─────────────────────────────────────────────────────────────────────────────
# POST /api/agent — LangChain ReAct agent endpoint.
#
# This is the new agentic endpoint that routes user messages through the
# full orchestrated agent (Salesforce + Confluence + Elasticsearch tools).
# The existing POST /api/chat endpoint is untouched.
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import uuid
import structlog

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional

from app.models.chat import MessageItem
from app.services.agent_service import invoke_agent, ToolStep
from app.services.kafka_service import publish_agent_event
from app.middleware.metrics import metrics_store

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/api", tags=["agent"])


# ── Request / Response schemas ────────────────────────────────────────────────

class AgentRequest(BaseModel):
    message: str = Field(..., description="The user's message or question")
    history: List[MessageItem] = Field(default_factory=list, description="Previous conversation turns")
    session_id: Optional[str] = Field(None, description="Optional session identifier for logging")


class ToolStepResponse(BaseModel):
    tool: str
    input: dict
    output: str


class AgentResponse(BaseModel):
    reply: str
    tool_steps: List[ToolStepResponse]
    latency_ms: Optional[int]
    model: str
    session_id: str


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/agent",
    response_model=AgentResponse,
    status_code=status.HTTP_200_OK,
    summary="Orchestrated ReAct Agent",
    description=(
        "Runs the LangChain ReAct agent against the user message. "
        "The agent autonomously selects and calls the right tools "
        "(Salesforce, Confluence, Elasticsearch) and returns a synthesised answer."
    ),
)
def agent(req: AgentRequest) -> AgentResponse:
    """
    Process a user message through the orchestrated ReAct agent:
    1. Select which tool(s) to call based on reasoning
    2. Execute tool calls (Salesforce / Confluence / Elasticsearch)
    3. Synthesise a final answer from tool outputs
    4. Return the answer + full trace of tool steps taken
    """
    session_id = req.session_id or str(uuid.uuid4())[:8]
    bound_log = log.bind(session_id=session_id, preview=req.message[:60])
    bound_log.info("agent.request")

    try:
        result = invoke_agent(
            message=req.message,
            history=req.history,
            session_id=session_id,
        )
    except Exception as exc:
        bound_log.exception("agent.request.error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Agent error: {exc}",
        ) from exc

    # Record metrics (reuse existing store)
    if result.latency_ms is not None:
        metrics_store.record(
            latency_ms=result.latency_ms,
            tool_calls=len(result.tool_steps),
        )

    # Publish to Kafka (fire-and-forget)
    publish_agent_event(
        session_id=session_id,
        message=req.message,
        reply=result.reply,
        model=result.model,
        latency_ms=result.latency_ms,
        tool_steps=result.tool_steps,
    )

    bound_log.info(
        "agent.response",
        latency_ms=result.latency_ms,
        tool_steps=len(result.tool_steps),
    )

    return AgentResponse(
        reply=result.reply,
        tool_steps=[
            ToolStepResponse(tool=s.tool, input=s.input, output=s.output)
            for s in result.tool_steps
        ],
        latency_ms=result.latency_ms,
        model=result.model,
        session_id=session_id,
    )
