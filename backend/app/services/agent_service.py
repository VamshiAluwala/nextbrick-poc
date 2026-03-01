# backend/app/services/agent_service.py
# ─────────────────────────────────────────────────────────────────────────────
# Single agent service: imports ALL_TOOLS from app.tools, builds system prompt
# from tool names/descriptions (no hardcoded keywords), and runs create_agent.
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import time
import structlog
from typing import List, Optional
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from app.config import settings
from app.models.chat import MessageItem
from app.tools import ALL_TOOLS

log = structlog.get_logger(__name__)

# ── System prompt (built from tools; no hardcoded tool names) ─────────────────

def _build_system_prompt(tools: list) -> str:
    """Build system prompt from the list of tools: intro + tool list from descriptions."""
    app_name = getattr(settings, "app_name", "AI Assistant")
    intro = f"You are an enterprise AI assistant for {app_name}. You have access to the following tools:\n\n"
    tool_lines = []
    for t in tools:
        name = getattr(t, "name", str(t))
        desc = getattr(t, "description", "") or ""
        tool_lines.append(f"- {name}: {desc.strip()}")
    instructions = """
INSTRUCTIONS:
- You MUST use the tools to answer. Do NOT give generic refusals, apologies, or made-up reasons (e.g. "I cannot connect", "I am unable to access the system", "try again later"). Always call the relevant tool first.
- For orders: use salesforce_get_all_orders to list orders, or salesforce_get_order for a specific order ID. For cases: use salesforce_get_case or salesforce_create_case. For custom data: use salesforce_query with SOQL.
- When creating a case: the user must provide both subject and description. If either is missing, do NOT call salesforce_create_case yet — ask the user to provide the subject and description; once they reply with both, then call the tool to create the case.
- For documentation or product questions: use elasticsearch_ollama_semantic_search to get content from the indexed docs; base your answer ONLY on the retrieved content.
- Only report errors when the tool itself returns an error. Do not refuse to call a tool or invent connectivity issues.
- Use the most specific tool for each task. Chain tools when needed (e.g. search then get details).

RESPONSE FORMAT — keep answers clean and user-facing only:
- Do NOT include in your reply: source citations, "Source: ...", file names, file paths, internal paths, "How to access", "Location (internal path)", or any tool/metadata attribution. The user should see only the actual answer.
- Provide data from the tools (embedding/index or API) only: summarize or quote the retrieved content clearly and concisely. No extra boilerplate about where the data came from.
- Be concise and professional. Give a clear, direct response with the information the user asked for.
"""
    return intro + "\n".join(tool_lines) + instructions


# ── Response dataclass ─────────────────────────────────────────────────────────

@dataclass
class ToolStep:
    tool: str
    input: dict
    output: str


@dataclass
class AgentResult:
    reply: str
    tool_steps: List[ToolStep] = field(default_factory=list)
    latency_ms: Optional[int] = None
    model: str = ""
    session_id: str = ""


# ── Agent factory ──────────────────────────────────────────────────────────────
# Uses create_agent with wrap_model_call middleware for dynamic model selection.
# Tools come only from app.tools.ALL_TOOLS (no separate tool service).

def _build_agent():
    """
    Build a LangChain agent via create_agent with dynamic model selection middleware.
    Basic model for short conversations; advanced (cloud) model when message count > threshold.
    Tools = ALL_TOOLS from app.tools. Returns None if no LLM is configured.
    """
    from app.services.llm_service import build_llm
    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

    basic_model = build_llm(profile="default")
    if basic_model is None:
        return None

    advanced_model = build_llm(profile="advanced") or basic_model
    message_count_threshold = getattr(settings, "agent_advanced_message_threshold", 10)

    @wrap_model_call
    def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
        message_count = len(request.state.get("messages", []))
        if message_count > message_count_threshold:
            model = advanced_model
        else:
            model = basic_model
        return handler(request.override(model=model))

    system_prompt = _build_system_prompt(ALL_TOOLS)
    return create_agent(
        model=basic_model,
        tools=ALL_TOOLS,
        system_prompt=system_prompt,
        middleware=[dynamic_model_selection],
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def invoke_agent(
    message: str,
    history: List[MessageItem],
    session_id: str = "",
) -> AgentResult:
    """
    Run the orchestrated ReAct agent on a user message.

    Falls back to a keyword-based demo response if no LLM is configured
    (preserves the existing demo-mode behaviour from the chat service).
    """
    bound_log = log.bind(session_id=session_id, preview=message[:60])
    bound_log.info("agent.invoke.start")
    start = time.perf_counter()

    agent = _build_agent()

    # ── Demo mode: no LLM configured ─────────────────────────────────────────
    if agent is None:
        bound_log.warning("agent.demo_mode")
        tool_names = [getattr(t, "name", str(t)) for t in ALL_TOOLS]
        return AgentResult(
            reply=(
                "[Demo mode — model URL not configured]\n"
                f"I would run the agent with all tools for: \"{message}\".\n"
                f"Available tools: {', '.join(tool_names)}."
            ),
            tool_steps=[],
            latency_ms=None,
            model=settings.effective_model_name,
            session_id=session_id,
        )

    # ── Build messages from history + new user message ────────────────────────
    # Use last N messages so in-memory session history correlates prompts (id-based)
    max_history = getattr(settings, "chat_memory_max_messages", 20)
    messages: List = []
    for item in history[-max_history:]:
        if item.role == "user":
            messages.append(HumanMessage(content=item.content))
        else:
            messages.append(AIMessage(content=item.content))
    messages.append(HumanMessage(content=message))

    # ── Live agent invocation ─────────────────────────────────────────────────
    try:
        result = agent.invoke({"messages": messages})

        # Extract final reply and tool steps from state messages
        out_messages = result.get("messages", [])
        reply = "No response generated."
        tool_steps: List[ToolStep] = []
        tool_call_names: dict = {}  # tool_call_id -> tool name

        for msg in out_messages:
            if isinstance(msg, AIMessage):
                if msg.content:
                    if isinstance(msg.content, str):
                        reply = msg.content
                    elif isinstance(msg.content, list) and msg.content:
                        part = msg.content[0]
                        reply = part.get("text", str(part)) if isinstance(part, dict) else str(part)
                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        tool_call_names[tc.get("id", "")] = tc.get("name", "unknown")
            elif isinstance(msg, ToolMessage):
                name = tool_call_names.get(getattr(msg, "tool_call_id", ""), "unknown")
                content = msg.content if isinstance(msg.content, str) else str(msg.content)[:500]
                tool_steps.append(ToolStep(tool=name, input={}, output=content))

        latency_ms = int((time.perf_counter() - start) * 1000)
        bound_log.info("agent.invoke.done", latency_ms=latency_ms, tool_steps=len(tool_steps))

        return AgentResult(
            reply=reply,
            tool_steps=tool_steps,
            latency_ms=latency_ms,
            model=settings.effective_model_name,
            session_id=session_id,
        )

    except Exception as exc:
        bound_log.exception("agent.invoke.error", error=str(exc))
        raise
