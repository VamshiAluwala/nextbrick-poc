# backend/app/services/agent_service.py
# ─────────────────────────────────────────────────────────────────────────────
# Orchestrated LangChain ReAct Agent.
#
# This is the core of the agentic architecture. It:
#   1. Binds ALL_TOOLS (Salesforce, Confluence, Elasticsearch) to the LLM
#   2. Creates a ReAct agent that reasons about which tool to call
#   3. Executes tool calls and feeds results back to the LLM
#   4. Returns both the final answer and a log of every tool step
#
# The agent uses the same LLM as the existing chat service, configured via .env.
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import time
import structlog
from typing import List, Optional
from dataclasses import dataclass, field

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.agents import AgentAction, AgentFinish

from app.config import settings
from app.models.chat import MessageItem
from app.tools import ALL_TOOLS

log = structlog.get_logger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """\
You are an enterprise AI assistant for Nextbrick. You have access to three categories of tools:

SALESFORCE TOOLS:
- salesforce_get_case: Look up a support case by ID
- salesforce_create_case: Open a new support case
- salesforce_get_order: Look up an order by ID

CONFLUENCE TOOLS:
- confluence_search: Search the internal knowledge base for guides, policies, how-tos
- confluence_get_page: Retrieve a specific Confluence page by ID

ELASTICSEARCH TOOLS:
- elasticsearch_semantic_search: Semantic search across all indexed documents (use as default fallback)
- elasticsearch_ingest_document: Add a new document to the knowledge base

INSTRUCTIONS:
- Always use the most specific tool for the task. If the user mentions an order, use salesforce_get_order.
- Chain tools when needed: search Confluence first, then get_page for the full content.
- Cite your sources in the final answer (tool name + key field like case ID or page title).
- If no specific tool applies, use elasticsearch_semantic_search to find relevant context.
- Be concise and professional. Support English, German, Spanish, and Chinese.
"""


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

def _build_agent_executor():
    """
    Build a LangChain AgentExecutor with ReAct prompting.
    Returns None if no LLM is configured.
    """
    from app.services.llm_service import build_llm
    llm = build_llm()
    if llm is None:
        return None

    from langchain.agents import create_react_agent, AgentExecutor
    from langchain.prompts import PromptTemplate

    # ReAct prompt template — LangChain's standard format
    react_template = """\
{system_prompt}

You have access to the following tools:
{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (as a JSON object)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        partial_variables={"system_prompt": AGENT_SYSTEM_PROMPT},
        template=react_template,
    )

    agent = create_react_agent(llm=llm, tools=ALL_TOOLS, prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        max_iterations=6,
        verbose=False,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    return executor


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

    executor = _build_agent_executor()

    # ── Demo mode: no LLM configured ─────────────────────────────────────────
    if executor is None:
        bound_log.warning("agent.demo_mode")
        return AgentResult(
            reply=(
                "[Demo mode — model URL not configured]\n"
                f"I would run the ReAct agent with all tools for: \"{message}\".\n"
                "Available tools: salesforce_get_case, salesforce_create_case, "
                "salesforce_get_order, confluence_search, confluence_get_page, "
                "elasticsearch_semantic_search, elasticsearch_ingest_document."
            ),
            tool_steps=[],
            latency_ms=None,
            model=settings.effective_model_name,
            session_id=session_id,
        )

    # ── Live agent invocation ─────────────────────────────────────────────────
    try:
        # Build conversation context string from history (last 4 turns)
        context_lines = []
        for item in history[-4:]:
            prefix = "User" if item.role == "user" else "Assistant"
            context_lines.append(f"{prefix}: {item.content}")
        context = "\n".join(context_lines)
        full_input = f"{context}\nUser: {message}" if context else message

        result = executor.invoke({"input": full_input})

        # Extract intermediate tool steps
        tool_steps: List[ToolStep] = []
        for action, observation in result.get("intermediate_steps", []):
            if isinstance(action, AgentAction):
                tool_steps.append(
                    ToolStep(
                        tool=action.tool,
                        input=action.tool_input if isinstance(action.tool_input, dict) else {"input": str(action.tool_input)},
                        output=str(observation)[:500],  # truncate long outputs
                    )
                )

        reply = result.get("output", "No response generated.")
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
        latency_ms = int((time.perf_counter() - start) * 1000)
        bound_log.exception("agent.invoke.error", error=str(exc))
        raise
