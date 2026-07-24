from __future__ import annotations

from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from .allowlist import build_locked_down_agent
from .baseline import BASELINE_AGENT_NAME, AgentDefinition, build_baseline_agent
from .denylist import guard_message

QueryFn = Callable[..., AsyncIterator[Any]]


def build_agent_options(
    agent: AgentDefinition,
    name: str = BASELINE_AGENT_NAME,
) -> Dict[str, Any]:
    definition: Dict[str, Any] = {
        "description": agent.description,
        "prompt": agent.prompt,
        "model": agent.model,
    }
    if agent.tools is not None:
        definition["tools"] = list(agent.tools)
    return {"agents": {name: definition}}


async def run_baseline_subagent(
    user_prompt: str,
    query_fn: QueryFn,
) -> List[Any]:
    agent = build_baseline_agent()
    options = build_agent_options(agent)
    messages: List[Any] = []
    async for message in query_fn(prompt=user_prompt, options=options):
        messages.append(message)
    return messages


async def run_guarded_subagent(
    user_prompt: str,
    query_fn: QueryFn,
    agent: Optional[AgentDefinition] = None,
) -> List[Any]:
    active_agent = agent if agent is not None else build_locked_down_agent()
    options = build_agent_options(active_agent)
    messages: List[Any] = []
    async for message in query_fn(prompt=user_prompt, options=options):
        guard_message(message)
        messages.append(message)
    return messages
