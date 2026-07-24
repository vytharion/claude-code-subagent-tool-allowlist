from __future__ import annotations

from typing import Any, AsyncIterator, Callable, Dict, List

from .baseline import BASELINE_AGENT_NAME, AgentDefinition, build_baseline_agent

QueryFn = Callable[..., AsyncIterator[Any]]


def build_agent_options(agent: AgentDefinition) -> Dict[str, Any]:
    definition: Dict[str, Any] = {
        "description": agent.description,
        "prompt": agent.prompt,
        "model": agent.model,
    }
    if agent.tools is not None:
        definition["tools"] = list(agent.tools)
    return {"agents": {BASELINE_AGENT_NAME: definition}}


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
