from __future__ import annotations

from typing import Tuple

from .baseline import AgentDefinition, build_baseline_agent
from .observe import DANGEROUS_TOOLS, KNOWN_TOOLS, REQUIRED_TOOLS
from .patterns import SCOPED_AGENT_TOOLS

# The explicit allowlist a file-explorer subagent needs and nothing more.
# Ordered so serialized configs diff cleanly across runs.
LOCKED_DOWN_AGENT_TOOLS: Tuple[str, ...] = tuple(sorted(REQUIRED_TOOLS))


def build_locked_down_agent() -> AgentDefinition:
    baseline = build_baseline_agent()
    return AgentDefinition(
        description=baseline.description,
        prompt=baseline.prompt,
        tools=LOCKED_DOWN_AGENT_TOOLS,
        model=baseline.model,
    )


def build_scoped_agent() -> AgentDefinition:
    baseline = build_baseline_agent()
    return AgentDefinition(
        description=baseline.description,
        prompt=baseline.prompt,
        tools=SCOPED_AGENT_TOOLS,
        model=baseline.model,
    )


def validate_allowlist(tools: Tuple[str, ...]) -> None:
    unknown = set(tools) - KNOWN_TOOLS
    if unknown:
        raise ValueError(f"unknown tools in allowlist: {sorted(unknown)}")
    dangerous = set(tools) & DANGEROUS_TOOLS
    if dangerous:
        raise ValueError(
            f"dangerous tools present in allowlist: {sorted(dangerous)}"
        )
