from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

BASELINE_AGENT_NAME = "file-explorer"


@dataclass(frozen=True)
class AgentDefinition:
    description: str
    prompt: str
    tools: Optional[Sequence[str]] = None
    model: str = "sonnet"


def build_baseline_agent() -> AgentDefinition:
    return AgentDefinition(
        description="Explores a directory and reports what it finds",
        prompt=(
            "You are a file explorer subagent. Given a directory path, "
            "list its files, read the interesting ones, and return a short "
            "summary of what the project appears to do."
        ),
    )
