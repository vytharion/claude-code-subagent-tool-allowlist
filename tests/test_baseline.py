from __future__ import annotations

from subagent_allowlist.baseline import (
    BASELINE_AGENT_NAME,
    AgentDefinition,
    build_baseline_agent,
)


def test_baseline_agent_name_is_stable() -> None:
    assert BASELINE_AGENT_NAME == "file-explorer"


def test_baseline_agent_has_description_and_prompt() -> None:
    agent = build_baseline_agent()
    assert isinstance(agent, AgentDefinition)
    assert agent.description.strip()
    assert agent.prompt.strip()


def test_baseline_agent_starts_unrestricted() -> None:
    # step 1 baseline inherits every tool; the allowlist arrives in step 3
    agent = build_baseline_agent()
    assert agent.tools is None


def test_baseline_agent_model_defaults_to_sonnet() -> None:
    agent = build_baseline_agent()
    assert agent.model == "sonnet"
