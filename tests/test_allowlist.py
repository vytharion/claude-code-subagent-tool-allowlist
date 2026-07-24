from __future__ import annotations

import pytest

from subagent_allowlist.allowlist import (
    LOCKED_DOWN_AGENT_TOOLS,
    build_locked_down_agent,
    validate_allowlist,
)
from subagent_allowlist.baseline import (
    BASELINE_AGENT_NAME,
    AgentDefinition,
    build_baseline_agent,
)
from subagent_allowlist.observe import (
    DANGEROUS_TOOLS,
    REQUIRED_TOOLS,
    assess_tool_exposure,
)
from subagent_allowlist.spawn import build_agent_options


def test_locked_down_tools_matches_required_set() -> None:
    assert frozenset(LOCKED_DOWN_AGENT_TOOLS) == REQUIRED_TOOLS


def test_locked_down_tools_are_sorted_and_unique() -> None:
    tools = list(LOCKED_DOWN_AGENT_TOOLS)
    assert tools == sorted(tools)
    assert len(tools) == len(set(tools))


def test_locked_down_tools_contain_no_dangerous_tools() -> None:
    assert set(LOCKED_DOWN_AGENT_TOOLS).isdisjoint(DANGEROUS_TOOLS)


def test_locked_down_agent_is_an_agent_definition() -> None:
    agent = build_locked_down_agent()
    assert isinstance(agent, AgentDefinition)


def test_locked_down_agent_declares_an_explicit_tools_list() -> None:
    agent = build_locked_down_agent()
    assert agent.tools is not None
    assert tuple(agent.tools) == LOCKED_DOWN_AGENT_TOOLS


def test_locked_down_agent_reuses_baseline_description_and_prompt() -> None:
    baseline = build_baseline_agent()
    locked = build_locked_down_agent()
    assert locked.description == baseline.description
    assert locked.prompt == baseline.prompt
    assert locked.model == baseline.model


def test_options_carry_the_tools_key_when_locked_down() -> None:
    options = build_agent_options(build_locked_down_agent())
    definition = options["agents"][BASELINE_AGENT_NAME]
    assert "tools" in definition
    assert definition["tools"] == list(LOCKED_DOWN_AGENT_TOOLS)


def test_report_flips_from_unrestricted_to_restricted() -> None:
    baseline_report = assess_tool_exposure(build_agent_options(build_baseline_agent()))
    locked_report = assess_tool_exposure(build_agent_options(build_locked_down_agent()))
    assert baseline_report.is_unrestricted is True
    assert locked_report.is_unrestricted is False


def test_locked_down_report_grants_no_dangerous_tools() -> None:
    report = assess_tool_exposure(build_agent_options(build_locked_down_agent()))
    assert report.dangerous_granted == frozenset()


def test_locked_down_report_grants_nothing_unnecessary() -> None:
    report = assess_tool_exposure(build_agent_options(build_locked_down_agent()))
    assert report.unnecessary_granted == frozenset()
    assert report.over_grant_ratio == 0.0


def test_locked_down_report_grants_exactly_the_required_tools() -> None:
    report = assess_tool_exposure(build_agent_options(build_locked_down_agent()))
    assert report.granted_tools == REQUIRED_TOOLS


def test_validate_allowlist_accepts_the_shipped_allowlist() -> None:
    validate_allowlist(LOCKED_DOWN_AGENT_TOOLS)


def test_validate_allowlist_rejects_dangerous_tools() -> None:
    with pytest.raises(ValueError, match="dangerous"):
        validate_allowlist(("Read", "Bash"))


def test_validate_allowlist_rejects_unknown_tools() -> None:
    with pytest.raises(ValueError, match="unknown"):
        validate_allowlist(("Read", "TeleportToMars"))
