from __future__ import annotations

from subagent_allowlist.baseline import BASELINE_AGENT_NAME, build_baseline_agent
from subagent_allowlist.observe import (
    DANGEROUS_TOOLS,
    KNOWN_TOOLS,
    REQUIRED_TOOLS,
    assess_tool_exposure,
    summarize_report,
)
from subagent_allowlist.spawn import build_agent_options


def _baseline_options() -> dict:
    return build_agent_options(build_baseline_agent())


def test_required_tools_are_a_subset_of_known() -> None:
    assert REQUIRED_TOOLS.issubset(KNOWN_TOOLS)


def test_dangerous_tools_are_a_subset_of_known() -> None:
    assert DANGEROUS_TOOLS.issubset(KNOWN_TOOLS)


def test_required_and_dangerous_do_not_overlap() -> None:
    # a read/glob/grep explorer must not need Bash/Write/Edit to do its job
    assert REQUIRED_TOOLS.isdisjoint(DANGEROUS_TOOLS)


def test_baseline_report_flags_unrestricted_access() -> None:
    report = assess_tool_exposure(_baseline_options())
    assert report.is_unrestricted is True
    assert report.granted_tools == KNOWN_TOOLS


def test_baseline_grants_every_dangerous_tool() -> None:
    report = assess_tool_exposure(_baseline_options())
    assert report.dangerous_granted == DANGEROUS_TOOLS
    for tool in ("Bash", "Write", "Edit", "WebFetch"):
        assert tool in report.dangerous_granted


def test_baseline_grants_far_more_than_the_task_requires() -> None:
    report = assess_tool_exposure(_baseline_options())
    unnecessary = report.unnecessary_granted
    assert unnecessary == KNOWN_TOOLS - REQUIRED_TOOLS
    # over-grant ratio should be well above half for the unrestricted baseline
    assert report.over_grant_ratio > 0.5


def test_report_accepts_a_custom_agent_name() -> None:
    options = {
        "agents": {
            "custom": {
                "description": "d",
                "prompt": "p",
                "model": "sonnet",
            }
        }
    }
    report = assess_tool_exposure(options, agent_name="custom")
    assert report.is_unrestricted is True
    assert report.granted_tools == KNOWN_TOOLS


def test_report_recognizes_a_hypothetical_allowlist() -> None:
    # sanity-check the reporter itself: when tools ARE declared, it must
    # treat that list as the granted set — this is the shape step 3 uses
    options = {
        "agents": {
            BASELINE_AGENT_NAME: {
                "description": "d",
                "prompt": "p",
                "model": "sonnet",
                "tools": ["Read", "Glob", "Grep"],
            }
        }
    }
    report = assess_tool_exposure(options)
    assert report.is_unrestricted is False
    assert report.granted_tools == REQUIRED_TOOLS
    assert report.dangerous_granted == frozenset()
    assert report.unnecessary_granted == frozenset()
    assert report.over_grant_ratio == 0.0


def test_summarize_report_returns_readable_lines() -> None:
    report = assess_tool_exposure(_baseline_options())
    lines = summarize_report(report)
    joined = "\n".join(lines)
    assert "unrestricted: True" in joined
    assert "dangerous_granted:" in joined
    assert "over_grant_ratio:" in joined
