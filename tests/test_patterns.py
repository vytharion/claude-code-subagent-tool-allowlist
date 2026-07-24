from __future__ import annotations

import pytest

from subagent_allowlist.allowlist import build_scoped_agent
from subagent_allowlist.baseline import (
    BASELINE_AGENT_NAME,
    AgentDefinition,
    build_baseline_agent,
)
from subagent_allowlist.observe import (
    REQUIRED_TOOLS,
    assess_tool_exposure,
)
from subagent_allowlist.patterns import (
    SAFE_BASH_COMMANDS,
    SCOPED_AGENT_TOOLS,
    WORKSPACE_SCOPE_PATTERN,
    PatternedTool,
    build_scoped_bash_commands,
    build_scoped_file_tools,
    parse_tool_name,
    parse_tool_pattern,
    render_patterned_tool,
    validate_patterned_allowlist,
)
from subagent_allowlist.spawn import build_agent_options


def test_patterned_tool_renders_as_name_paren_pattern() -> None:
    assert PatternedTool("Read", "./src/**").render() == "Read(./src/**)"


def test_render_helper_matches_dataclass_render() -> None:
    assert (
        render_patterned_tool("Bash", "pytest")
        == PatternedTool("Bash", "pytest").render()
    )


def test_workspace_scope_targets_the_src_tree() -> None:
    assert WORKSPACE_SCOPE_PATTERN == "./src/**"


def test_safe_bash_commands_are_read_only_and_sorted() -> None:
    assert SAFE_BASH_COMMANDS == (
        "git diff",
        "git log",
        "git status",
        "pytest",
    )
    assert list(SAFE_BASH_COMMANDS) == sorted(SAFE_BASH_COMMANDS)


def test_scoped_file_tools_apply_workspace_scope_to_every_file_tool() -> None:
    tools = build_scoped_file_tools()
    assert tools == (
        f"Glob({WORKSPACE_SCOPE_PATTERN})",
        f"Grep({WORKSPACE_SCOPE_PATTERN})",
        f"Read({WORKSPACE_SCOPE_PATTERN})",
    )


def test_scoped_bash_commands_pin_each_literal_command() -> None:
    bash_tools = build_scoped_bash_commands()
    assert bash_tools == tuple(f"Bash({cmd})" for cmd in SAFE_BASH_COMMANDS)


def test_scoped_agent_tools_alias_the_file_scope_tuple() -> None:
    assert SCOPED_AGENT_TOOLS == build_scoped_file_tools()


def test_parse_tool_name_extracts_the_tool() -> None:
    assert parse_tool_name("Read(./src/**)") == "Read"
    assert parse_tool_name("Bash(git status)") == "Bash"


def test_parse_tool_pattern_extracts_the_pattern() -> None:
    assert parse_tool_pattern("Read(./src/**)") == "./src/**"
    assert parse_tool_pattern("Bash(git status)") == "git status"


def test_parse_falls_back_on_bare_entries() -> None:
    assert parse_tool_name("Read") == "Read"
    assert parse_tool_pattern("Read") == ""


def test_scoped_agent_is_an_agent_definition() -> None:
    assert isinstance(build_scoped_agent(), AgentDefinition)


def test_scoped_agent_reuses_baseline_description_and_prompt() -> None:
    baseline = build_baseline_agent()
    scoped = build_scoped_agent()
    assert scoped.description == baseline.description
    assert scoped.prompt == baseline.prompt
    assert scoped.model == baseline.model


def test_scoped_agent_declares_patterned_tools() -> None:
    scoped = build_scoped_agent()
    assert scoped.tools is not None
    assert tuple(scoped.tools) == SCOPED_AGENT_TOOLS


def test_scoped_agent_options_carry_patterned_tool_strings() -> None:
    options = build_agent_options(build_scoped_agent())
    tools = options["agents"][BASELINE_AGENT_NAME]["tools"]
    assert tools == list(SCOPED_AGENT_TOOLS)
    for entry in tools:
        assert entry.endswith(")")
        assert "(" in entry


def test_scoped_agent_audit_matches_locked_down_shape() -> None:
    report = assess_tool_exposure(build_agent_options(build_scoped_agent()))
    assert report.is_unrestricted is False
    assert report.granted_tools == REQUIRED_TOOLS
    assert report.dangerous_granted == frozenset()
    assert report.unnecessary_granted == frozenset()
    assert report.over_grant_ratio == 0.0


def test_validate_patterned_allowlist_accepts_scoped_agent_tools() -> None:
    validate_patterned_allowlist(SCOPED_AGENT_TOOLS)


def test_validate_patterned_allowlist_accepts_safe_bash_patterns() -> None:
    validate_patterned_allowlist(build_scoped_bash_commands())


def test_validate_patterned_allowlist_rejects_bare_dangerous_tools() -> None:
    with pytest.raises(ValueError, match="dangerous"):
        validate_patterned_allowlist(("Read(./src/**)", "Bash"))


def test_validate_patterned_allowlist_rejects_unknown_tool_names() -> None:
    with pytest.raises(ValueError, match="unknown"):
        validate_patterned_allowlist(("Read(./src/**)", "TeleportToMars(now)"))
