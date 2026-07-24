from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List

import pytest

from subagent_allowlist.allowlist import LOCKED_DOWN_AGENT_TOOLS
from subagent_allowlist.baseline import BASELINE_AGENT_NAME
from subagent_allowlist.denylist import DeniedInvocationError
from subagent_allowlist.observe import (
    DANGEROUS_TOOLS,
    REQUIRED_TOOLS,
    assess_tool_exposure,
)
from subagent_allowlist.patterns import (
    SCOPED_AGENT_TOOLS,
    parse_tool_name,
    parse_tool_pattern,
)
from subagent_allowlist.template import (
    DEFAULT_FILE_TOOLS,
    RECIPE_SCHEMA_VERSION,
    SUBAGENT_RECIPES,
    SubagentTemplate,
    docs_reviewer_recipe,
    git_history_auditor_recipe,
    list_recipe_names,
    load_recipe,
    read_only_explorer_recipe,
    run_template_subagent,
)


def _minimal_template(**overrides: Any) -> SubagentTemplate:
    kwargs: Dict[str, Any] = {
        "name": "explorer",
        "description": "d",
        "prompt": "p",
        "allowed_paths": ("./src/**",),
    }
    kwargs.update(overrides)
    return SubagentTemplate(**kwargs)


# --- Construction + validation ---------------------------------------


def test_template_defaults_to_glob_grep_read_file_tools() -> None:
    tpl = _minimal_template()
    assert tpl.file_tools == DEFAULT_FILE_TOOLS
    assert tpl.model == "sonnet"


def test_template_is_frozen() -> None:
    tpl = _minimal_template()
    with pytest.raises(Exception):
        tpl.name = "other"  # type: ignore[misc]


def test_template_rejects_blank_name() -> None:
    with pytest.raises(ValueError, match="name"):
        SubagentTemplate(name="", description="d", prompt="p")


def test_template_rejects_blank_description() -> None:
    with pytest.raises(ValueError, match="description"):
        SubagentTemplate(name="n", description="   ", prompt="p")


def test_template_rejects_blank_prompt() -> None:
    with pytest.raises(ValueError, match="prompt"):
        SubagentTemplate(name="n", description="d", prompt="")


def test_template_rejects_denied_path_at_construction() -> None:
    with pytest.raises(ValueError, match="denylist"):
        _minimal_template(allowed_paths=("~/.ssh/**",))


def test_template_rejects_denied_bash_at_construction() -> None:
    with pytest.raises(ValueError, match="denylist"):
        _minimal_template(
            allowed_paths=(),
            allowed_bash=("sudo rm -rf /",),
        )


def test_template_rejects_unknown_file_tool_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown"):
        _minimal_template(file_tools=("TeleportToMars",))


# --- to_tools / to_agent_definition / to_agent_options ---------------


def test_to_tools_renders_sorted_deduped_patterned_entries() -> None:
    tpl = _minimal_template(
        allowed_paths=("./src/**", "./tests/**"),
        allowed_bash=("git status", "git log"),
    )
    tools = tpl.to_tools()
    assert list(tools) == sorted(tools)
    assert len(tools) == len(set(tools))
    for entry in tools:
        assert "(" in entry and entry.endswith(")")


def test_to_tools_covers_every_path_x_file_tool_combination() -> None:
    tpl = _minimal_template(
        allowed_paths=("./src/**", "./tests/**"),
        file_tools=("Glob", "Read"),
    )
    expected = {
        "Glob(./src/**)",
        "Glob(./tests/**)",
        "Read(./src/**)",
        "Read(./tests/**)",
    }
    assert set(tpl.to_tools()) == expected


def test_to_tools_returns_empty_when_nothing_allowed() -> None:
    tpl = SubagentTemplate(name="n", description="d", prompt="p")
    assert tpl.to_tools() == ()


def test_to_agent_definition_carries_prompt_and_tools() -> None:
    tpl = _minimal_template()
    agent = tpl.to_agent_definition()
    assert agent.description == tpl.description
    assert agent.prompt == tpl.prompt
    assert agent.model == tpl.model
    assert tuple(agent.tools or ()) == tpl.to_tools()


def test_to_agent_options_registers_under_template_name_not_baseline() -> None:
    tpl = _minimal_template(name="custom-explorer")
    options = tpl.to_agent_options()
    assert "custom-explorer" in options["agents"]
    assert BASELINE_AGENT_NAME not in options["agents"]


def test_to_agent_options_declares_the_expected_toolset() -> None:
    tpl = _minimal_template()
    definition = tpl.to_agent_options()["agents"][tpl.name]
    assert definition["tools"] == list(tpl.to_tools())
    assert definition["description"] == tpl.description
    assert definition["prompt"] == tpl.prompt
    assert definition["model"] == tpl.model


def test_template_exposure_report_matches_locked_down_shape() -> None:
    tpl = _minimal_template()
    report = assess_tool_exposure(tpl.to_agent_options(), agent_name=tpl.name)
    assert report.is_unrestricted is False
    assert report.granted_tools == REQUIRED_TOOLS
    assert report.dangerous_granted == frozenset()
    assert report.unnecessary_granted == frozenset()


# --- Serialization round-trip ----------------------------------------


def test_to_dict_carries_schema_version_and_all_fields() -> None:
    tpl = _minimal_template(allowed_bash=("git status",))
    data = tpl.to_dict()
    assert data["schema_version"] == RECIPE_SCHEMA_VERSION
    assert data["name"] == tpl.name
    assert data["description"] == tpl.description
    assert data["prompt"] == tpl.prompt
    assert data["model"] == tpl.model
    assert data["file_tools"] == list(tpl.file_tools)
    assert data["allowed_paths"] == list(tpl.allowed_paths)
    assert data["allowed_bash"] == list(tpl.allowed_bash)


def test_from_dict_reconstructs_an_equal_template() -> None:
    original = _minimal_template(allowed_bash=("git status", "pytest"))
    clone = SubagentTemplate.from_dict(original.to_dict())
    assert clone == original


def test_from_dict_rejects_wrong_schema_version() -> None:
    payload = _minimal_template().to_dict()
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        SubagentTemplate.from_dict(payload)


def test_to_json_round_trips_via_from_json() -> None:
    original = _minimal_template(allowed_bash=("git diff",))
    text = original.to_json()
    parsed = json.loads(text)
    assert parsed["schema_version"] == RECIPE_SCHEMA_VERSION
    assert SubagentTemplate.from_json(text) == original


def test_to_json_is_sorted_and_indented_for_diffs() -> None:
    tpl = _minimal_template()
    text = tpl.to_json()
    # sort_keys=True → alphabetized top-level keys diff cleanly across runs
    assert text.index('"allowed_bash"') < text.index('"allowed_paths"')
    assert text.index('"allowed_paths"') < text.index('"description"')
    assert "\n" in text  # indent=2 emitted newlines


def test_describe_lists_the_visible_recipe_fields() -> None:
    tpl = _minimal_template(allowed_bash=("git status",))
    text = tpl.describe()
    assert f"name: {tpl.name}" in text
    assert "allowed_paths: ['./src/**']" in text
    assert "allowed_bash: ['git status']" in text
    assert "granted_tools:" in text


# --- Preset recipes ---------------------------------------------------


def test_read_only_explorer_recipe_matches_scoped_agent_shape() -> None:
    tpl = read_only_explorer_recipe()
    assert tpl.name == "file-explorer"
    assert tpl.to_tools() == SCOPED_AGENT_TOOLS


def test_read_only_explorer_grants_only_required_tools() -> None:
    tpl = read_only_explorer_recipe()
    report = assess_tool_exposure(tpl.to_agent_options(), agent_name=tpl.name)
    assert report.granted_tools == REQUIRED_TOOLS
    assert report.dangerous_granted == frozenset()


def test_docs_reviewer_recipe_scopes_to_docs_tree() -> None:
    tpl = docs_reviewer_recipe()
    for entry in tpl.to_tools():
        assert parse_tool_pattern(entry) == "./docs/**"
    tool_names = {parse_tool_name(entry) for entry in tpl.to_tools()}
    assert tool_names == set(DEFAULT_FILE_TOOLS)


def test_git_history_auditor_recipe_includes_bash_and_src_scope() -> None:
    tpl = git_history_auditor_recipe()
    tools = tpl.to_tools()
    assert "Bash(git diff)" in tools
    assert "Bash(git log)" in tools
    assert "Bash(git status)" in tools
    assert "Read(./src/**)" in tools


def test_every_shipped_recipe_passes_denylist_validation() -> None:
    # Constructing each recipe already runs validate_patterned_allowlist +
    # assert_denylist_safe in __post_init__. This test just walks the
    # registry to make sure every entry is present and constructible.
    for name in list_recipe_names():
        tpl = load_recipe(name)
        assert isinstance(tpl, SubagentTemplate)
        assert tpl.name == name


def test_every_shipped_recipe_omits_bare_dangerous_tools() -> None:
    for name in list_recipe_names():
        tpl = load_recipe(name)
        options = tpl.to_agent_options()
        granted = options["agents"][tpl.name]["tools"]
        for dangerous in DANGEROUS_TOOLS:
            assert dangerous not in granted


def test_load_recipe_raises_on_unknown_name() -> None:
    with pytest.raises(KeyError, match="unknown recipe"):
        load_recipe("does-not-exist")


def test_list_recipe_names_returns_sorted_registry_keys() -> None:
    names = list_recipe_names()
    assert list(names) == sorted(SUBAGENT_RECIPES)


# --- End-to-end runner -----------------------------------------------


def _tool_use(tool_name: str, argument: str) -> Dict[str, Any]:
    key = "command" if tool_name == "Bash" else "file_path"
    return {"type": "tool_use", "name": tool_name, "input": {key: argument}}


def _make_stub(messages: List[Dict[str, Any]]):
    async def stub_query(
        *, prompt: str, options: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        for message in messages:
            yield message

    return stub_query


def test_run_template_subagent_delivers_safe_stream_untouched() -> None:
    tpl = read_only_explorer_recipe()
    payload = [
        _tool_use("Read", "./src/subagent_allowlist/allowlist.py"),
        _tool_use("Grep", "./src/subagent_allowlist/observe.py"),
        {"type": "text", "text": "explored the workspace"},
    ]
    delivered = asyncio.run(
        run_template_subagent("explore the repo", _make_stub(payload), tpl)
    )
    assert delivered == payload


def test_run_template_subagent_refuses_forbidden_tool_use() -> None:
    tpl = read_only_explorer_recipe()
    stub = _make_stub([_tool_use("Bash", "sudo rm -rf /")])
    with pytest.raises(DeniedInvocationError):
        asyncio.run(run_template_subagent("do harm", stub, tpl))


def test_run_template_subagent_passes_template_options_to_query_fn() -> None:
    captured: Dict[str, Any] = {}
    tpl = git_history_auditor_recipe()

    async def stub_query(
        *, prompt: str, options: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        captured["options"] = options
        yield {"type": "text", "text": "ok"}

    asyncio.run(run_template_subagent("audit", stub_query, tpl))
    assert tpl.name in captured["options"]["agents"]
    granted = captured["options"]["agents"][tpl.name]["tools"]
    assert granted == list(tpl.to_tools())


def test_run_template_subagent_stops_at_first_denied_message() -> None:
    tpl = read_only_explorer_recipe()
    seen: List[str] = []

    async def stub_query(
        *, prompt: str, options: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        seen.append("safe")
        yield _tool_use("Read", "./src/subagent_allowlist/allowlist.py")
        seen.append("danger")
        yield _tool_use("Read", ".env")
        seen.append("after")
        yield {"type": "text", "text": "should never arrive"}

    with pytest.raises(DeniedInvocationError):
        asyncio.run(run_template_subagent("mixed", stub_query, tpl))
    assert seen == ["safe", "danger"]


# --- Backwards compatibility with the earlier build_agent_options ----


def test_build_agent_options_default_name_still_baseline() -> None:
    # The step-1 through step-6 tests all rely on this default; the
    # `name` parameter is additive and must not change existing calls.
    from subagent_allowlist.baseline import build_baseline_agent
    from subagent_allowlist.spawn import build_agent_options

    options = build_agent_options(build_baseline_agent())
    assert BASELINE_AGENT_NAME in options["agents"]


def test_locked_down_allowlist_is_still_a_valid_recipe_footprint() -> None:
    # The template's default file_tools + a matching path list should
    # produce the same granted-tool set as the step-3 locked-down agent.
    tpl = _minimal_template()
    assert set(parse_tool_name(t) for t in tpl.to_tools()) == set(
        LOCKED_DOWN_AGENT_TOOLS
    )
