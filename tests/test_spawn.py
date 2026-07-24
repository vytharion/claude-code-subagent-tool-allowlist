from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List

from subagent_allowlist.baseline import BASELINE_AGENT_NAME, build_baseline_agent
from subagent_allowlist.spawn import build_agent_options, run_baseline_subagent


def test_options_registers_subagent_by_name() -> None:
    options = build_agent_options(build_baseline_agent())
    assert BASELINE_AGENT_NAME in options["agents"]


def test_options_carry_description_prompt_and_model() -> None:
    agent = build_baseline_agent()
    definition = build_agent_options(agent)["agents"][BASELINE_AGENT_NAME]
    assert definition["description"] == agent.description
    assert definition["prompt"] == agent.prompt
    assert definition["model"] == agent.model


def test_options_omit_tools_key_when_unrestricted() -> None:
    options = build_agent_options(build_baseline_agent())
    definition = options["agents"][BASELINE_AGENT_NAME]
    # absence of "tools" is what makes this the *baseline*; step 3 adds it
    assert "tools" not in definition


def test_run_baseline_subagent_invokes_injected_query() -> None:
    calls: List[Dict[str, Any]] = []

    async def stub_query(*, prompt: str, options: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        calls.append({"prompt": prompt, "options": options})
        yield {"type": "text", "text": "stub reply"}

    messages = asyncio.run(run_baseline_subagent("Explore ./tmp", stub_query))

    assert messages == [{"type": "text", "text": "stub reply"}]
    assert len(calls) == 1
    assert calls[0]["prompt"] == "Explore ./tmp"
    assert BASELINE_AGENT_NAME in calls[0]["options"]["agents"]
