from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple

from .observe import DANGEROUS_TOOLS, KNOWN_TOOLS

# Scope for every file-touching tool the file-explorer holds. Anchoring on
# ./src/** keeps the subagent inside the tracked source tree even if the
# operator points it at a wider working directory by accident.
WORKSPACE_SCOPE_PATTERN: str = "./src/**"

# Bash commands considered read-only enough to hand to a hypothetical
# builder subagent. Ordered alphabetically so serialized configs diff
# cleanly across runs.
SAFE_BASH_COMMANDS: Tuple[str, ...] = (
    "git diff",
    "git log",
    "git status",
    "pytest",
)

_ENTRY_RE = re.compile(r"^(?P<name>[A-Z][A-Za-z]*)\((?P<pattern>[^)]+)\)$")


@dataclass(frozen=True)
class PatternedTool:
    name: str
    pattern: str

    def render(self) -> str:
        return f"{self.name}({self.pattern})"


def render_patterned_tool(name: str, pattern: str) -> str:
    return PatternedTool(name, pattern).render()


def parse_tool_name(entry: str) -> str:
    match = _ENTRY_RE.match(entry)
    if match is None:
        return entry
    return match.group("name")


def parse_tool_pattern(entry: str) -> str:
    match = _ENTRY_RE.match(entry)
    if match is None:
        return ""
    return match.group("pattern")


def build_scoped_file_tools() -> Tuple[str, ...]:
    file_tools = ("Glob", "Grep", "Read")
    return tuple(
        render_patterned_tool(name, WORKSPACE_SCOPE_PATTERN) for name in file_tools
    )


def build_scoped_bash_commands() -> Tuple[str, ...]:
    return tuple(render_patterned_tool("Bash", cmd) for cmd in SAFE_BASH_COMMANDS)


SCOPED_AGENT_TOOLS: Tuple[str, ...] = build_scoped_file_tools()


def validate_patterned_allowlist(tools: Tuple[str, ...]) -> None:
    unknown = sorted(
        entry for entry in tools if parse_tool_name(entry) not in KNOWN_TOOLS
    )
    if unknown:
        raise ValueError(f"unknown tools in allowlist: {unknown}")
    unconstrained_dangerous = sorted(
        entry
        for entry in tools
        if parse_tool_name(entry) in DANGEROUS_TOOLS and parse_tool_pattern(entry) == ""
    )
    if unconstrained_dangerous:
        raise ValueError(
            "dangerous tools without a constraining pattern: "
            f"{unconstrained_dangerous}"
        )
