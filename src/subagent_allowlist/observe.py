from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Optional

from .baseline import BASELINE_AGENT_NAME

# Canonical set of Claude Code built-in tools a subagent can inherit
# when no `tools` allowlist is supplied. Keep this list conservative and
# alphabetized so drift is easy to spot in diffs.
KNOWN_TOOLS: FrozenSet[str] = frozenset(
    {
        "Bash",
        "Edit",
        "Glob",
        "Grep",
        "NotebookEdit",
        "Read",
        "WebFetch",
        "WebSearch",
        "Write",
    }
)

# Tools whose misuse costs the operator real money, real data, or real
# blast radius. A file-explorer subagent has no business touching any of
# these, but the unrestricted baseline hands them out anyway.
DANGEROUS_TOOLS: FrozenSet[str] = frozenset(
    {"Bash", "Edit", "NotebookEdit", "WebFetch", "Write"}
)

# The minimal set the file-explorer prompt actually needs: walk the tree,
# read files, grep for interesting strings.
REQUIRED_TOOLS: FrozenSet[str] = frozenset({"Glob", "Grep", "Read"})


@dataclass(frozen=True)
class ToolExposureReport:
    is_unrestricted: bool
    granted_tools: FrozenSet[str]
    dangerous_granted: FrozenSet[str]
    unnecessary_granted: FrozenSet[str]

    @property
    def over_grant_ratio(self) -> float:
        if not self.granted_tools:
            return 0.0
        return len(self.unnecessary_granted) / len(self.granted_tools)


def _resolve_granted(tools: Optional[Iterable[str]]) -> FrozenSet[str]:
    if tools is None:
        return KNOWN_TOOLS
    return frozenset(tools)


def assess_tool_exposure(
    options: Dict[str, Any],
    agent_name: str = BASELINE_AGENT_NAME,
) -> ToolExposureReport:
    definition = options["agents"][agent_name]
    declared = definition.get("tools")
    granted = _resolve_granted(declared)
    return ToolExposureReport(
        is_unrestricted=declared is None,
        granted_tools=granted,
        dangerous_granted=granted & DANGEROUS_TOOLS,
        unnecessary_granted=granted - REQUIRED_TOOLS,
    )


def summarize_report(report: ToolExposureReport) -> List[str]:
    lines = [
        f"unrestricted: {report.is_unrestricted}",
        f"granted: {sorted(report.granted_tools)}",
        f"dangerous_granted: {sorted(report.dangerous_granted)}",
        f"unnecessary_granted: {sorted(report.unnecessary_granted)}",
        f"over_grant_ratio: {report.over_grant_ratio:.2f}",
    ]
    return lines
