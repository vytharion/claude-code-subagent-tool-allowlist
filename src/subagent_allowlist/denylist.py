from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Any, FrozenSet, Iterable, Mapping, Tuple

from .patterns import parse_tool_name, parse_tool_pattern

# Substrings inside a Bash invocation that must never run, no matter what
# the allowlist claims. These target destructive filesystem ops, privilege
# escalation, credential exfiltration, and remote-code-execution shell
# patterns. Case-insensitive comparison happens in is_denied_bash().
DENIED_BASH_SUBSTRINGS: Tuple[str, ...] = (
    ":(){ :|:& };:",
    "chmod 777",
    "dd if=",
    "mkfs",
    "rm -fr",
    "rm -rf",
    "shutdown",
    "sudo",
)

# Path globs that a file-reading subagent must never touch — even in
# read mode — because they contain secrets or escape the workspace scope.
DENIED_PATH_GLOBS: Tuple[str, ...] = (
    "**/.aws/**",
    "**/.env",
    "**/.env.*",
    "**/.git/**",
    "**/.ssh/**",
    "**/id_ed25519*",
    "**/id_rsa*",
    "/etc/**",
    "~/.aws/**",
    "~/.ssh/**",
)

# Piping a downloaded script straight into a shell is the canonical
# remote-code-execution primitive; catch it as a regex instead of listing
# every scheme (curl, wget, fetch, http_load...).
_PIPE_TO_SHELL_RE = re.compile(r"\|\s*(sh|bash|zsh|ksh|fish)\b")

_FILE_TOOLS: FrozenSet[str] = frozenset(
    {"Edit", "Glob", "Grep", "NotebookEdit", "Read", "Write"}
)


class DeniedInvocationError(RuntimeError):
    """Raised when a tool invocation matches the denylist."""


@dataclass(frozen=True)
class DenylistViolation:
    entry: str
    reason: str


def is_denied_bash(command: str) -> bool:
    lowered = command.lower()
    if any(token in lowered for token in DENIED_BASH_SUBSTRINGS):
        return True
    return _PIPE_TO_SHELL_RE.search(lowered) is not None


def _glob_match(path: str, glob: str) -> bool:
    if fnmatch.fnmatch(path, glob):
        return True
    # fnmatch's `**` is not recursive — it can't consume zero directory
    # segments, so `**/.env` misses a top-level `.env`. Retry with the
    # `**/` prefix stripped so the glob matches at the workspace root too.
    if glob.startswith("**/"):
        return fnmatch.fnmatch(path, glob[3:])
    return False


def is_denied_path(path: str) -> bool:
    return any(_glob_match(path, glob) for glob in DENIED_PATH_GLOBS)


def is_denied_tool_call(tool_name: str, argument: str) -> bool:
    if tool_name == "Bash":
        return is_denied_bash(argument)
    if tool_name in _FILE_TOOLS:
        return is_denied_path(argument)
    return False


def guard_tool_use(tool_name: str, argument: str) -> None:
    if not is_denied_tool_call(tool_name, argument):
        return
    raise DeniedInvocationError(
        f"tool call {tool_name}({argument!r}) blocked by denylist"
    )


def _describe_violation(name: str, pattern: str) -> str:
    if name == "Bash":
        return f"bash command matches denylist: {pattern!r}"
    return f"file path matches denied glob: {pattern!r}"


def scan_allowlist_for_denied_entries(
    tools: Iterable[str],
) -> Tuple[DenylistViolation, ...]:
    violations = []
    for entry in tools:
        name = parse_tool_name(entry)
        pattern = parse_tool_pattern(entry)
        if pattern and is_denied_tool_call(name, pattern):
            violations.append(
                DenylistViolation(entry=entry, reason=_describe_violation(name, pattern))
            )
    return tuple(violations)


def assert_denylist_safe(tools: Iterable[str]) -> None:
    violations = scan_allowlist_for_denied_entries(tools)
    if not violations:
        return
    rendered = "; ".join(f"{v.entry} — {v.reason}" for v in violations)
    raise ValueError(f"denylist violations in allowlist: {rendered}")


# Argument-key hints per tool. The SDK routes each tool's input through a
# tool-specific schema, so Bash carries "command" while file tools carry a
# path under one of several field names — check them in order.
_BASH_ARG_KEYS: Tuple[str, ...] = ("command",)
_PATH_ARG_KEYS: Tuple[str, ...] = (
    "file_path",
    "notebook_path",
    "path",
    "pattern",
)


def _pick_argument(tool_name: str, payload: Mapping[str, Any]) -> str:
    keys = _BASH_ARG_KEYS if tool_name == "Bash" else _PATH_ARG_KEYS
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return str(value)
    return ""


def guard_message(message: Any) -> None:
    if not isinstance(message, Mapping):
        return
    if message.get("type") != "tool_use":
        return
    tool_name = str(message.get("name", ""))
    payload = message.get("input") or {}
    guard_tool_use(tool_name, _pick_argument(tool_name, payload))
