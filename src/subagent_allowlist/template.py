from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from .baseline import AgentDefinition
from .denylist import assert_denylist_safe, guard_message
from .patterns import (
    render_patterned_tool,
    validate_patterned_allowlist,
)
from .spawn import QueryFn, build_agent_options

# The default file-reading toolset every scoped explorer starts with.
# Kept alphabetical so serialized recipes diff cleanly across runs.
DEFAULT_FILE_TOOLS: Tuple[str, ...] = ("Glob", "Grep", "Read")

# Version stamp embedded in serialized recipes. Bump when the on-disk
# shape changes so old configs surface a clear error instead of silently
# drifting.
RECIPE_SCHEMA_VERSION: int = 1


@dataclass(frozen=True)
class SubagentTemplate:
    """A validated, serializable recipe for a locked-down Claude Code subagent.

    Composing a template runs both the allowlist and denylist checks on
    construction, so an unsafe recipe fails fast at import time rather
    than at first tool call.
    """

    name: str
    description: str
    prompt: str
    allowed_paths: Tuple[str, ...] = ()
    allowed_bash: Tuple[str, ...] = ()
    file_tools: Tuple[str, ...] = DEFAULT_FILE_TOOLS
    model: str = "sonnet"

    def __post_init__(self) -> None:
        _require_nonempty("name", self.name)
        _require_nonempty("description", self.description)
        _require_nonempty("prompt", self.prompt)
        tools = self.to_tools()
        validate_patterned_allowlist(tools)
        assert_denylist_safe(tools)

    def to_tools(self) -> Tuple[str, ...]:
        file_entries = [
            render_patterned_tool(tool, path)
            for path in self.allowed_paths
            for tool in self.file_tools
        ]
        bash_entries = [
            render_patterned_tool("Bash", cmd) for cmd in self.allowed_bash
        ]
        return tuple(sorted(set(file_entries + bash_entries)))

    def to_agent_definition(self) -> AgentDefinition:
        return AgentDefinition(
            description=self.description,
            prompt=self.prompt,
            tools=self.to_tools(),
            model=self.model,
        )

    def to_agent_options(self) -> Dict[str, Any]:
        return build_agent_options(self.to_agent_definition(), name=self.name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": RECIPE_SCHEMA_VERSION,
            "name": self.name,
            "description": self.description,
            "prompt": self.prompt,
            "model": self.model,
            "file_tools": list(self.file_tools),
            "allowed_paths": list(self.allowed_paths),
            "allowed_bash": list(self.allowed_bash),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def describe(self) -> str:
        return "\n".join(
            [
                f"name: {self.name}",
                f"description: {self.description}",
                f"model: {self.model}",
                f"file_tools: {list(self.file_tools)}",
                f"allowed_paths: {list(self.allowed_paths)}",
                f"allowed_bash: {list(self.allowed_bash)}",
                f"granted_tools: {list(self.to_tools())}",
            ]
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubagentTemplate":
        version = data.get("schema_version", RECIPE_SCHEMA_VERSION)
        if version != RECIPE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported recipe schema_version {version!r}; "
                f"expected {RECIPE_SCHEMA_VERSION}"
            )
        return cls(
            name=data["name"],
            description=data["description"],
            prompt=data["prompt"],
            allowed_paths=tuple(data.get("allowed_paths", ())),
            allowed_bash=tuple(data.get("allowed_bash", ())),
            file_tools=tuple(data.get("file_tools", DEFAULT_FILE_TOOLS)),
            model=data.get("model", "sonnet"),
        )

    @classmethod
    def from_json(cls, text: str) -> "SubagentTemplate":
        return cls.from_dict(json.loads(text))


def _require_nonempty(field_name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"SubagentTemplate.{field_name} must be non-empty")


# --- Preset recipes ---------------------------------------------------
#
# Each factory returns a ready-to-run SubagentTemplate. Copy one into
# your own repo, tweak the prompt, and pass the result to
# `run_template_subagent`. The validation in __post_init__ guarantees the
# recipe cannot regress into an unsafe posture without failing at import.


def read_only_explorer_recipe() -> SubagentTemplate:
    return SubagentTemplate(
        name="file-explorer",
        description="Explores a directory and reports what it finds",
        prompt=(
            "You are a file explorer subagent. Given a directory path, "
            "list its files, read the interesting ones, and return a short "
            "summary of what the project appears to do."
        ),
        allowed_paths=("./src/**",),
    )


def docs_reviewer_recipe() -> SubagentTemplate:
    return SubagentTemplate(
        name="docs-reviewer",
        description="Reads Markdown/MDX docs and reports inconsistencies",
        prompt=(
            "You are a documentation reviewer. Read the docs under the "
            "project's docs tree, cross-reference claims, and report "
            "sections that contradict each other or the code."
        ),
        allowed_paths=("./docs/**",),
        file_tools=("Glob", "Grep", "Read"),
    )


def git_history_auditor_recipe() -> SubagentTemplate:
    return SubagentTemplate(
        name="git-history-auditor",
        description="Reads code + runs read-only git commands to audit recent history",
        prompt=(
            "You are a git history auditor. Use `git log`, `git diff`, and "
            "`git status` to survey recent changes, then read the affected "
            "source files and summarize what shipped and what regressed."
        ),
        allowed_paths=("./src/**",),
        allowed_bash=("git diff", "git log", "git status"),
    )


SUBAGENT_RECIPES: Dict[str, Callable[[], SubagentTemplate]] = {
    "file-explorer": read_only_explorer_recipe,
    "docs-reviewer": docs_reviewer_recipe,
    "git-history-auditor": git_history_auditor_recipe,
}


def list_recipe_names() -> Tuple[str, ...]:
    return tuple(sorted(SUBAGENT_RECIPES))


def load_recipe(name: str) -> SubagentTemplate:
    factory = SUBAGENT_RECIPES.get(name)
    if factory is None:
        raise KeyError(
            f"unknown recipe {name!r}; available: {list_recipe_names()}"
        )
    return factory()


# --- End-to-end runner ------------------------------------------------


async def run_template_subagent(
    user_prompt: str,
    query_fn: QueryFn,
    template: SubagentTemplate,
) -> List[Any]:
    """Spawn `template`'s subagent, guarding every streamed message.

    Mirrors `run_guarded_subagent` but uses the template's own name for
    the subagent slot and its declared toolset — so the config the
    recipe describes is the config the model actually sees.
    """

    options = template.to_agent_options()
    messages: List[Any] = []
    async for message in query_fn(prompt=user_prompt, options=options):
        guard_message(message)
        messages.append(message)
    return messages
