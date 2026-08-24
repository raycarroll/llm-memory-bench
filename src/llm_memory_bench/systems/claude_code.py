from __future__ import annotations

from ..providers.base import ToolCall
from .base import MemorySystem

# Faithfully reproduces the auto-memory instructions from Claude Code's
# system prompt. The LLM receives these as part of its system context and
# must decide what to write, which type to use, and how to structure the
# frontmatter and body.

SYSTEM_PROMPT = """\
You are a helpful assistant. You have a persistent, file-based memory system. \
You should build up this memory over time so that future conversations can \
have a complete picture of who the user is, how they'd like to collaborate \
with you, what behaviors to avoid or repeat, and the context behind the work \
the user gives you.

## Types of memory

There are several discrete types of memory that you can store:

### user
Contain information about the user's role, goals, responsibilities, and \
knowledge. Great user memories help you tailor your future behavior to the \
user's preferences and perspective. Your goal in reading and writing these \
memories is to build up an understanding of who the user is and how you can \
be most helpful to them specifically. Avoid writing memories about the user \
that could be viewed as a negative judgement or that are not relevant to the \
work you're trying to accomplish together.

### feedback
Guidance the user has given you about how to approach work — both what to \
avoid and what to keep doing. Record from failure AND success: if you only \
save corrections, you will avoid past mistakes but drift away from approaches \
the user has already validated. Include *why* so you can judge edge cases later.

### project
Information that you learn about ongoing work, goals, initiatives, bugs, or \
incidents within the project that is not otherwise derivable from the code or \
git history. Always convert relative dates to absolute dates when saving.

### reference
Stores pointers to where information can be found in external systems. These \
memories allow you to remember where to look to find up-to-date information \
outside of the project directory.

## What NOT to save

- Code patterns, conventions, architecture, file paths, or project structure \
— these can be derived by reading the current project state.
- Git history, recent changes, or who-changed-what — git log / git blame are \
authoritative.
- Debugging solutions or fix recipes — the fix is in the code; the commit \
message has the context.
- Ephemeral task details: in-progress work, temporary state, current \
conversation context.

## When to save

Save immediately when you learn details about the user's role, preferences, \
responsibilities, or knowledge. Save when the user corrects your approach \
OR confirms a non-obvious approach worked. Save when you learn who is doing \
what, why, or by when. Save when you learn about resources in external systems.

When in doubt, check if there is an existing memory you can update before \
writing a new one. Do not write duplicate memories."""

SAVE_MEMORY_TOOL = {
    "name": "save_memory",
    "description": (
        "Save a memory to the persistent file-based memory system. "
        "Each memory has a name (short kebab-case slug), a one-line description "
        "used to decide relevance in future conversations, a type (user, feedback, "
        "project, or reference), and a body with the memory content."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "Short kebab-case slug for the memory file, e.g. "
                    "'user-role', 'feedback-testing', 'project-auth-rewrite'."
                ),
            },
            "description": {
                "type": "string",
                "description": (
                    "One-line summary used to decide relevance in future "
                    "conversations. Be specific."
                ),
            },
            "type": {
                "type": "string",
                "enum": ["user", "feedback", "project", "reference"],
                "description": "The type of memory.",
            },
            "body": {
                "type": "string",
                "description": (
                    "The memory content. For feedback/project types, structure "
                    "as: rule/fact, then a Why: line and a How to apply: line."
                ),
            },
        },
        "required": ["name", "description", "type", "body"],
    },
}


class ClaudeCodeMemorySystem(MemorySystem):
    name = "claude_code"
    description = (
        "Claude Code's file-based auto-memory system with typed memories "
        "(user, feedback, project, reference) and structured frontmatter"
    )

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def tool_definitions(self) -> list[dict]:
        return [SAVE_MEMORY_TOOL]

    def extract_stored_fact(self, tool_call: ToolCall) -> str:
        body = tool_call.arguments.get("body", "")
        desc = tool_call.arguments.get("description", "")
        if body:
            return body
        return desc

    def format_tool_result(self, tool_call: ToolCall) -> dict:
        name = tool_call.arguments.get("name", "unknown")
        return {
            "status": "success",
            "message": f"Memory saved to {name}.md",
        }
