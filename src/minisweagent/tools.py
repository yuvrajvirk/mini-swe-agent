"""Tool definitions for mini-swe-agent."""


def get_tasks_tool_definition() -> dict:
    """Get the tasks graph tool definition."""
    return {
        "type": "function",
        "function": {
            "name": "tasks",
            "description": (
                "Manage a lightweight task graph without shell commands. "
                "Use this tool for task creation, dependency links, notes, and task views."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["create", "get", "list", "note_append", "dep_add", "close", "delete"],
                        "description": "Operation to execute on the task graph.",
                    },
                    "id": {"type": "string", "description": "Task ID for get/note_append/close/delete."},
                    "title": {"type": "string", "description": "Title for create."},
                    "description": {"type": "string", "description": "Description for create."},
                    "client_id": {
                        "type": "string",
                        "description": (
                            "Optional idempotency key for create. "
                            "If reused, returns the existing task instead of creating duplicates."
                        ),
                    },
                    "view": {
                        "type": "string",
                        "enum": ["open", "ready", "closed", "all"],
                        "description": "List filter for list operation. Defaults to open.",
                    },
                    "include_full": {
                        "type": "boolean",
                        "description": "For list: include full notes/deps instead of compact rows.",
                    },
                    "limit": {"type": "integer", "description": "Optional list page size."},
                    "offset": {"type": "integer", "description": "Optional list offset."},
                    "note": {"type": "string", "description": "Note text for note_append."},
                    "blocker_id": {"type": "string", "description": "Dependency blocker task id."},
                    "blocked_id": {"type": "string", "description": "Dependency blocked task id."},
                    "reason": {"type": "string", "description": "Reason for close."},
                    "hard": {"type": "boolean", "description": "For delete: hard delete when true."},
                },
                "required": ["op"],
            },
        },
    }

