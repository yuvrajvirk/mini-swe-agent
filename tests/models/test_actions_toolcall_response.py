import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.actions_toolcall_response import (
    TASKS_TOOL_RESPONSE_API,
    parse_toolcall_actions_response,
)


def test_parse_tasks_tool_response_action():
    output = [
        {
            "type": "function_call",
            "call_id": "call_tasks_1",
            "name": "tasks",
            "arguments": '{"op":"list","view":"all"}',
        }
    ]
    result = parse_toolcall_actions_response(output, format_error_template="{{ error }}")
    assert result == [{"tasks_args": {"op": "list", "view": "all"}, "tool_call_id": "call_tasks_1"}]


def test_parse_tasks_tool_response_missing_op():
    output = [
        {
            "type": "function_call",
            "call_id": "call_tasks_1",
            "name": "tasks",
            "arguments": '{"view":"all"}',
        }
    ]
    with pytest.raises(FormatError) as exc_info:
        parse_toolcall_actions_response(output, format_error_template="{{ error }}")
    text = exc_info.value.messages[0]["content"][0]["text"]
    assert "Missing 'op' argument" in text


def test_tasks_tool_response_schema():
    assert TASKS_TOOL_RESPONSE_API["type"] == "function"
    assert TASKS_TOOL_RESPONSE_API["name"] == "tasks"
    assert "op" in TASKS_TOOL_RESPONSE_API["parameters"]["properties"]

