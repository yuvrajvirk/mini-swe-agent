from minisweagent.tools import get_tasks_tool_definition


def test_tasks_tool_definition_includes_update_op():
    tool = get_tasks_tool_definition()
    props = tool["function"]["parameters"]["properties"]
    op_enum = props["op"]["enum"]
    assert "update" in op_enum
    assert "include_findings" in props
    assert "include_findings" in props["view"]["enum"]
