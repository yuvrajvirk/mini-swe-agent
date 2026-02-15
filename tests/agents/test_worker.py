from pathlib import Path

import yaml

from minisweagent.agents.worker import Worker
from minisweagent.environments.local import LocalEnvironment
from minisweagent.integrations.tasks_graph import TasksGraphStore
from minisweagent.models.test_models import (
    DeterministicModel,
    DeterministicToolcallModel,
    make_output,
    make_toolcall_output,
)


def _load_default_agent_config() -> dict:
    config_path = Path("src/minisweagent/config/default.yaml")
    return yaml.safe_load(config_path.read_text())["agent"]


def _load_toolcall_agent_config() -> dict:
    config_path = Path("src/minisweagent/config/mini.yaml")
    return yaml.safe_load(config_path.read_text())["agent"]


def test_worker_adds_summary_note_on_limits_exceeded(tmp_path: Path):
    tasks_path = tmp_path / "tasks.json"
    store = TasksGraphStore(tasks_path)
    task_id = store.create("Assigned task", "Investigate behavior")["task"]["id"]

    model = DeterministicModel(outputs=[make_output("run step", [{"command": "echo 'working'"}])])
    agent = Worker(
        model=model,
        env=LocalEnvironment(),
        **(_load_default_agent_config() | {"step_limit": 1}),
    )
    info = agent.run(
        "Do work",
        worker_task_payload={"id": task_id, "task": "Assigned task", "description": "Investigate behavior", "notes": ""},
        tasks_graph_path=str(tasks_path),
    )

    assert info["exit_status"] == "LimitsExceededWithSummary"
    notes = store.get(task_id)["notes"]
    assert any("[auto-summary][LimitsExceeded]" in note.get("text", "") for note in notes)


def test_worker_adds_summary_note_on_error(tmp_path: Path):
    tasks_path = tmp_path / "tasks.json"
    store = TasksGraphStore(tasks_path)
    task_id = store.create("Assigned task", "Cause an exception")["task"]["id"]

    class _FailingEnv(LocalEnvironment):
        def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None):
            raise RuntimeError("boom")

    model = DeterministicModel(outputs=[make_output("run", [{"command": "echo 'run'"}])])
    agent = Worker(
        model=model,
        env=_FailingEnv(),
        **_load_default_agent_config(),
    )
    info = agent.run(
        "Do work",
        worker_task_payload={"id": task_id, "task": "Assigned task", "description": "Cause an exception", "notes": ""},
        tasks_graph_path=str(tasks_path),
    )

    assert info["exit_status"] == "ErrorWithSummary"
    notes = store.get(task_id)["notes"]
    assert any("[auto-summary][RuntimeError]" in note.get("text", "") for note in notes)


def test_worker_executes_tasks_tool_action(tmp_path: Path):
    tasks_path = tmp_path / "tasks.json"
    store = TasksGraphStore(tasks_path)
    store.create("Seed task", "Seed description")

    tool_calls = [
        {
            "id": "call_tasks_1",
            "type": "function",
            "function": {"name": "tasks", "arguments": '{"op":"list","view":"all"}'},
        }
    ]
    outputs = [
        make_toolcall_output(
            "Listing tasks",
            tool_calls=tool_calls,
            actions=[{"tasks_args": {"op": "list", "view": "all"}, "tool_call_id": "call_tasks_1"}],
        ),
        make_toolcall_output(
            "Finish",
            tool_calls=[
                {
                    "id": "call_bash_1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\necho done"}',
                    },
                }
            ],
            actions=[{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho done", "tool_call_id": "call_bash_1"}],
        ),
    ]
    model = DeterministicToolcallModel(outputs=outputs)
    agent = Worker(
        model=model,
        env=LocalEnvironment(),
        **_load_toolcall_agent_config(),
    )
    info = agent.run(
        "Do work",
        worker_task_payload={"id": "task-1", "task": "Seed task", "description": "Seed", "notes": ""},
        tasks_graph_path=str(tasks_path),
    )

    assert info["exit_status"] == "Submitted"
    tool_messages = [m for m in agent.messages if m.get("role") == "tool"]
    assert any('"ok": true' in msg.get("content", "").lower() for msg in tool_messages)
