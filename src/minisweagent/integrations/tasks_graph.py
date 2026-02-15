"""JSON-backed task graph store for tool-calling workflows."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TasksGraphStore:
    path: Path

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"next_id": 1, "tasks": {}, "client_ids": {}}
        data = json.loads(self.path.read_text() or "{}")
        data.setdefault("next_id", 1)
        data.setdefault("tasks", {})
        data.setdefault("client_ids", {})
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    def _new_task_id(self, data: dict[str, Any]) -> str:
        task_id = f"task-{data['next_id']}"
        data["next_id"] += 1
        return task_id

    def _require_task(self, data: dict[str, Any], task_id: str) -> dict[str, Any]:
        task = data["tasks"].get(task_id)
        if task is None:
            raise ValueError(f"Unknown task id: {task_id}")
        return task

    def _is_ready(self, task: dict[str, Any], tasks: dict[str, dict[str, Any]]) -> bool:
        if task["status"] != "open":
            return False
        for blocker_id in task["blockers"]:
            blocker = tasks.get(blocker_id)
            if blocker and blocker["status"] == "open":
                return False
        return True

    def create(self, title: str, description: str, client_id: str | None = None) -> dict[str, Any]:
        data = self._load()
        if client_id:
            existing = data["client_ids"].get(client_id)
            if existing and existing in data["tasks"]:
                return {"created": False, "task": self._full_task(data["tasks"][existing], data["tasks"])}

        task_id = self._new_task_id(data)
        now = _utc_now()
        task = {
            "id": task_id,
            "title": title,
            "description": description,
            "status": "open",
            "created_at": now,
            "updated_at": now,
            "notes": [],
            "blockers": [],
            "blocked": [],
            "close_reason": "",
            "client_id": client_id or "",
        }
        data["tasks"][task_id] = task
        if client_id:
            data["client_ids"][client_id] = task_id
        self._save(data)
        return {"created": True, "task": self._full_task(task, data["tasks"])}

    def get(self, task_id: str) -> dict[str, Any]:
        data = self._load()
        return self._full_task(self._require_task(data, task_id), data["tasks"])

    def list(
        self,
        view: str = "open",
        include_full: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        data = self._load()
        tasks = list(data["tasks"].values())
        tasks.sort(key=lambda x: x["id"])
        if view == "open":
            tasks = [t for t in tasks if t["status"] == "open"]
        elif view == "closed":
            tasks = [t for t in tasks if t["status"] == "closed"]
        elif view == "ready":
            tasks = [t for t in tasks if self._is_ready(t, data["tasks"])]
        elif view != "all":
            raise ValueError(f"Unsupported view: {view}")

        rows = [
            self._full_task(t, data["tasks"]) if include_full else self._compact_task(t, data["tasks"]) for t in tasks
        ]
        start = max(0, offset)
        page = rows[start : start + limit] if limit is not None else rows[start:]
        return {"view": view, "total": len(rows), "offset": start, "count": len(page), "tasks": page}

    def note_append(self, task_id: str, note: str) -> dict[str, Any]:
        data = self._load()
        task = self._require_task(data, task_id)
        task["notes"].append({"ts": _utc_now(), "text": note})
        task["updated_at"] = _utc_now()
        self._save(data)
        return self._full_task(task, data["tasks"])

    def dep_add(self, blocker_id: str, blocked_id: str) -> dict[str, Any]:
        if blocker_id == blocked_id:
            raise ValueError("A task cannot depend on itself")
        data = self._load()
        blocker = self._require_task(data, blocker_id)
        blocked = self._require_task(data, blocked_id)
        changed = False
        if blocker_id not in blocked["blockers"]:
            blocked["blockers"].append(blocker_id)
            changed = True
        if blocked_id not in blocker["blocked"]:
            blocker["blocked"].append(blocked_id)
            changed = True
        if changed:
            blocked["updated_at"] = _utc_now()
            blocker["updated_at"] = _utc_now()
            self._save(data)
        return {"changed": changed, "blocker_id": blocker_id, "blocked_id": blocked_id}

    def close(self, task_id: str, reason: str) -> dict[str, Any]:
        data = self._load()
        task = self._require_task(data, task_id)
        task["status"] = "closed"
        task["close_reason"] = reason
        task["updated_at"] = _utc_now()
        self._save(data)
        return self._full_task(task, data["tasks"])

    def delete(self, task_id: str, hard: bool = False) -> dict[str, Any]:
        data = self._load()
        task = self._require_task(data, task_id)
        if hard:
            for blocker_id in task["blockers"]:
                blocker = data["tasks"].get(blocker_id)
                if blocker:
                    blocker["blocked"] = [x for x in blocker["blocked"] if x != task_id]
            for blocked_id in task["blocked"]:
                blocked = data["tasks"].get(blocked_id)
                if blocked:
                    blocked["blockers"] = [x for x in blocked["blockers"] if x != task_id]
            if task.get("client_id"):
                data["client_ids"].pop(task["client_id"], None)
            del data["tasks"][task_id]
            self._save(data)
            return {"deleted": True, "hard": True, "id": task_id}

        task["status"] = "deleted"
        task["updated_at"] = _utc_now()
        self._save(data)
        return self._full_task(task, data["tasks"])

    def _compact_task(self, task: dict[str, Any], tasks: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return {
            "id": task["id"],
            "title": task["title"],
            "status": task["status"],
            "ready": self._is_ready(task, tasks),
            "blockers": list(task["blockers"]),
            "blocked": list(task["blocked"]),
            "notes_count": len(task["notes"]),
            "updated_at": task["updated_at"],
        }

    def _full_task(self, task: dict[str, Any], tasks: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return {
            "id": task["id"],
            "title": task["title"],
            "description": task["description"],
            "status": task["status"],
            "ready": self._is_ready(task, tasks),
            "notes": list(task["notes"]),
            "blockers": list(task["blockers"]),
            "blocked": list(task["blocked"]),
            "created_at": task["created_at"],
            "updated_at": task["updated_at"],
            "close_reason": task.get("close_reason", ""),
            "client_id": task.get("client_id", ""),
        }


def resolve_tasks_store_path(extra_template_vars: dict[str, Any], env: Any) -> Path:
    custom = extra_template_vars.get("tasks_graph_path")
    if custom:
        return Path(str(custom))

    output_dir = extra_template_vars.get("output_dir")
    instance_id = extra_template_vars.get("instance_id")
    if output_dir and instance_id:
        return Path(str(output_dir)) / str(instance_id) / "tasks.json"

    env_cwd = getattr(getattr(env, "config", None), "cwd", "")
    base = Path(str(env_cwd)) if env_cwd else Path(os.getcwd())
    return base / ".tasks.json"

