"""Stateful session management for REPL mode."""
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path


class Session:
    """Manages persistent session state for the CLI REPL.

    State is stored at ~/.cli-anything-lfiax/session.json with
    file locking to prevent concurrent write corruption.
    """

    def __init__(self, session_file=None):
        if session_file is None:
            session_dir = Path.home() / ".cli-anything-lfiax"
            session_dir.mkdir(parents=True, exist_ok=True)
            self._path = str(session_dir / "session.json")
        else:
            self._path = session_file

        self._data = {
            "active_project": None,
            "history": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if os.path.isfile(self._path):
            self.load()

    @property
    def active_project(self):
        """Current active project path."""
        return self._data.get("active_project")

    def set_project(self, path):
        """Set the active project path."""
        self._data["active_project"] = path
        self._data["updated_at"] = datetime.now(timezone.utc).isoformat()

    @property
    def history(self):
        """Command history list."""
        return self._data.get("history", [])

    def add_to_history(self, command):
        """Add a command to history."""
        self._data.setdefault("history", []).append({
            "command": command,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        # Keep last 100 entries
        if len(self._data["history"]) > 100:
            self._data["history"] = self._data["history"][-100:]
        self._data["updated_at"] = datetime.now(timezone.utc).isoformat()

    def save(self):
        """Persist session to JSON with file locking."""
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)

        # Use file locking to prevent concurrent writes
        mode = "r+" if os.path.isfile(self._path) else "w"
        if mode == "w":
            with open(self._path, "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    json.dump(self._data, f, indent=2, default=str)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        else:
            with open(self._path, "r+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    f.truncate()
                    json.dump(self._data, f, indent=2, default=str)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

    def load(self):
        """Load session from JSON file."""
        if not os.path.isfile(self._path):
            return
        try:
            with open(self._path) as f:
                self._data = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass  # Keep defaults on corrupt file

    def to_dict(self):
        """Serialize session for --json output."""
        return {
            "session_file": self._path,
            "active_project": self.active_project,
            "history_count": len(self.history),
            "created_at": self._data.get("created_at"),
            "updated_at": self._data.get("updated_at"),
        }
