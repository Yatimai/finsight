"""
Persistent verification store.
Stores verification results in a JSON file so they survive server restarts.
Used by the async verification flow: the frontend polls for results
after the initial response is returned.
"""

import json
import threading
from pathlib import Path

from app.logging import get_logger

logger = get_logger("verification_store")


class VerificationStore:
    """
    Thread-safe, file-backed store for verification results.

    On write: updates in-memory dict + writes to disk.
    On startup: loads existing results from disk.
    """

    def __init__(self, store_path: str = "./data/verifications.json"):
        self.path = Path(store_path)
        self._lock = threading.Lock()
        self._data: dict[str, dict] = self._load()

    def _load(self) -> dict:
        """Load existing verifications from disk."""
        if self.path.exists():
            try:
                with open(self.path) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("store_load_failed", error=str(e))
                return {}
        return {}

    def _save(self):
        """Persist current state to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.path, "w") as f:
                json.dump(self._data, f, ensure_ascii=False, default=str)
        except OSError as e:
            logger.error("store_save_failed", error=str(e))

    def get(self, query_id: str) -> dict | None:
        """Get verification result for a query_id."""
        with self._lock:
            return self._data.get(query_id)

    def set(self, query_id: str, verification: dict) -> None:
        """Store a verification result."""
        with self._lock:
            self._data[query_id] = verification
            self._save()

        logger.debug("store_updated", query_id=query_id, status=verification.get("status"))

    def set_pending(self, query_id: str, batch_id: str | None = None) -> None:
        """Mark a verification as pending (submitted but not yet complete)."""
        with self._lock:
            self._data[query_id] = {
                "status": "pending",
                "batch_id": batch_id,
                "confidence": None,
                "claims": [],
                "summary": "Verification in progress",
            }
            self._save()

    def exists(self, query_id: str) -> bool:
        """Check if a query_id has any entry (including pending)."""
        with self._lock:
            return query_id in self._data

    def count(self) -> int:
        """Total number of stored verifications."""
        with self._lock:
            return len(self._data)

    def cleanup(self, max_entries: int = 10000) -> int:
        """Remove oldest entries if store exceeds max_entries. Returns count removed."""
        with self._lock:
            if len(self._data) <= max_entries:
                return 0

            # Keep most recent entries (by insertion order in Python 3.7+)
            keys = list(self._data.keys())
            to_remove = keys[: len(keys) - max_entries]
            for key in to_remove:
                del self._data[key]

            self._save()
            return len(to_remove)
