"""Tests for persistent verification store."""

from app.cache.verification_store import VerificationStore


class TestVerificationStore:
    def test_set_and_get(self, tmp_path):
        store = VerificationStore(store_path=str(tmp_path / "verif.json"))
        store.set("q1", {"status": "verified", "confidence": 0.95})
        result = store.get("q1")
        assert result is not None
        assert result["status"] == "verified"
        assert result["confidence"] == 0.95

    def test_get_nonexistent(self, tmp_path):
        store = VerificationStore(store_path=str(tmp_path / "verif.json"))
        assert store.get("nonexistent") is None

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "verif.json")
        store1 = VerificationStore(store_path=path)
        store1.set("q1", {"status": "verified"})

        # Create new store instance — should load from disk
        store2 = VerificationStore(store_path=path)
        assert store2.get("q1") == {"status": "verified"}

    def test_set_pending(self, tmp_path):
        store = VerificationStore(store_path=str(tmp_path / "verif.json"))
        store.set_pending("q1", batch_id="batch_123")
        result = store.get("q1")
        assert result["status"] == "pending"
        assert result["batch_id"] == "batch_123"

    def test_exists(self, tmp_path):
        store = VerificationStore(store_path=str(tmp_path / "verif.json"))
        assert store.exists("q1") is False
        store.set("q1", {"status": "verified"})
        assert store.exists("q1") is True

    def test_count(self, tmp_path):
        store = VerificationStore(store_path=str(tmp_path / "verif.json"))
        assert store.count() == 0
        store.set("q1", {"status": "verified"})
        store.set("q2", {"status": "pending"})
        assert store.count() == 2

    def test_cleanup(self, tmp_path):
        store = VerificationStore(store_path=str(tmp_path / "verif.json"))
        for i in range(10):
            store.set(f"q{i}", {"status": "verified"})
        removed = store.cleanup(max_entries=5)
        assert removed == 5
        assert store.count() == 5

    def test_corrupt_file(self, tmp_path):
        path = tmp_path / "verif.json"
        path.write_text("not valid json")
        store = VerificationStore(store_path=str(path))
        assert store.count() == 0
