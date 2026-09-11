"""
Unit tests for Gemini File Search management, store reuse, and Cloudflare D1 migration schema.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from scripts.dataset.manage_file_search import (
    load_checkpoint,
    save_checkpoint,
    find_or_create_store,
    STORE_DISPLAY_NAME,
)


def test_checkpoint_save_and_load():
    """Verify upload checkpoint serializes and deserializes accurately."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cp_file = Path(tmpdir) / "checkpoint.json"
        shards = {"shard_001.txt", "shard_002.txt", "shard_003.txt"}

        save_checkpoint(cp_file, shards)
        assert cp_file.exists()

        loaded = load_checkpoint(cp_file)
        assert loaded == shards
        assert len(loaded) == 3


def test_find_or_create_store_reuses_existing_by_display_name():
    """Verify existing store with matching display name is reused without creating a duplicate."""
    mock_client = MagicMock()
    mock_store_1 = MagicMock()
    mock_store_1.name = "fileSearchStores/existing-123"
    mock_store_1.display_name = STORE_DISPLAY_NAME

    mock_client.file_search_stores.list.return_value = [mock_store_1]

    store_name = find_or_create_store(mock_client, force_create=False)
    assert store_name == "fileSearchStores/existing-123"
    # Ensure create was NOT called
    assert mock_client.file_search_stores.create.call_count == 0


def test_find_or_create_store_creates_when_not_found():
    """Verify store is created when no matching display name exists."""
    mock_client = MagicMock()
    mock_client.file_search_stores.list.return_value = []
    
    mock_created = MagicMock()
    mock_created.name = "fileSearchStores/created-456"
    mock_created.display_name = STORE_DISPLAY_NAME
    mock_client.file_search_stores.create.return_value = mock_created

    with patch.dict("os.environ", {}, clear=True):
        store_name = find_or_create_store(mock_client, force_create=True)
        assert store_name == "fileSearchStores/created-456"
        assert mock_client.file_search_stores.create.call_count == 1


def test_d1_migration_sql_schema_validity():
    """Verify D1 migration file exists, creates conversations and messages tables with indexes and owner isolation."""
    migration_file = Path(__file__).resolve().parents[1] / "backend" / "cloudflare" / "migrations" / "0001_conversations.sql"
    assert migration_file.exists(), f"Migration file missing at {migration_file}"

    sql = migration_file.read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS conversations" in sql
    assert "owner_id TEXT NOT NULL" in sql
    assert "CREATE TABLE IF NOT EXISTS messages" in sql
    assert "idx_conversations_owner_updated" in sql
    assert "idx_messages_conversation_created" in sql
    assert "FOREIGN KEY (conversation_id)" in sql
    assert "ON DELETE CASCADE" in sql
