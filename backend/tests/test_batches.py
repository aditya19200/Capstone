"""
tests/test_batches.py — Tests for batch ingestion, listing, and export.

Covers three layers so a passing route test can't hide a broken mock branch
underneath:
  1. services/mock_db.py's new batches/batch_items functions, in isolation.
  2. repositories/batches.py and repositories/batch_items.py mock-mode
     fallbacks, in isolation.
  3. The POST/GET /batches/* routes end-to-end.
"""

import io

import pytest
from fastapi.testclient import TestClient

from main import app
from repositories import batch_items as batch_items_repo
from repositories import batches as batches_repo
from services import mock_db


@pytest.fixture(autouse=True)
def clear_mock_db():
    mock_db._predictions.clear()
    mock_db._annotations.clear()
    mock_db._xai_jobs.clear()
    mock_db._model_versions.clear()
    mock_db._retrain_jobs.clear()
    mock_db._dataset_versions.clear()
    mock_db._batches.clear()
    mock_db._batch_items.clear()
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ===========================================================================
# mock_db — unit tests on the new batches/batch_items functions directly
# ===========================================================================

class TestMockDbBatches:

    def test_create_batch_returns_pending_status_and_zero_completed(self):
        batch = mock_db.create_batch(source="paste", total_items=3)
        assert batch["status"] == "pending"
        assert batch["completed_items"] == 0
        assert batch["total_items"] == 3
        assert batch["filename"] is None

    def test_create_batch_stores_row_retrievable_by_get(self):
        batch = mock_db.create_batch(source="csv", total_items=2, filename="in.csv")
        fetched = mock_db.get_batch(batch["id"])
        assert fetched is not None
        assert fetched["filename"] == "in.csv"

    def test_get_batch_unknown_id_returns_none(self):
        assert mock_db.get_batch("nonexistent") is None

    def test_update_batch_status_flips_status(self):
        batch = mock_db.create_batch(source="paste", total_items=1)
        updated = mock_db.update_batch_status(batch["id"], "done")
        assert updated["status"] == "done"
        assert mock_db.get_batch(batch["id"])["status"] == "done"

    def test_update_batch_status_unknown_id_returns_none(self):
        assert mock_db.update_batch_status("nonexistent", "done") is None

    def test_increment_batch_completed_increments_by_one_each_call(self):
        batch = mock_db.create_batch(source="paste", total_items=5)
        mock_db.increment_batch_completed(batch["id"])
        result = mock_db.increment_batch_completed(batch["id"])
        assert result["completed_items"] == 2

    def test_increment_batch_completed_unknown_id_returns_none(self):
        assert mock_db.increment_batch_completed("nonexistent") is None

    def test_list_batches_sorted_newest_first(self):
        first = mock_db.create_batch(source="paste", total_items=1)
        second = mock_db.create_batch(source="csv", total_items=1)
        rows = mock_db.list_batches()
        assert [r["id"] for r in rows] == [second["id"], first["id"]]

    def test_list_batches_respects_limit(self):
        for _ in range(3):
            mock_db.create_batch(source="paste", total_items=1)
        assert len(mock_db.list_batches(limit=2)) == 2

    def test_create_batch_items_assigns_1_based_seq_in_order(self):
        batch = mock_db.create_batch(source="paste", total_items=2)
        items = mock_db.create_batch_items(batch["id"], ["first text", "second text"])
        assert [i["seq"] for i in items] == [1, 2]
        assert items[0]["text_content"] == "first text"
        assert items[0]["status"] == "pending"

    def test_get_batch_item_returns_stored_row(self):
        batch = mock_db.create_batch(source="paste", total_items=1)
        [item] = mock_db.create_batch_items(batch["id"], ["only text"])
        fetched = mock_db.get_batch_item(item["id"])
        assert fetched["text_content"] == "only text"

    def test_list_batch_items_paginates_and_orders_by_seq(self):
        batch = mock_db.create_batch(source="paste", total_items=3)
        mock_db.create_batch_items(batch["id"], ["a", "b", "c"])
        page1 = mock_db.list_batch_items(batch["id"], offset=0, limit=2)
        page2 = mock_db.list_batch_items(batch["id"], offset=2, limit=2)
        assert [i["seq"] for i in page1] == [1, 2]
        assert [i["seq"] for i in page2] == [3]

    def test_list_batch_items_only_returns_items_for_that_batch(self):
        batch_a = mock_db.create_batch(source="paste", total_items=1)
        batch_b = mock_db.create_batch(source="paste", total_items=1)
        mock_db.create_batch_items(batch_a["id"], ["a"])
        mock_db.create_batch_items(batch_b["id"], ["b"])
        assert len(mock_db.list_batch_items(batch_a["id"])) == 1

    def test_list_all_batch_items_returns_every_row_unsliced(self):
        batch = mock_db.create_batch(source="paste", total_items=5)
        mock_db.create_batch_items(batch["id"], ["a", "b", "c", "d", "e"])
        assert len(mock_db.list_all_batch_items(batch["id"])) == 5


# ===========================================================================
# repositories/batches.py — mock-mode fallbacks
# ===========================================================================

class TestBatchesRepo:

    def test_insert_persists_and_is_retrievable_via_get(self):
        batch = batches_repo.insert(source="paste", total_items=4)
        fetched = batches_repo.get(batch["id"])
        assert fetched is not None
        assert fetched["total_items"] == 4

    def test_get_unknown_id_returns_none(self):
        assert batches_repo.get("nonexistent") is None

    def test_update_status_flips_status(self):
        batch = batches_repo.insert(source="paste", total_items=1)
        updated = batches_repo.update_status(batch["id"], "processing")
        assert updated["status"] == "processing"

    def test_increment_completed_increments(self):
        batch = batches_repo.insert(source="paste", total_items=2)
        batches_repo.increment_completed(batch["id"])
        result = batches_repo.increment_completed(batch["id"])
        assert result["completed_items"] == 2

    def test_list_all_sorted_newest_first_and_respects_limit(self):
        first = batches_repo.insert(source="paste", total_items=1)
        second = batches_repo.insert(source="paste", total_items=1)
        rows = batches_repo.list_all(limit=1)
        assert len(rows) == 1
        assert rows[0]["id"] == second["id"]
        assert first["id"] != second["id"]


# ===========================================================================
# repositories/batch_items.py — mock-mode fallbacks
# ===========================================================================

class TestBatchItemsRepo:

    def test_insert_many_persists_all_rows(self):
        batch = batches_repo.insert(source="paste", total_items=2)
        rows = batch_items_repo.insert_many(batch["id"], ["one", "two"])
        assert len(rows) == 2
        assert batch_items_repo.get(rows[0]["id"])["text_content"] == "one"

    def test_get_unknown_id_returns_none(self):
        assert batch_items_repo.get("nonexistent") is None

    def test_list_by_batch_paginates(self):
        batch = batches_repo.insert(source="paste", total_items=3)
        batch_items_repo.insert_many(batch["id"], ["a", "b", "c"])
        page = batch_items_repo.list_by_batch(batch["id"], offset=1, limit=1)
        assert len(page) == 1
        assert page[0]["seq"] == 2

    def test_list_all_by_batch_returns_everything_unpaginated(self):
        batch = batches_repo.insert(source="paste", total_items=3)
        batch_items_repo.insert_many(batch["id"], ["a", "b", "c"])
        assert len(batch_items_repo.list_all_by_batch(batch["id"])) == 3


# ===========================================================================
# POST /batches/paste
# ===========================================================================

class TestSubmitBatchPaste:

    def test_happy_path_returns_batch_id_and_total_items(self, client):
        resp = client.post("/batches/paste", json={"texts": ["first text", "second text"]})
        assert resp.status_code == 202
        data = resp.json()
        assert "batch_id" in data
        assert data["total_items"] == 2

    def test_creates_pending_batch_items(self, client):
        client.post("/batches/paste", json={"texts": ["only text here"]})
        assert len(mock_db._batch_items) == 1
        item = list(mock_db._batch_items.values())[0]
        assert item["status"] == "pending"
        assert item["text_content"] == "only text here"

    def test_empty_list_returns_422(self, client):
        resp = client.post("/batches/paste", json={"texts": []})
        assert resp.status_code == 422

    def test_all_blank_strings_returns_422(self, client):
        resp = client.post("/batches/paste", json={"texts": ["   ", ""]})
        assert resp.status_code == 422

    def test_mixed_blank_and_non_blank_drops_only_blanks(self, client):
        """2 blank + 3 non-blank entries must produce exactly 3 items, not 5."""
        resp = client.post(
            "/batches/paste",
            json={"texts": ["  ", "first", "", "second", "third"]},
        )

        assert resp.status_code == 202
        assert resp.json()["total_items"] == 3
        assert len(mock_db._batch_items) == 3
        texts = {item["text_content"] for item in mock_db._batch_items.values()}
        assert texts == {"first", "second", "third"}


# ===========================================================================
# POST /batches/csv
# ===========================================================================

class TestSubmitBatchCsv:

    def test_happy_path_with_text_column(self, client):
        csv_content = "text\nfirst row\nsecond row\n"
        resp = client.post(
            "/batches/csv",
            files={"file": ("in.csv", io.BytesIO(csv_content.encode()), "text/csv")},
        )
        assert resp.status_code == 202
        assert resp.json()["total_items"] == 2

    def test_single_column_fallback_when_not_named_text(self, client):
        csv_content = "document\nrow one\nrow two\nrow three\n"
        resp = client.post(
            "/batches/csv",
            files={"file": ("in.csv", io.BytesIO(csv_content.encode()), "text/csv")},
        )
        assert resp.status_code == 202
        assert resp.json()["total_items"] == 3

    def test_ambiguous_multi_column_without_text_returns_422(self, client):
        csv_content = "foo,bar\nrow one,x\nrow two,y\n"
        resp = client.post(
            "/batches/csv",
            files={"file": ("in.csv", io.BytesIO(csv_content.encode()), "text/csv")},
        )
        assert resp.status_code == 422

    def test_stores_filename_on_batch(self, client):
        csv_content = "text\nonly row\n"
        client.post(
            "/batches/csv",
            files={"file": ("uploaded.csv", io.BytesIO(csv_content.encode()), "text/csv")},
        )
        batch = list(mock_db._batches.values())[0]
        assert batch["filename"] == "uploaded.csv"


# ===========================================================================
# GET /batches/{id}
# ===========================================================================

class TestGetBatch:

    def test_unknown_id_returns_404(self, client):
        resp = client.get("/batches/nonexistent")
        assert resp.status_code == 404

    def test_returns_expected_shape(self, client):
        create_resp = client.post("/batches/paste", json={"texts": ["a", "b"]})
        batch_id = create_resp.json()["batch_id"]

        resp = client.get(f"/batches/{batch_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["batch_id"] == batch_id
        assert data["status"] == "pending"
        assert data["total_items"] == 2
        assert data["completed_items"] == 0


# ===========================================================================
# GET /batches/{id}/items
# ===========================================================================

class TestListBatchItems:

    def test_unknown_batch_returns_404(self, client):
        resp = client.get("/batches/nonexistent/items")
        assert resp.status_code == 404

    def test_pagination(self, client):
        create_resp = client.post("/batches/paste", json={"texts": ["a", "b", "c"]})
        batch_id = create_resp.json()["batch_id"]

        resp = client.get(f"/batches/{batch_id}/items", params={"page": 1, "page_size": 2})

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert data["page"] == 1
        assert data["page_size"] == 2
        assert len(data["items"]) == 2
        assert data["items"][0]["seq"] == 1

    def test_item_shape(self, client):
        create_resp = client.post("/batches/paste", json={"texts": ["only text"]})
        batch_id = create_resp.json()["batch_id"]

        resp = client.get(f"/batches/{batch_id}/items")

        item = resp.json()["items"][0]
        assert item["text_content"] == "only text"
        assert item["status"] == "pending"
        assert item["predicted_label"] is None


# ===========================================================================
# GET /batches/{id}/export
# ===========================================================================

class TestExportBatch:

    def test_unknown_batch_returns_404(self, client):
        resp = client.get("/batches/nonexistent/export")
        assert resp.status_code == 404

    def test_returns_csv_with_attachment_header(self, client):
        create_resp = client.post("/batches/paste", json={"texts": ["export me"]})
        batch_id = create_resp.json()["batch_id"]

        resp = client.get(f"/batches/{batch_id}/export")

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")
        assert f'batch_{batch_id}.csv' in resp.headers["content-disposition"]
        assert "export me" in resp.text
