"""Tests for question set endpoints (/question-sets)."""

from unittest.mock import AsyncMock, patch

from src.core.config import settings


def _make_truth_payload():
    """Build a mock TruthPayload for testing."""
    from src.schemas.truth import (
        AnswerTruth,
        RetrievalTruth,
        TruthMetadata,
        TruthPayload,
    )

    return TruthPayload(
        answer_truth=AnswerTruth(required_concepts=["concept A", "concept B"]),
        retrieval_truth=RetrievalTruth(
            required_documents=["doc.pdf"],
            expected_chunk_refs=["chunk:1"],
            evidence_mode="grounded_from_manual_answer",
        ),
        metadata=TruthMetadata(
            generated_by_model="test-judge",
            generated_at="2026-01-01T00:00:00",
            source_chunk_ids=[1],
        ),
    )


# --- CRUD tests ---


def test_create_question_set(client):
    """Should create a question set and return its data."""
    response = client.post(
        "/question-sets/",
        json={
            "name": "Test Set",
            "questions": [{"question": "What is AI?"}],
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Set"
    assert len(data["questions"]) == 1
    assert data["questions"][0]["question"] == "What is AI?"


def test_list_question_sets(client):
    """Should list created question sets."""
    client.post(
        "/question-sets/",
        json={"name": "Set A", "questions": [{"question": "Q1"}]},
    )
    response = client.get("/question-sets/")
    assert response.status_code == 200
    assert len(response.json()) >= 1


def test_get_question_set_by_id(client):
    """Should return a single question set by ID."""
    create_resp = client.post(
        "/question-sets/",
        json={"name": "By ID", "questions": [{"question": "Q1"}]},
    )
    qs_id = create_resp.json()["id"]
    response = client.get(f"/question-sets/{qs_id}")
    assert response.status_code == 200
    assert response.json()["id"] == qs_id


def test_get_question_set_not_found(client):
    """Should return 404 for non-existent question set."""
    response = client.get("/question-sets/999")
    assert response.status_code == 404


def test_delete_question_set(client):
    """Should delete a question set."""
    create_resp = client.post(
        "/question-sets/",
        json={"name": "Delete Me", "questions": [{"question": "Q1"}]},
    )
    qs_id = create_resp.json()["id"]
    del_resp = client.delete(f"/question-sets/{qs_id}")
    assert del_resp.status_code == 204

    get_resp = client.get(f"/question-sets/{qs_id}")
    assert get_resp.status_code == 404


def test_delete_question_set_cascades_eval_runs(client):
    """Should delete associated eval runs and results when question set is deleted."""
    create_resp = client.post(
        "/question-sets/",
        json={"name": "Cascade Set", "questions": [{"question": "Q1"}]},
    )
    qs_id = create_resp.json()["id"]

    with patch("src.routes.evaluation._run_evaluation", new_callable=AsyncMock):
        eval_resp = client.post(
            "/evaluations/",
            json={
                "model_name": "test-model",
                "question_set_id": qs_id,
                "questions": [{"question": "Q1"}],
            },
        )
    eval_id = eval_resp.json()["eval_run_id"]

    del_resp = client.delete(f"/question-sets/{qs_id}")
    assert del_resp.status_code == 204

    get_eval = client.get(f"/evaluations/{eval_id}")
    assert get_eval.status_code == 404


# --- PATCH tests ---


def test_update_question_set_name(client):
    """Should update only the name when questions are not provided."""
    create_resp = client.post(
        "/question-sets/",
        json={"name": "Original", "questions": [{"question": "Q1"}]},
    )
    qs_id = create_resp.json()["id"]

    patch_resp = client.patch(f"/question-sets/{qs_id}", json={"name": "Renamed"})
    assert patch_resp.status_code == 200
    data = patch_resp.json()
    assert data["name"] == "Renamed"
    assert data["questions"][0]["question"] == "Q1"


def test_update_question_set_questions(client):
    """Should update only the questions when name is not provided."""
    create_resp = client.post(
        "/question-sets/",
        json={"name": "Keep Name", "questions": [{"question": "Old Q"}]},
    )
    qs_id = create_resp.json()["id"]

    patch_resp = client.patch(
        f"/question-sets/{qs_id}",
        json={"questions": [{"question": "New Q1"}, {"question": "New Q2"}]},
    )
    assert patch_resp.status_code == 200
    data = patch_resp.json()
    assert data["name"] == "Keep Name"
    assert len(data["questions"]) == 2
    assert data["questions"][0]["question"] == "New Q1"


def test_update_question_set_both(client):
    """Should update both name and questions when both provided."""
    create_resp = client.post(
        "/question-sets/",
        json={"name": "Old", "questions": [{"question": "Old Q"}]},
    )
    qs_id = create_resp.json()["id"]

    patch_resp = client.patch(
        f"/question-sets/{qs_id}",
        json={"name": "New", "questions": [{"question": "New Q"}]},
    )
    assert patch_resp.status_code == 200
    data = patch_resp.json()
    assert data["name"] == "New"
    assert data["questions"][0]["question"] == "New Q"


def test_update_question_set_not_found(client):
    """Should return 404 for non-existent question set."""
    response = client.patch("/question-sets/999", json={"name": "Nope"})
    assert response.status_code == 404


def test_update_question_set_preserves_existing_truth(client, monkeypatch):
    """Should preserve truth payloads on questions that already have them."""
    monkeypatch.setattr(settings, "JUDGE_MODEL_NAME", "test-judge", raising=False)
    truth = _make_truth_payload()

    with patch(
        "src.routes.question_sets.generate_truth_from_manual_answer",
        new_callable=AsyncMock,
        return_value=truth,
    ):
        create_resp = client.post(
            "/question-sets/",
            json={
                "name": "Truth Set",
                "questions": [
                    {"question": "What is AI?", "expected_answer": "Artificial intelligence."},
                ],
            },
        )

    qs_id = create_resp.json()["id"]
    created_truth = create_resp.json()["questions"][0]["truth"]

    with patch(
        "src.routes.question_sets.generate_truth_from_manual_answer",
        new_callable=AsyncMock,
    ) as mock_gen:
        patch_resp = client.patch(
            f"/question-sets/{qs_id}",
            json={
                "questions": [
                    {
                        "question": "What is AI?",
                        "expected_answer": "Artificial intelligence.",
                        "truth": created_truth,
                    },
                ],
            },
        )

    assert patch_resp.status_code == 200
    mock_gen.assert_not_called()
    assert patch_resp.json()["questions"][0]["truth"] is not None


def test_update_question_set_includes_updated_at(client):
    """Should return updated_at in the response."""
    create_resp = client.post(
        "/question-sets/",
        json={"name": "Timestamps", "questions": [{"question": "Q1"}]},
    )
    qs_id = create_resp.json()["id"]

    patch_resp = client.patch(f"/question-sets/{qs_id}", json={"name": "Updated"})
    assert patch_resp.status_code == 200
    assert "updated_at" in patch_resp.json()


# --- Truth generation tests ---


def test_create_question_set_generates_truth(client, monkeypatch):
    """Should generate truth for questions with expected answers."""
    monkeypatch.setattr(settings, "JUDGE_MODEL_NAME", "test-judge", raising=False)

    truth = _make_truth_payload()

    with patch(
        "src.routes.question_sets.generate_truth_from_manual_answer",
        new_callable=AsyncMock,
        return_value=truth,
    ) as mock_gen:
        response = client.post(
            "/question-sets/",
            json={
                "name": "Truth Set",
                "questions": [
                    {"question": "What is AI?", "expected_answer": "Artificial intelligence."},
                ],
            },
        )

    assert response.status_code == 201
    mock_gen.assert_called_once()
    data = response.json()
    q = data["questions"][0]
    assert q["truth"] is not None
    assert q["truth"]["answer_truth"]["required_concepts"] == ["concept A", "concept B"]
    assert q["truth"]["retrieval_truth"]["evidence_mode"] == "grounded_from_manual_answer"


def test_create_question_set_skips_truth_when_no_judge(client, monkeypatch):
    """Should not generate truth when no judge model is configured."""
    monkeypatch.setattr(settings, "JUDGE_MODEL_NAME", "", raising=False)
    monkeypatch.setattr(settings, "MODEL_A_NAME", "", raising=False)
    monkeypatch.setattr(settings, "MODEL_B_NAME", "", raising=False)

    with patch(
        "src.routes.question_sets.generate_truth_from_manual_answer",
        new_callable=AsyncMock,
    ) as mock_gen:
        response = client.post(
            "/question-sets/",
            json={
                "name": "No Judge Set",
                "questions": [
                    {"question": "What is AI?", "expected_answer": "Artificial intelligence."},
                ],
            },
        )

    assert response.status_code == 201
    mock_gen.assert_not_called()


def test_create_question_set_skips_truth_without_expected_answer(client, monkeypatch):
    """Should not generate truth for questions without expected answers."""
    monkeypatch.setattr(settings, "JUDGE_MODEL_NAME", "test-judge", raising=False)

    with patch(
        "src.routes.question_sets.generate_truth_from_manual_answer",
        new_callable=AsyncMock,
    ) as mock_gen:
        response = client.post(
            "/question-sets/",
            json={
                "name": "No Expected Set",
                "questions": [{"question": "What is AI?"}],
            },
        )

    assert response.status_code == 201
    mock_gen.assert_not_called()


def test_create_question_set_graceful_on_truth_failure(client, monkeypatch):
    """Should save question set even when truth generation fails."""
    monkeypatch.setattr(settings, "JUDGE_MODEL_NAME", "test-judge", raising=False)

    with patch(
        "src.routes.question_sets.generate_truth_from_manual_answer",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM unavailable"),
    ):
        response = client.post(
            "/question-sets/",
            json={
                "name": "Failure Set",
                "questions": [
                    {"question": "What is AI?", "expected_answer": "Artificial intelligence."},
                ],
            },
        )

    assert response.status_code == 201
    q = response.json()["questions"][0]
    assert q.get("truth") is None
    assert q["expected_answer"] == "Artificial intelligence."
