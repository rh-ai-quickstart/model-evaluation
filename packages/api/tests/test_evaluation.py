"""Tests for evaluation endpoints (/evaluations)."""

from unittest.mock import AsyncMock, patch

from src.core.config import settings

# --- Tests ---


def test_create_eval_run(client):
    """Should create an evaluation run and return its ID."""
    # Mock background task so it doesn't actually run
    with patch("src.routes.evaluation._run_evaluation"):
        response = client.post(
            "/evaluations/",
            json={
                "model_name": "granite-3.1-8b-instruct",
                "questions": ["What is AI?", "Explain RAG."],
            },
        )

    assert response.status_code == 201
    data = response.json()
    assert data["eval_run_id"] >= 1
    assert data["model_name"] == "granite-3.1-8b-instruct"
    assert data["status"] == "pending"
    assert data["total_questions"] == 2
    assert "2 questions" in data["message"]


def test_create_eval_run_validates_empty_questions(client):
    """Should reject empty questions list."""
    response = client.post(
        "/evaluations/",
        json={"model_name": "granite-3.1-8b-instruct", "questions": []},
    )
    assert response.status_code == 422


def test_list_eval_runs_empty(client):
    """Should return empty list when no runs exist."""
    response = client.get("/evaluations/")
    assert response.status_code == 200
    assert response.json() == []


def test_list_eval_runs_after_create(client):
    """Should include created run in list."""
    with patch("src.routes.evaluation._run_evaluation"):
        client.post(
            "/evaluations/",
            json={
                "model_name": "granite-3.1-8b-instruct",
                "questions": ["What is AI?"],
            },
        )

    response = client.get("/evaluations/")
    assert response.status_code == 200
    runs = response.json()
    assert len(runs) == 1
    assert runs[0]["model_name"] == "granite-3.1-8b-instruct"


def test_get_eval_run_by_id(client):
    """Should return eval run with its results."""
    with patch("src.routes.evaluation._run_evaluation"):
        create_resp = client.post(
            "/evaluations/",
            json={
                "model_name": "granite-3.1-8b-instruct",
                "questions": ["What is AI?"],
            },
        )
    run_id = create_resp.json()["eval_run_id"]

    response = client.get(f"/evaluations/{run_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == run_id
    assert data["model_name"] == "granite-3.1-8b-instruct"
    assert "results" in data


def test_get_eval_run_not_found(client):
    """Should return 404 for non-existent eval run."""
    response = client.get("/evaluations/999")
    assert response.status_code == 404


# --- Rerun tests ---


def test_rerun_creates_new_run_with_same_questions(client, _setup_db):
    """Should create a new run copying questions from the original."""
    with patch("src.routes.evaluation._run_evaluation"):
        create_resp = client.post(
            "/evaluations/",
            json={
                "model_name": "granite-3.1-8b-instruct",
                "questions": ["What is AI?", "Explain RAG."],
            },
        )
    original_id = create_resp.json()["eval_run_id"]

    # Seed EvalResult rows since _run_evaluation is mocked and won't create them
    import asyncio

    from db import EvalResult

    _, async_session = _setup_db

    async def _seed_results():
        async with async_session() as session:
            for q in ["What is AI?", "Explain RAG."]:
                session.add(EvalResult(eval_run_id=original_id, question=q))
            await session.commit()

    asyncio.run(_seed_results())

    with patch("src.routes.evaluation._run_evaluation"):
        rerun_resp = client.post(
            f"/evaluations/{original_id}/rerun",
            json={"model_name": "llama-3.1-8b-instruct"},
        )

    assert rerun_resp.status_code == 201
    data = rerun_resp.json()
    assert data["model_name"] == "llama-3.1-8b-instruct"
    assert data["total_questions"] == 2
    assert f"run #{original_id}" in data["message"]
    assert data["eval_run_id"] != original_id


def test_rerun_not_found(client):
    """Should return 404 when original run does not exist."""
    response = client.post(
        "/evaluations/999/rerun",
        json={"model_name": "llama-3.1-8b-instruct"},
    )
    assert response.status_code == 404


# --- Compare tests ---


def test_compare_two_runs(client):
    """Should return side-by-side comparison of two runs."""
    with patch("src.routes.evaluation._run_evaluation"):
        resp_a = client.post(
            "/evaluations/",
            json={
                "model_name": "granite-3.1-8b-instruct",
                "questions": ["What is AI?"],
            },
        )
        resp_b = client.post(
            "/evaluations/",
            json={
                "model_name": "llama-3.1-8b-instruct",
                "questions": ["What is AI?"],
            },
        )

    id_a = resp_a.json()["eval_run_id"]
    id_b = resp_b.json()["eval_run_id"]

    response = client.get(f"/evaluations/compare?run_a_id={id_a}&run_b_id={id_b}")
    assert response.status_code == 200
    data = response.json()
    assert data["run_a"]["id"] == id_a
    assert data["run_b"]["id"] == id_b
    assert len(data["metrics"]) == 13
    assert data["metrics"][0]["metric"] == "groundedness"


def test_compare_not_found(client):
    """Should return 404 when a run does not exist."""
    with patch("src.routes.evaluation._run_evaluation"):
        resp = client.post(
            "/evaluations/",
            json={
                "model_name": "granite-3.1-8b-instruct",
                "questions": ["What is AI?"],
            },
        )
    run_id = resp.json()["eval_run_id"]

    response = client.get(f"/evaluations/compare?run_a_id={run_id}&run_b_id=999")
    assert response.status_code == 404


# --- Inline truth generation tests ---


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


def test_create_eval_run_generates_truth_for_inline_questions(client, monkeypatch):
    """Should generate truth for inline questions with expected answers."""
    monkeypatch.setattr(settings, "JUDGE_MODEL_NAME", "test-judge", raising=False)

    truth = _make_truth_payload()

    with (
        patch("src.routes.evaluation._run_evaluation"),
        patch(
            "src.routes.evaluation.generate_truth_from_manual_answer",
            new_callable=AsyncMock,
            return_value=truth,
        ) as mock_gen,
    ):
        response = client.post(
            "/evaluations/",
            json={
                "model_name": "granite-3.1-8b-instruct",
                "questions": [
                    {"question": "What is AI?", "expected_answer": "Artificial intelligence."},
                ],
            },
        )

    assert response.status_code == 201
    mock_gen.assert_called_once()


def test_create_eval_run_skips_truth_when_no_judge(client, monkeypatch):
    """Should not generate truth when no judge model is configured."""
    monkeypatch.setattr(settings, "JUDGE_MODEL_NAME", "", raising=False)
    monkeypatch.setattr(settings, "MODEL_A_NAME", "", raising=False)
    monkeypatch.setattr(settings, "MODEL_B_NAME", "", raising=False)

    with (
        patch("src.routes.evaluation._run_evaluation"),
        patch(
            "src.routes.evaluation.generate_truth_from_manual_answer",
            new_callable=AsyncMock,
        ) as mock_gen,
    ):
        response = client.post(
            "/evaluations/",
            json={
                "model_name": "granite-3.1-8b-instruct",
                "questions": [
                    {"question": "What is AI?", "expected_answer": "Artificial intelligence."},
                ],
            },
        )

    assert response.status_code == 201
    mock_gen.assert_not_called()


def test_create_eval_run_skips_truth_when_already_present(client, monkeypatch):
    """Should not regenerate truth when question already has truth."""
    monkeypatch.setattr(settings, "JUDGE_MODEL_NAME", "test-judge", raising=False)

    truth_dict = _make_truth_payload().model_dump(mode="json")

    with (
        patch("src.routes.evaluation._run_evaluation"),
        patch(
            "src.routes.evaluation.generate_truth_from_manual_answer",
            new_callable=AsyncMock,
        ) as mock_gen,
    ):
        response = client.post(
            "/evaluations/",
            json={
                "model_name": "granite-3.1-8b-instruct",
                "questions": [
                    {
                        "question": "What is AI?",
                        "expected_answer": "Artificial intelligence.",
                        "truth": truth_dict,
                    },
                ],
            },
        )

    assert response.status_code == 201
    mock_gen.assert_not_called()


def test_create_eval_run_graceful_on_truth_failure(client, monkeypatch):
    """Should still create run when truth generation fails."""
    monkeypatch.setattr(settings, "JUDGE_MODEL_NAME", "test-judge", raising=False)

    with (
        patch("src.routes.evaluation._run_evaluation"),
        patch(
            "src.routes.evaluation.generate_truth_from_manual_answer",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM unavailable"),
        ),
    ):
        response = client.post(
            "/evaluations/",
            json={
                "model_name": "granite-3.1-8b-instruct",
                "questions": [
                    {"question": "What is AI?", "expected_answer": "Artificial intelligence."},
                ],
            },
        )

    assert response.status_code == 201
    data = response.json()
    assert data["total_questions"] == 1


# --- Run metadata capture tests ---


def test_eval_run_captures_judge_model(client, monkeypatch):
    """Should store judge_model_name on the eval run at creation time."""
    monkeypatch.setattr(settings, "JUDGE_MODEL_NAME", "test-judge", raising=False)

    with patch("src.routes.evaluation._run_evaluation"):
        create_resp = client.post(
            "/evaluations/",
            json={
                "model_name": "granite-3.1-8b-instruct",
                "questions": ["What is AI?"],
            },
        )
    run_id = create_resp.json()["eval_run_id"]

    detail = client.get(f"/evaluations/{run_id}")
    assert detail.status_code == 200
    data = detail.json()
    assert data["judge_model_name"] == "test-judge"


def test_eval_run_captures_corpus_snapshot(client):
    """Should store corpus_snapshot on the eval run at creation time."""
    with patch("src.routes.evaluation._run_evaluation"):
        create_resp = client.post(
            "/evaluations/",
            json={
                "model_name": "granite-3.1-8b-instruct",
                "questions": ["What is AI?"],
            },
        )
    run_id = create_resp.json()["eval_run_id"]

    detail = client.get(f"/evaluations/{run_id}")
    data = detail.json()
    snapshot = data["corpus_snapshot"]
    assert snapshot is not None
    assert "documents" in snapshot
    assert "total_documents" in snapshot
    assert "total_chunks" in snapshot
    assert isinstance(snapshot["documents"], list)


def test_eval_run_captures_retrieval_config_with_profile(client):
    """Should store retrieval_config when a profile is specified."""
    with patch("src.routes.evaluation._run_evaluation"):
        create_resp = client.post(
            "/evaluations/",
            json={
                "model_name": "granite-3.1-8b-instruct",
                "questions": ["What is AI?"],
                "profile_id": "fsi_compliance_v1",
            },
        )
    run_id = create_resp.json()["eval_run_id"]

    detail = client.get(f"/evaluations/{run_id}")
    data = detail.json()
    assert data["retrieval_config"] is not None
    assert "top_k" in data["retrieval_config"]
    assert data["profile_version"] is not None


def test_eval_run_no_retrieval_config_without_profile(client):
    """Should have null retrieval_config when no profile is specified."""
    with patch("src.routes.evaluation._run_evaluation"):
        create_resp = client.post(
            "/evaluations/",
            json={
                "model_name": "granite-3.1-8b-instruct",
                "questions": ["What is AI?"],
            },
        )
    run_id = create_resp.json()["eval_run_id"]

    detail = client.get(f"/evaluations/{run_id}")
    data = detail.json()
    assert data["retrieval_config"] is None


# --- Comparison warning tests ---


def test_compare_warns_on_judge_model_mismatch(client, monkeypatch):
    """Should warn when runs used different judge models."""
    with patch("src.routes.evaluation._run_evaluation"):
        monkeypatch.setattr(settings, "JUDGE_MODEL_NAME", "judge-a", raising=False)
        resp_a = client.post(
            "/evaluations/",
            json={"model_name": "model-a", "questions": ["Q?"]},
        )
        monkeypatch.setattr(settings, "JUDGE_MODEL_NAME", "judge-b", raising=False)
        resp_b = client.post(
            "/evaluations/",
            json={"model_name": "model-b", "questions": ["Q?"]},
        )

    id_a = resp_a.json()["eval_run_id"]
    id_b = resp_b.json()["eval_run_id"]

    response = client.get(f"/evaluations/compare?run_a_id={id_a}&run_b_id={id_b}")
    data = response.json()
    warning_codes = [w["code"] for w in data["warnings"]]
    assert "JUDGE_MODEL_MISMATCH" in warning_codes


def test_compare_warns_on_corpus_change(client, _setup_db):
    """Should warn when corpus snapshots differ between runs."""
    from db import Document

    engine, async_session = _setup_db

    # Create run A with empty corpus
    with patch("src.routes.evaluation._run_evaluation"):
        resp_a = client.post(
            "/evaluations/",
            json={"model_name": "model-a", "questions": ["Q?"]},
        )

    # Add a document so the corpus changes
    import asyncio

    async def _add_doc():
        async with async_session() as session:
            session.add(Document(filename="new.pdf", status="ready", chunk_count=5))
            await session.commit()

    asyncio.run(_add_doc())

    # Create run B with different corpus
    with patch("src.routes.evaluation._run_evaluation"):
        resp_b = client.post(
            "/evaluations/",
            json={"model_name": "model-b", "questions": ["Q?"]},
        )

    id_a = resp_a.json()["eval_run_id"]
    id_b = resp_b.json()["eval_run_id"]

    response = client.get(f"/evaluations/compare?run_a_id={id_a}&run_b_id={id_b}")
    data = response.json()
    warning_codes = [w["code"] for w in data["warnings"]]
    assert "CORPUS_MISMATCH" in warning_codes


def test_compare_no_warnings_when_same_conditions(client, monkeypatch):
    """Should have no metadata warnings when runs share same conditions."""
    monkeypatch.setattr(settings, "JUDGE_MODEL_NAME", "same-judge", raising=False)

    with patch("src.routes.evaluation._run_evaluation"):
        resp_a = client.post(
            "/evaluations/",
            json={"model_name": "model-a", "questions": ["Q?"]},
        )
        resp_b = client.post(
            "/evaluations/",
            json={"model_name": "model-b", "questions": ["Q?"]},
        )

    id_a = resp_a.json()["eval_run_id"]
    id_b = resp_b.json()["eval_run_id"]

    response = client.get(f"/evaluations/compare?run_a_id={id_a}&run_b_id={id_b}")
    data = response.json()
    warning_codes = [w["code"] for w in data["warnings"]]
    # No metadata mismatch warnings expected
    assert "JUDGE_MODEL_MISMATCH" not in warning_codes
    assert "CORPUS_MISMATCH" not in warning_codes


# --- Truth persistence tests ---


def test_eval_result_response_includes_truth_field(client):
    """Should include truth in eval result response schema."""
    with patch("src.routes.evaluation._run_evaluation"):
        create_resp = client.post(
            "/evaluations/",
            json={
                "model_name": "granite-3.1-8b-instruct",
                "questions": ["What is AI?"],
            },
        )
    run_id = create_resp.json()["eval_run_id"]

    detail = client.get(f"/evaluations/{run_id}")
    assert detail.status_code == 200
    data = detail.json()
    # No results yet (background task is mocked), but schema is valid
    assert "results" in data


def test_truth_persisted_on_eval_result(client, _setup_db):
    """Should persist truth_payload on EvalResult when stored."""
    import asyncio

    from db import EvalResult

    _, async_session = _setup_db

    truth_data = _make_truth_payload().model_dump(mode="json")

    with patch("src.routes.evaluation._run_evaluation"):
        create_resp = client.post(
            "/evaluations/",
            json={
                "model_name": "granite-3.1-8b-instruct",
                "questions": ["What is AI?"],
            },
        )
    run_id = create_resp.json()["eval_run_id"]

    # Manually seed an EvalResult with truth_payload
    async def _seed():
        async with async_session() as session:
            session.add(
                EvalResult(
                    eval_run_id=run_id,
                    question="What is AI?",
                    expected_answer="Artificial intelligence.",
                    answer="AI is artificial intelligence.",
                    truth_payload=truth_data,
                )
            )
            await session.commit()

    asyncio.run(_seed())

    detail = client.get(f"/evaluations/{run_id}")
    data = detail.json()
    assert len(data["results"]) == 1
    result = data["results"][0]
    assert result["truth"] is not None
    assert "answer_truth" in result["truth"]
    assert "retrieval_truth" in result["truth"]
    assert "metadata" in result["truth"]
    assert result["truth"]["answer_truth"]["required_concepts"] == ["concept A", "concept B"]


# --- Rerun truth and metadata tests ---


def test_rerun_copies_truth_from_original(client, _setup_db):
    """Should copy truth_payload from original results to rerun questions."""
    import asyncio

    from db import EvalResult

    _, async_session = _setup_db

    truth_data = _make_truth_payload().model_dump(mode="json")

    with patch("src.routes.evaluation._run_evaluation"):
        create_resp = client.post(
            "/evaluations/",
            json={
                "model_name": "granite-3.1-8b-instruct",
                "questions": ["What is AI?"],
            },
        )
    original_id = create_resp.json()["eval_run_id"]

    # Seed original results with truth
    async def _seed():
        async with async_session() as session:
            session.add(
                EvalResult(
                    eval_run_id=original_id,
                    question="What is AI?",
                    expected_answer="Artificial intelligence.",
                    truth_payload=truth_data,
                )
            )
            await session.commit()

    asyncio.run(_seed())

    with patch("src.routes.evaluation._run_evaluation") as mock_run:
        rerun_resp = client.post(
            f"/evaluations/{original_id}/rerun",
            json={"model_name": "llama-3.1-8b-instruct"},
        )

    assert rerun_resp.status_code == 201

    # Verify the questions passed to _run_evaluation have truth
    call_kwargs = mock_run.call_args
    questions = call_kwargs.kwargs.get("questions") or call_kwargs[1].get("questions")
    assert questions[0].truth is not None
    assert questions[0].truth.answer_truth.required_concepts == ["concept A", "concept B"]


def test_rerun_captures_run_metadata(client, _setup_db, monkeypatch):
    """Should snapshot judge model and corpus on rerun like create_eval_run."""
    import asyncio

    from db import EvalResult

    _, async_session = _setup_db
    monkeypatch.setattr(settings, "JUDGE_MODEL_NAME", "test-judge", raising=False)

    with patch("src.routes.evaluation._run_evaluation"):
        create_resp = client.post(
            "/evaluations/",
            json={
                "model_name": "granite-3.1-8b-instruct",
                "questions": ["What is AI?"],
            },
        )
    original_id = create_resp.json()["eval_run_id"]

    async def _seed():
        async with async_session() as session:
            session.add(
                EvalResult(
                    eval_run_id=original_id,
                    question="What is AI?",
                )
            )
            await session.commit()

    asyncio.run(_seed())

    with patch("src.routes.evaluation._run_evaluation"):
        rerun_resp = client.post(
            f"/evaluations/{original_id}/rerun",
            json={"model_name": "llama-3.1-8b-instruct"},
        )

    rerun_id = rerun_resp.json()["eval_run_id"]
    detail = client.get(f"/evaluations/{rerun_id}")
    data = detail.json()
    assert data["judge_model_name"] == "test-judge"
    assert data["corpus_snapshot"] is not None


def test_rerun_warns_on_unparseable_truth(client, _setup_db):
    """Should include warning in message when truth_payload cannot be parsed."""
    import asyncio

    from db import EvalResult

    _, async_session = _setup_db

    with patch("src.routes.evaluation._run_evaluation"):
        create_resp = client.post(
            "/evaluations/",
            json={
                "model_name": "granite-3.1-8b-instruct",
                "questions": ["What is AI?"],
            },
        )
    original_id = create_resp.json()["eval_run_id"]

    # Seed with invalid truth_payload
    async def _seed():
        async with async_session() as session:
            session.add(
                EvalResult(
                    eval_run_id=original_id,
                    question="What is AI?",
                    truth_payload={"invalid": "not a truth payload"},
                )
            )
            await session.commit()

    asyncio.run(_seed())

    with patch("src.routes.evaluation._run_evaluation"):
        rerun_resp = client.post(
            f"/evaluations/{original_id}/rerun",
            json={"model_name": "llama-3.1-8b-instruct"},
        )

    assert rerun_resp.status_code == 201
    data = rerun_resp.json()
    assert "unparseable truth" in data["message"]
    assert "1 question(s)" in data["message"]
