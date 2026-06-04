"""Evaluation profile loader -- reads versioned YAML profiles from disk."""

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"


class GenerationConfig(BaseModel):
    """Generation parameters driven by the evaluation profile."""

    # When set, overrides default max output tokens for RAG answers (chat completions).
    max_tokens: int | None = Field(default=None, ge=256, le=8192)


class RetrievalConfig(BaseModel):
    """Retrieval parameters driven by the evaluation profile."""

    top_k: int = 10
    max_chunks_per_document: int = 4
    rerank_depth: int = 20
    document_diversity_min: int = 3
    keyword_search_enabled: bool = True
    dedup_threshold: float = 0.85
    diversity_relevance_threshold: float = 0.3
    # After merge + document diversity, optionally force each sub-query to
    # contribute at least one chunk (see evaluation._process_question).
    ensure_sub_query_representation: bool = True
    # Swap-in chunk must score at least diversity_relevance_threshold * this
    # vs bare threshold alone, and must beat the evicted chunk (see evaluation).
    sub_query_swap_score_multiplier: float = 1.1


class EvalProfile(BaseModel):
    """An evaluation profile defining thresholds, retrieval config, and generation behavior."""

    id: str
    version: str = "1.0"
    domain: str = ""
    description: str = ""
    system_prompt: str = ""
    answer_contract: list[str] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)
    critical_thresholds: dict[str, float] = Field(default_factory=dict)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


def load_profile(profile_id: str) -> EvalProfile:
    """Load an evaluation profile by ID from the profiles directory.

    Args:
        profile_id: The profile filename stem (e.g., 'fsi_compliance_v1').

    Returns:
        Parsed EvalProfile.

    Raises:
        FileNotFoundError: If the profile YAML does not exist.
        ValueError: If the YAML is malformed or fails validation.
    """
    path = _PROFILES_DIR / f"{profile_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profile '{profile_id}' not found at {path}")

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return EvalProfile(**data)
    except Exception as e:
        raise ValueError(f"Failed to load profile '{profile_id}': {e}") from e


def list_profiles() -> list[str]:
    """Return available profile IDs (filename stems from profiles directory)."""
    if not _PROFILES_DIR.exists():
        return []
    return sorted(p.stem for p in _PROFILES_DIR.glob("*.yaml"))
