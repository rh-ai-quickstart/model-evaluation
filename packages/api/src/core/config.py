"""Application configuration via pydantic-settings."""

import logging
from pathlib import Path
from typing import Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# packages/api/src/core/config.py -> repo root (5 levels up)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_ENV_FILE = _REPO_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    APP_NAME: str = "openshift-ai-model-evaluation"
    DEBUG: bool = False

    # CORS
    ALLOWED_HOSTS: list[str] = ["http://localhost:3000"]

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:changeme@localhost:5432/model-evaluation"

    # Shared MaaS endpoint and API token (single platform)
    MAAS_ENDPOINT: str = ""
    API_TOKEN: str = ""

    # LiteLLM admin UI base URL (for model metadata: context length, pricing, capabilities)
    LITELLM_ADMIN_URL: str = ""

    # Model names
    MODEL_A_NAME: str = ""
    MODEL_A_DEPLOYMENT_MODE: str = "maas"
    MODEL_B_NAME: str = ""
    MODEL_B_DEPLOYMENT_MODE: str = "maas"
    EMBEDDING_MODEL: str = ""
    JUDGE_MODEL_NAME: str = ""
    # If set, used for /evaluations/synthesize only. Otherwise MODEL_A_NAME, then JUDGE_MODEL_NAME.
    QUESTION_SYNTHESIS_MODEL_NAME: str = ""

    # Safety (Llama Guard)
    SAFETY_MODEL_NAME: str = ""
    SAFETY_ENABLED: bool = True

    # Document parsing
    DOCLING_ENABLED: bool = True

    # S3/MinIO ingestion
    S3_ENDPOINT_URL: str = ""
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""

    # Retrieval: max vector/keyword candidate pool per query (see retrieval.compute_search_depth).
    # Caps ``doc_count * rerank_depth`` when diversity is enabled to avoid huge DB LIMITs.
    RETRIEVAL_MAX_SEARCH_DEPTH: int = 400

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def api_token_bare(self) -> str:
        """API token without a ``Bearer `` prefix (headers add it)."""
        t = (self.API_TOKEN or "").strip()
        if t.lower().startswith("bearer "):
            return t[7:].strip()
        return t

    @property
    def question_synthesis_model_name(self) -> str:
        """Model for question generation: explicit override, then chat model A, then judge."""
        for candidate in (
            self.QUESTION_SYNTHESIS_MODEL_NAME,
            self.MODEL_A_NAME,
            self.JUDGE_MODEL_NAME,
        ):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return ""

    @property
    def resolved_judge_model_name(self) -> str:
        """Model for LLM-as-judge scoring: explicit judge, then chat A, then B."""
        for candidate in (self.JUDGE_MODEL_NAME, self.MODEL_A_NAME, self.MODEL_B_NAME):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return ""

    def get_model_config(self, model_name: str) -> dict:
        """Get endpoint and token for a model by name.

        Returns dict with 'endpoint', 'token' keys.
        All models share the same MaaS endpoint and API token.
        """
        base = (self.MAAS_ENDPOINT or "").rstrip("/")
        return {"endpoint": base, "token": self.api_token_bare}

    @property
    def any_token_configured(self) -> bool:
        """Return True if an API token is configured."""
        return bool(self.API_TOKEN)

    @model_validator(mode="after")
    def validate_api_tokens(self) -> Self:
        """Warn if no API token is set."""
        if not self.API_TOKEN:
            logger.warning("No API token set. Set API_TOKEN to enable model serving.")
        return self


settings = Settings()
