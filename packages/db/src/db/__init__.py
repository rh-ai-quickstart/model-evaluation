
__version__ = "0.0.0"

# Export main database classes and functions
from .database import DatabaseService, get_db_service, get_db, Base
from .models import EMBEDDING_DIMENSION, Chunk, Document, EvalResult, EvalRun, ModelConfig, QuestionSet

__all__ = [
    "DatabaseService",
    "get_db_service",
    "get_db",
    "Base",
    "Chunk",
    "Document",
    "EMBEDDING_DIMENSION",
    "EvalResult",
    "EvalRun",
    "ModelConfig",
    "QuestionSet",
    "__version__",
]
