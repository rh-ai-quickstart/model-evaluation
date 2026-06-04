
"""SQLAdmin configuration for database administration UI."""

from db import ModelConfig
from sqladmin import Admin, ModelView
from sqlalchemy import create_engine

from .core.config import settings

# Create sync engine for SQLAdmin (it requires a sync engine internally)
sync_url = settings.DATABASE_URL.replace("+asyncpg", "")
engine = create_engine(sync_url, echo=False)


class ModelConfigAdmin(ModelView, model=ModelConfig):
    """Admin view for model configurations."""

    column_list = [
        ModelConfig.id,
        ModelConfig.name,
        ModelConfig.deployment_mode,
        ModelConfig.is_active,
        ModelConfig.created_at,
    ]
    column_searchable_list = [ModelConfig.name]
    column_sortable_list = [ModelConfig.id, ModelConfig.name, ModelConfig.is_active]
    column_default_sort = [(ModelConfig.created_at, True)]
    name = "Model Config"
    name_plural = "Model Configs"
    icon = "fa-solid fa-robot"


def setup_admin(app):
    """Set up SQLAdmin and mount it to the FastAPI app."""
    admin = Admin(app, engine, title="Model Evaluation Admin")
    admin.add_view(ModelConfigAdmin)
    return admin
