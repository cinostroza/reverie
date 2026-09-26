"""Reverie -- long-term memory for AI agents that learns which memories are
worth having.

    from reverie import Reverie

    mem = Reverie(scope="agent:deploy-bot")
    result = mem.recall("migrate orders table, add shipping_status")

    with mem.episode("migrate orders", recall_id=result.recall_id) as ep:
        code = run_migration()
        ep.step("bash", "alembic upgrade head", exit_code=code)
        ep.outcome("failure" if code else "success", tier=1, evidence=f"exit={code}")

    mem.dream()   # attribute outcomes, then consolidate
"""

from __future__ import annotations

from ._math import beta_ppf, utility_lcb
from .api import EpisodeRecorder, Reverie
from .attribution import AttributionEngine, credit_shares
from .config import Config
from .consolidate import Consolidator, Distiller, NullDistiller, Unit
from .embed import HashingEmbedder, get_embedder
from .ingest import Ingestor
from .models import (
    AgentRef,
    Episode,
    Node,
    OutcomeReport,
    RecallResult,
    SignalTier,
    Step,
    TaskRef,
    ValidationError,
)
from .recall import RecallEngine
from .store import SQLiteStore

__version__ = "0.1.0.dev0"

__all__ = [
    "Reverie",
    "EpisodeRecorder",
    "Config",
    "Episode",
    "Step",
    "TaskRef",
    "AgentRef",
    "OutcomeReport",
    "Node",
    "RecallResult",
    "SignalTier",
    "ValidationError",
    "SQLiteStore",
    "Ingestor",
    "RecallEngine",
    "AttributionEngine",
    "Consolidator",
    "Distiller",
    "NullDistiller",
    "Unit",
    "HashingEmbedder",
    "get_embedder",
    "credit_shares",
    "beta_ppf",
    "utility_lcb",
    "__version__",
]
