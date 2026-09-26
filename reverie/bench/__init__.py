"""reverie-bench -- does an agent actually get better at its job? (HLD M0)"""

from .agents import EvidenceAgent, LLMAgent
from .backends import NoMemoryBackend, ReplayBackend, ReverieBackend
from .core import Attempt, Evidence, Task, TaskFamily, compare_arms, run_arm
from .families import (ALL_FAMILIES, ApiIntegrationFamily, IncidentFamily,
                       MigrationFamily, PipelineFamily)
from .human import HumanFeedback, HumanLesson, HumanOracle
from .objective import (SEARCH_SPACE, Score, baselines_for, config_from_params,
                        evaluate, make_objective)

__all__ = [
    "Task", "Attempt", "Evidence", "TaskFamily", "run_arm", "compare_arms",
    "EvidenceAgent", "LLMAgent",
    "NoMemoryBackend", "ReplayBackend", "ReverieBackend",
    "MigrationFamily", "ApiIntegrationFamily", "PipelineFamily",
    "IncidentFamily", "ALL_FAMILIES",
    "HumanOracle", "HumanLesson", "HumanFeedback",
    "evaluate", "Score", "baselines_for", "make_objective", "SEARCH_SPACE",
    "config_from_params",
]
