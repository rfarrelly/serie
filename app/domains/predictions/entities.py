from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

from ..data.entities import Match
from ..shared.value_objects import Probabilities


@dataclass
class Prediction:
    match: Match
    probabilities: Probabilities
    lambda_home: float
    lambda_away: float
    model_type: str
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class ModelPerformance:
    accuracy: float
    log_loss: float
    brier_score: float
    total_predictions: int
