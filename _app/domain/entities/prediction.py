from dataclasses import dataclass
from typing import Optional

from ..value_objects.probability import Probability


@dataclass
class Prediction:
    match_id: str
    home_team: str
    away_team: str
    home_win_prob: Probability
    draw_prob: Probability
    away_win_prob: Probability
    expected_home_goals: float
    expected_away_goals: float
    model_type: str
    confidence: Optional[float] = None

    @property
    def most_likely_outcome(self) -> str:
        probs = {
            "Home": self.home_win_prob.value,
            "Draw": self.draw_prob.value,
            "Away": self.away_win_prob.value,
        }
        return max(probs, key=probs.get)


@dataclass
class PredictionResult:
    prediction: Prediction
    actual_result: Optional[str] = None
    accuracy: Optional[float] = None

    def is_correct(self) -> Optional[bool]:
        if self.actual_result is None:
            return None
        return self.prediction.most_likely_outcome == self.actual_result
