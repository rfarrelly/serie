from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class PredictionRequest:
    league: Optional[str] = None
    season: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    prediction_method: str = "zip"


@dataclass
class PredictionResponse:
    match_id: str
    home_team: str
    away_team: str
    league: str
    date: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    expected_home_goals: float
    expected_away_goals: float
    model_type: str
    confidence: Optional[float] = None
