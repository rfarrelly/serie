from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional


@dataclass
class EnhancedPredictionResponse:
    match_id: str
    home_team: str
    away_team: str
    league: str
    date: str
    week: Optional[int]

    # All method probabilities
    poisson_prob_h: float
    poisson_prob_d: float
    poisson_prob_a: float
    zip_prob_h: float
    zip_prob_d: float
    zip_prob_a: float
    mov_prob_h: float
    mov_prob_d: float
    mov_prob_a: float

    # Expected goals
    lambda_home: float
    lambda_away: float

    # MOV prediction
    mov_prediction: float
    mov_std_error: float

    # Market data
    sharp_odds_h: Optional[float] = None
    sharp_odds_d: Optional[float] = None
    sharp_odds_a: Optional[float] = None
    soft_odds_h: Optional[float] = None
    soft_odds_d: Optional[float] = None
    soft_odds_a: Optional[float] = None

    # Fair probabilities
    fair_prob_h: Optional[float] = None
    fair_prob_d: Optional[float] = None
    fair_prob_a: Optional[float] = None

    # PPI data
    home_ppi: Optional[float] = None
    away_ppi: Optional[float] = None
    ppi_diff: Optional[float] = None

    # Betting analysis
    bet_type: Optional[str] = None
    edge: Optional[float] = None
    model_prob: Optional[float] = None
    market_prob: Optional[float] = None
    expected_value: Optional[float] = None
    is_betting_candidate: bool = False


@dataclass
class EnhancedBettingOpportunityDTO:
    match_id: str
    home_team: str
    away_team: str
    league: str
    date: str
    bet_type: str
    odds: float
    stake: float
    model_probability: float
    market_probability: float
    edge: float
    expected_value: float

    # All method probabilities for reference
    poisson_prob_h: float
    poisson_prob_d: float
    poisson_prob_a: float
    zip_prob_h: float
    zip_prob_d: float
    zip_prob_a: float
    mov_prob_h: float
    mov_prob_d: float
    mov_prob_a: float

    # Additional edge calculations
    ev_h: float
    ev_d: float
    ev_a: float
    prob_edge_h: float
    prob_edge_d: float
    prob_edge_a: float
    kelly_h: float
    kelly_d: float
    kelly_a: float

    # PPI context
    ppi_diff: Optional[float] = None
