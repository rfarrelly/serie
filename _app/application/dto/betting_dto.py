from dataclasses import dataclass


@dataclass
class BettingOpportunityDTO:
    match_id: str
    home_team: str
    away_team: str
    bet_type: str
    odds: float
    stake: float
    model_probability: float
    market_probability: float
    edge: float
    expected_value: float
    league: str
    date: str


@dataclass
class BettingResultDTO:
    opportunity: BettingOpportunityDTO
    actual_outcome: str
    profit: float
    won: bool
    date_placed: str
