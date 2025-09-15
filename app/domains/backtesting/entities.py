from dataclasses import dataclass
from datetime import datetime
from typing import List

import pandas as pd


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""

    min_training_weeks: int = 10
    betting_threshold: float = 0.02
    stake_size: float = 1.0
    max_stake_fraction: float = 0.05
    min_edge: float = 0.02
    min_prob: float = 0.1
    max_odds: float = 10.0


@dataclass
class BacktestResult:
    """Results from a single betting decision."""

    match_id: str
    date: datetime
    home_team: str
    away_team: str
    bet_type: str
    stake: float
    odds: float
    profit: float
    model_prob: float
    market_prob: float
    edge: float
    expected_value: float


@dataclass
class BacktestSummary:
    """Summary of backtest performance."""

    total_bets: int
    total_profit: float
    total_staked: float
    roi_percent: float
    win_rate: float
    avg_edge: float
    final_bankroll: float
    predictions_df: pd.DataFrame
    betting_results: List[BacktestResult]
