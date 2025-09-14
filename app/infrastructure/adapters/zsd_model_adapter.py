from typing import Any

import pandas as pd
from domains.shared.exceptions import ModelNotFittedException
from models.core import ModelConfig
from models.zsd_model import ZSDPoissonModel


class ZSDModelAdapter:
    def __init__(self, config: ModelConfig):
        self.model = ZSDPoissonModel(config)
        self._fitted = False
        self.league = None

    def fit(self, historical_data: pd.DataFrame, league: str) -> None:
        league_data = historical_data[historical_data["League"] == league].copy()
        if len(league_data) < 100:
            raise ModelNotFittedException(f"Insufficient data for {league}")

        self.model.fit(league_data)
        self._fitted = True
        self.league = league

    def predict_match(self, home_team: str, away_team: str, method: str = "zip") -> Any:
        if not self._fitted:
            raise ModelNotFittedException("Model must be fitted before prediction")

        return self.model.predict_match(home_team, away_team, method)

    def is_fitted(self) -> bool:
        return self._fitted
