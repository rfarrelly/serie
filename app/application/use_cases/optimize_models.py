from typing import List

import pandas as pd
from domains.data.repositories import MatchRepository
from managers.model_manager import ModelManager


class OptimizeModelsUseCase:
    def __init__(self, model_manager: ModelManager, match_repository: MatchRepository):
        self.model_manager = model_manager
        self.match_repository = match_repository

    def execute(self, leagues: List[str]) -> None:
        """Optimize model parameters for given leagues"""
        # Load historical data
        historical_data = pd.read_csv("historical_ppi_and_odds.csv")

        for league in leagues:
            league_data = historical_data[historical_data["League"] == league]

            if len(league_data) > 179:
                print(f"Optimizing parameters for {league}...")
                self.model_manager.optimize_league_parameters(
                    historical_data, league, save_config=True
                )
            else:
                print(f"Skipping {league} - insufficient data")
