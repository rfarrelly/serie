from datetime import datetime
from typing import Any, Dict

from domain.repositories.match_repository import MatchRepository
from domain.services.model_service import ModelService
from infrastructure.storage.config_storage import ConfigStorage


class OptimizeParametersUseCase:
    def __init__(
        self,
        match_repository: MatchRepository,
        model_service: ModelService,
        config_storage: ConfigStorage,
    ):
        self.match_repository = match_repository
        self.model_service = model_service
        self.config_storage = config_storage

    def execute(self, league: str = None) -> Dict[str, Any]:
        # Get historical matches
        historical_matches = self.match_repository.find_played_matches()

        if league:
            historical_matches = [m for m in historical_matches if m.league == league]

        if len(historical_matches) < 100:
            raise ValueError(
                f"Insufficient data for optimization: {len(historical_matches)} matches"
            )

        # Optimize model parameters
        # This would involve cross-validation and parameter grid search
        # For now, we'll use the existing model service
        team_ratings = self.model_service.fit_ratings(historical_matches)

        # Save optimized parameters
        config = {
            "decay_rate": self.model_service.decay_rate,
            "l2_reg": self.model_service.l2_reg,
            "home_advantage": self.model_service.home_advantage,
            "away_adjustment": self.model_service.away_adjustment,
            "optimized_at": datetime.now().isoformat(),
        }

        config_name = f"{league}_model" if league else "global_model"
        self.config_storage.save_config(config_name, config)

        return {
            "optimized_parameters": config,
            "num_teams": len(team_ratings),
            "num_matches": len(historical_matches),
        }
