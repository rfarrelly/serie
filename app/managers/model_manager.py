"""
Model manager for handling multiple league models.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from application.services.model_optimization_service import ModelOptimizationService
from domains.betting.services import BettingAnalysisService
from models.core import ModelConfig
from models.zsd_model import ZSDPoissonModel
from utils.config_manager import ConfigManager


class ModelManager:
    """Manages ZSD Poisson models for multiple leagues."""

    def __init__(self, global_config, config_dir: Path = Path("zsd_configs")):
        self.global_config = global_config
        self.config_manager = ConfigManager(config_dir)
        self.betting_service = BettingAnalysisService()
        self.optimization_service = ModelOptimizationService(config_dir)
        self.models = {}

    def optimize_league_parameters(
        self, historical_data: pd.DataFrame, league: str, save_config: bool = True
    ) -> ModelConfig:
        """Optimize regularization parameters for a specific league using the optimization service."""
        return self.optimization_service.optimize_league_parameters(
            historical_data, league, save_config
        )

    def fit_league_model(
        self,
        historical_data: pd.DataFrame,
        league: str,
        min_training_matches: int = 180,
    ) -> Optional[ZSDPoissonModel]:
        """Fit a ZSD Poisson model for a specific league."""
        print(f"Fitting ZSD model for {league}...")

        league_data = historical_data[
            (historical_data["League"] == league)
            & historical_data["Season"].isin(
                [
                    self.global_config.current_season,
                    self.global_config.previous_season,
                ]
            )
        ].copy()
        league_data = league_data.sort_values("Date").reset_index(drop=True)

        if len(league_data) < min_training_matches:
            print(f"Insufficient data for {league}: {len(league_data)} matches")
            return None

        config = self.config_manager.load_league_config(league)

        try:
            model = ZSDPoissonModel(config)
            model.fit(league_data)
            self.models[league] = model

            print(f"  Successfully fitted model for {league}")
            print(
                f"  Training data: {len(league_data)} matches, Teams: {len(model.teams)}"
            )
            print(f"  Last match in data:\r\n{league_data.tail(1)}")
            return model

        except Exception as e:
            print(f"Error fitting model for {league}: {e}")
            return None

    def predict_matches(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for upcoming matches."""
        predictions = []
        previous_season = self.global_config.previous_season

        for _, match in fixtures_df.iterrows():
            league = match.get("League")

            if self._is_relegated_or_promoted(
                teams=set(match[["Home", "Away"]]),
                league=league,
                previous_season=previous_season,
            ):
                print(
                    f"Rejected fixture: {match['Home']} v {match['Away']} - one or more teams not in previous season"
                )
                continue

            model = self.models.get(league)

            if model is None:
                print(f"No model available for {league}")
                continue

            try:
                pred_dict = self._generate_match_predictions(match, model)
                predictions.append(pred_dict)
            except Exception as e:
                print(f"Error predicting {match['Home']} vs {match['Away']}: {e}")

        return pd.DataFrame(predictions)

    def _is_relegated_or_promoted(
        self, teams: set[str], league: str, previous_season: str
    ) -> bool:
        """Checks if teams have been relegated or promoted to/from previous season"""
        previous_season_file_path = f"{self.global_config.fbref_data_dir}/{league}/{league}_{previous_season}.csv"
        if not teams.issubset(pd.read_csv(previous_season_file_path)["Home"]):
            return True
        return False

    def _generate_match_predictions(self, match, model) -> Dict:
        """Generate all prediction types for a single match."""
        home_team, away_team = match["Home"], match["Away"]

        # Get predictions for all methods
        methods = ["poisson", "zip", "mov"]
        predictions = {
            method: model.predict_match(home_team, away_team, method=method)
            for method in methods
        }

        # Build base prediction dictionary
        pred_dict = {
            "Date": match["Date"],
            "League": match.get("League"),
            "Home": home_team,
            "Away": away_team,
        }

        # Add predictions for each method
        method_names = ["Poisson", "ZIP", "MOV"]
        for method, name in zip(methods, method_names):
            pred = predictions[method]
            pred_dict.update(
                {
                    f"{name}_Prob_H": pred.prob_home_win,
                    f"{name}_Prob_D": pred.prob_draw,
                    f"{name}_Prob_A": pred.prob_away_win,
                }
            )

        # Add backward compatibility fields (using ZIP as primary)
        zip_pred = predictions["zip"]
        pred_dict.update(
            {
                "ZSD_Prob_H": zip_pred.prob_home_win,
                "ZSD_Prob_D": zip_pred.prob_draw,
                "ZSD_Prob_A": zip_pred.prob_away_win,
                "ZSD_Lambda_H": zip_pred.lambda_home,
                "ZSD_Lambda_A": zip_pred.lambda_away,
                "ZSD_MOV": zip_pred.mov_prediction,
                "ZSD_MOV_SE": zip_pred.mov_std_error,
                "Model_Type": zip_pred.model_type,
            }
        )

        # Add fixture columns if present
        fixture_cols = [
            "Wk",
            "PSH",
            "PSD",
            "PSA",
            "PSCH",
            "PSCD",
            "PSCA",
            "B365H",
            "B365D",
            "B365A",
            "B365CH",
            "B365CD",
            "B365CA",
            "PPI_Diff",
            "hPPI",
            "aPPI",
        ]

        for col in fixture_cols:
            if col in match.index:
                pred_dict[col] = match[col]

        # Add betting analysis if odds are available
        pred_series = pd.Series(pred_dict)
        betting_metrics = self.betting_service.analyze_prediction_row(pred_series)

        if betting_metrics:
            # Add betting metrics to prediction
            betting_dict = self.betting_service._metrics_to_dict(betting_metrics)
            pred_dict.update(betting_dict)
        else:
            # Add default values if no betting analysis possible
            pred_dict.update(
                {
                    "Bet_Type": None,
                    "Edge": 0.0,
                    "Model_Prob": 0.0,
                    "Market_Prob": 0.0,
                    "Sharp_Odds": 0.0,
                    "Soft_Odds": 0.0,
                    "Fair_Odds_Selected": 0.0,
                    "EV_H": 0.0,
                    "EV_D": 0.0,
                    "EV_A": 0.0,
                    "Prob_Edge_H": 0.0,
                    "Prob_Edge_D": 0.0,
                    "Prob_Edge_A": 0.0,
                    "Kelly_H": 0.0,
                    "Kelly_D": 0.0,
                    "Kelly_A": 0.0,
                }
            )

        return pred_dict
