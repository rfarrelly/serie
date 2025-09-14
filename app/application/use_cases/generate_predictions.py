from datetime import datetime
from typing import Dict, List

from domains.data.repositories import MatchRepository
from domains.predictions.repositories import PredictionRepository
from domains.predictions.services import PredictionService
from domains.shared.exceptions import InsufficientDataException


class GeneratePredictionsUseCase:
    def __init__(
        self,
        prediction_service: PredictionService,
        prediction_repository: PredictionRepository,
        match_repository: MatchRepository,
    ):
        self.prediction_service = prediction_service
        self.prediction_repository = prediction_repository
        self.match_repository = match_repository

    def execute(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        fixtures = self.match_repository.get_by_date_range(start_date, end_date)

        if not fixtures:
            raise InsufficientDataException("No fixtures found in date range")

        predictions = self.prediction_service.generate_predictions(fixtures)
        self.prediction_repository.save_predictions(predictions)

        # Convert to dictionary format for compatibility
        prediction_dicts = []
        for pred in predictions:
            pred_dict = {
                "Date": pred.match.date,
                "League": pred.match.home_team.league,
                "Home": pred.match.home_team.name,
                "Away": pred.match.away_team.name,
                "ZSD_Prob_H": float(pred.probabilities.home),
                "ZSD_Prob_D": float(pred.probabilities.draw),
                "ZSD_Prob_A": float(pred.probabilities.away),
                "ZSD_Lambda_H": pred.lambda_home,
                "ZSD_Lambda_A": pred.lambda_away,
                "Model_Type": pred.model_type,
            }

            # Add method-specific predictions
            if "all_methods" in pred.metadata:
                for method, probs in pred.metadata["all_methods"].items():
                    method_name = method.capitalize()
                    pred_dict.update(
                        {
                            f"{method_name}_Prob_H": probs["prob_home"],
                            f"{method_name}_Prob_D": probs["prob_draw"],
                            f"{method_name}_Prob_A": probs["prob_away"],
                        }
                    )

            prediction_dicts.append(pred_dict)

        return prediction_dicts
