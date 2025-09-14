from decimal import Decimal
from typing import List, Protocol

from ..data.entities import Match
from ..shared.exceptions import ModelNotFittedException
from ..shared.value_objects import Probabilities
from .entities import Prediction


class PredictionModel(Protocol):
    def predict_match(
        self, home_team: str, away_team: str, method: str = "zip"
    ) -> dict:
        pass

    def is_fitted(self) -> bool:
        pass


class PredictionService:
    def __init__(self, model: PredictionModel):
        self.model = model

    def generate_predictions(self, matches: List[Match]) -> List[Prediction]:
        if not self.model.is_fitted():
            raise ModelNotFittedException(
                "Model must be fitted before generating predictions"
            )

        predictions = []
        for match in matches:
            try:
                # Get predictions for all methods
                methods = ["poisson", "zip", "mov"]
                method_predictions = {}

                for method in methods:
                    pred_result = self.model.predict_match(
                        match.home_team.name, match.away_team.name, method=method
                    )
                    method_predictions[method] = pred_result

                # Use ZIP as primary prediction
                primary_pred = method_predictions["zip"]
                probabilities = Probabilities(
                    home=Decimal(str(primary_pred.prob_home_win)),
                    draw=Decimal(str(primary_pred.prob_draw)),
                    away=Decimal(str(primary_pred.prob_away_win)),
                )

                prediction = Prediction(
                    match=match,
                    probabilities=probabilities,
                    lambda_home=primary_pred.lambda_home,
                    lambda_away=primary_pred.lambda_away,
                    model_type="ZSD",
                    created_at=match.date,
                    metadata={
                        "all_methods": {
                            method: {
                                "prob_home": pred.prob_home_win,
                                "prob_draw": pred.prob_draw,
                                "prob_away": pred.prob_away_win,
                            }
                            for method, pred in method_predictions.items()
                        }
                    },
                )
                predictions.append(prediction)

            except Exception as e:
                continue  # Skip matches that can't be predicted

        return predictions
