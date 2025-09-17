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
        """Generate predictions with better error handling and debugging."""
        print(
            f"Generating predictions for date range: {start_date.date()} to {end_date.date()}"
        )

        try:
            fixtures = self.match_repository.get_by_date_range(start_date, end_date)

            if not fixtures:
                # Try to be more helpful with debugging info
                print("No fixtures found in date range. Checking available data...")

                # Check if we have any fixtures at all
                all_fixtures = self.match_repository.get_by_date_range(
                    datetime(2020, 1, 1), datetime(2030, 12, 31)
                )

                if all_fixtures:
                    print(
                        f"Found {len(all_fixtures)} total fixtures outside date range"
                    )
                    print("Sample fixture dates:")
                    for fixture in all_fixtures[:5]:
                        print(
                            f"  {fixture.home_team.name} vs {fixture.away_team.name}: {fixture.date.date()}"
                        )
                else:
                    print("No fixtures found at all - check data files")

                raise InsufficientDataException("No fixtures found in date range")

            print(f"Found {len(fixtures)} fixtures to predict")

            predictions = self.prediction_service.generate_predictions(fixtures)

            if not predictions:
                print("Prediction service returned no predictions")
                return []

            self.prediction_repository.save_predictions(predictions)

            # Convert to dictionary format for compatibility
            prediction_dicts = []
            for pred in predictions:
                pred_dict = {
                    "Date": (
                        pred.match.date.strftime("%Y-%m-%d")
                        if hasattr(pred.match.date, "strftime")
                        else str(pred.match.date)
                    ),
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

                # Add method-specific predictions with proper error handling
                if "all_methods" in pred.metadata:
                    for method, probs in pred.metadata["all_methods"].items():
                        method_name = method.capitalize()
                        try:
                            pred_dict.update(
                                {
                                    f"{method_name}_Prob_H": float(probs["prob_home"]),
                                    f"{method_name}_Prob_D": float(probs["prob_draw"]),
                                    f"{method_name}_Prob_A": float(probs["prob_away"]),
                                }
                            )
                        except (KeyError, TypeError, ValueError) as e:
                            print(f"Warning: Error adding {method} predictions: {e}")
                            # Add default values
                            pred_dict.update(
                                {
                                    f"{method_name}_Prob_H": 0.33,
                                    f"{method_name}_Prob_D": 0.34,
                                    f"{method_name}_Prob_A": 0.33,
                                }
                            )

                prediction_dicts.append(pred_dict)

            print(f"Generated {len(prediction_dicts)} prediction dictionaries")
            return prediction_dicts

        except Exception as e:
            print(f"Error in execute: {e}")
            import traceback

            traceback.print_exc()
            raise
