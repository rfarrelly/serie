from typing import List, Optional

from domains.predictions.entities import ModelPerformance, Prediction
from domains.predictions.repositories import PredictionRepository

from ..adapters.file_adapter import CSVFileAdapter


class CSVPredictionRepository(PredictionRepository):
    def __init__(self, file_adapter: CSVFileAdapter):
        self.file_adapter = file_adapter

    def save_predictions(self, predictions: List[Prediction]) -> None:
        data = []
        for pred in predictions:
            # Convert prediction to flat dictionary for CSV
            base_data = {
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

            # Add all method predictions from metadata
            if "all_methods" in pred.metadata:
                for method, probs in pred.metadata["all_methods"].items():
                    method_name = method.capitalize()
                    base_data.update(
                        {
                            f"{method_name}_Prob_H": probs["prob_home"],
                            f"{method_name}_Prob_D": probs["prob_draw"],
                            f"{method_name}_Prob_A": probs["prob_away"],
                        }
                    )

            data.append(base_data)

        self.file_adapter.write_dict_list(data, "latest_zsd_enhanced.csv")

    def get_latest_predictions(self) -> List[Prediction]:
        # Implementation would read from CSV and convert back to Prediction objects
        # For now, return empty list
        return []

    def get_performance_metrics(self, model_type: str) -> Optional[ModelPerformance]:
        # Implementation would calculate performance from historical predictions
        return None
