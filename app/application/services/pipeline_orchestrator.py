from datetime import datetime, timedelta
from typing import Dict, List

from application.use_cases.calculate_betting_edges import CalculateBettingEdgesUseCase
from application.use_cases.generate_predictions import GeneratePredictionsUseCase
from application.use_cases.ingest_data import IngestDataUseCase
from application.use_cases.optimize_models import OptimizeModelsUseCase
from domains.shared.exceptions import DomainException


class PipelineOrchestrator:
    def __init__(
        self,
        generate_predictions_uc: GeneratePredictionsUseCase,
        calculate_edges_uc: CalculateBettingEdgesUseCase,
        optimize_models_uc: OptimizeModelsUseCase,
        ingest_data_uc: IngestDataUseCase,
    ):
        self.generate_predictions_uc = generate_predictions_uc
        self.calculate_edges_uc = calculate_edges_uc
        self.optimize_models_uc = optimize_models_uc
        self.ingest_data_uc = ingest_data_uc

    def run_full_prediction_pipeline(self) -> Dict[str, List]:
        """Run the complete prediction and betting analysis pipeline"""
        try:
            # Generate predictions
            today = datetime.now()
            end_date = today + timedelta(days=7)

            print("Generating predictions...")
            prediction_dicts = self.generate_predictions_uc.execute(today, end_date)

            if not prediction_dicts:
                print("No predictions generated")
                return {"predictions": [], "betting_candidates": []}

            print(f"Generated {len(prediction_dicts)} predictions")

            # Convert back to domain objects for betting analysis
            predictions = self._dict_to_predictions(prediction_dicts)

            # Calculate betting edges
            print("Calculating betting edges...")
            betting_candidates = self.calculate_edges_uc.execute(predictions)

            print(f"Found {len(betting_candidates)} betting candidates")

            return {
                "predictions": prediction_dicts,
                "betting_candidates": betting_candidates,
            }

        except DomainException as e:
            print(f"Domain error: {e}")
            return {"predictions": [], "betting_candidates": []}
        except Exception as e:
            print(f"Pipeline error: {e}")
            return {"predictions": [], "betting_candidates": []}

    def run_optimization_pipeline(self, leagues: List[str]) -> None:
        """Run model optimization for specified leagues"""
        try:
            self.optimize_models_uc.execute(leagues)
            print("Model optimization completed")
        except Exception as e:
            print(f"Optimization error: {e}")

    def run_data_ingestion_pipeline(self, leagues: List[str]) -> None:
        """Run data ingestion and PPI calculation"""
        try:
            self.ingest_data_uc.execute_latest_ppi(leagues)
            print("Data ingestion completed")
        except Exception as e:
            print(f"Data ingestion error: {e}")

    def _dict_to_predictions(self, prediction_dicts: List[Dict]) -> List:
        """Convert prediction dictionaries back to domain objects"""
        # This is a simplified conversion - in a real implementation
        # we'd properly reconstruct the domain objects
        from decimal import Decimal

        from domains.data.entities import Match, Team
        from domains.predictions.entities import Prediction
        from domains.shared.value_objects import Probabilities

        predictions = []
        for pred_dict in prediction_dicts:
            home_team = Team(name=pred_dict["Home"], league=pred_dict["League"])
            away_team = Team(name=pred_dict["Away"], league=pred_dict["League"])

            match = Match(
                id=f"{pred_dict['Home']}_{pred_dict['Away']}_{pred_dict['Date']}",
                date=pred_dict["Date"],
                home_team=home_team,
                away_team=away_team,
            )

            probabilities = Probabilities(
                home=Decimal(str(pred_dict["ZSD_Prob_H"])),
                draw=Decimal(str(pred_dict["ZSD_Prob_D"])),
                away=Decimal(str(pred_dict["ZSD_Prob_A"])),
            )

            prediction = Prediction(
                match=match,
                probabilities=probabilities,
                lambda_home=pred_dict["ZSD_Lambda_H"],
                lambda_away=pred_dict["ZSD_Lambda_A"],
                model_type=pred_dict["Model_Type"],
                created_at=pred_dict["Date"],
                metadata={},
            )
            predictions.append(prediction)

        return predictions
