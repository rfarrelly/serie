# domain/services/model_optimizer.py
"""
Model Optimizer Domain Service - Self-contained within new architecture

Encapsulates parameter optimization logic using domain services and entities.
No dependencies on old project structure.
"""

import itertools
import random
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from domain.entities.match import Match
from domain.services.edge_calculator import EdgeCalculator
from domain.services.ppi_calculator import PPICalculator
from shared.types.common_types import LeagueName, Season


@dataclass(frozen=True)
class ModelParameters:
    """Immutable model parameters for optimization"""

    l1_regularization: Decimal = Decimal("0.0")
    l2_regularization: Decimal = Decimal("0.01")
    team_regularization: Decimal = Decimal("0.005")
    decay_rate: Decimal = Decimal("0.001")
    max_goals: int = 15
    min_matches_per_team: int = 5
    home_advantage_prior: Decimal = Decimal("0.2")
    away_adjustment_prior: Decimal = Decimal("-0.1")


@dataclass
class OptimizationResult:
    """Result of parameter optimization"""

    best_parameters: ModelParameters
    best_score: Decimal
    all_trials: List[Dict]
    convergence_history: List[Decimal]
    optimization_time: float
    league: LeagueName
    success: bool
    error_message: Optional[str] = None


@dataclass
class ModelPerformance:
    """Performance metrics for a model configuration"""

    accuracy: Decimal
    log_loss: Decimal
    brier_score: Decimal
    calibration_error: Decimal
    betting_roi: Decimal
    num_predictions: int
    num_betting_opportunities: int


class ModelOptimizer:
    """
    Domain service for optimizing model parameters.

    Self-contained within new architecture using domain services.
    """

    def __init__(self, ppi_calculator: PPICalculator, edge_calculator: EdgeCalculator):
        self._ppi_calculator = ppi_calculator
        self._edge_calculator = edge_calculator

        # Optimization configuration
        self._max_trials = 12
        self._parameter_bounds = {
            "l1_regularization": [0.0, 0.001, 0.005],
            "l2_regularization": [0.005, 0.01, 0.05],
            "team_regularization": [0.001, 0.005, 0.01],
            "decay_rate": [0.0005, 0.001, 0.002],
        }

    def optimize_parameters(
        self,
        league: LeagueName,
        training_matches: List[Match],
        validation_matches: List[Match],
        objective: str = "combined",
    ) -> OptimizationResult:
        """
        Optimize model parameters using cross-validation.

        Args:
            league: League to optimize for
            training_matches: Historical matches for training
            validation_matches: Matches for validation
            objective: Optimization objective ('accuracy', 'betting_roi', 'combined')
        """
        print(f"🔧 Optimizing parameters for {league}")
        print(f"   Training matches: {len(training_matches)}")
        print(f"   Validation matches: {len(validation_matches)}")

        start_time = datetime.now()

        # Generate parameter combinations
        parameter_combinations = self._generate_parameter_combinations()
        print(f"   Testing {len(parameter_combinations)} parameter combinations")

        best_parameters = None
        best_score = (
            Decimal("inf") if objective in ["log_loss", "combined"] else Decimal("-inf")
        )
        all_trials = []
        convergence_history = []

        for i, params in enumerate(parameter_combinations):
            print(
                f"   Trial {i + 1}/{len(parameter_combinations)}: {self._params_to_string(params)}"
            )

            try:
                # Evaluate parameter combination
                performance = self._evaluate_parameters(
                    params, training_matches, validation_matches
                )

                # Calculate optimization score
                score = self._calculate_optimization_score(performance, objective)

                trial_record = {
                    "trial": i + 1,
                    "parameters": params,
                    "performance": performance,
                    "score": score,
                    "success": True,
                }

                all_trials.append(trial_record)
                convergence_history.append(score)

                # Check if this is the best so far
                is_better = (
                    objective in ["log_loss", "combined"] and score < best_score
                ) or (objective in ["accuracy", "betting_roi"] and score > best_score)

                if is_better:
                    best_parameters = params
                    best_score = score
                    print(f"     ✅ New best! Score: {score:.4f}")
                else:
                    print(f"     Score: {score:.4f}")

            except Exception as e:
                print(f"     ❌ Trial failed: {e}")
                all_trials.append(
                    {
                        "trial": i + 1,
                        "parameters": params,
                        "error": str(e),
                        "success": False,
                    }
                )

        optimization_time = (datetime.now() - start_time).total_seconds()

        if best_parameters:
            print(f"✅ Optimization complete! Best score: {best_score:.4f}")
            print(f"   Best parameters: {self._params_to_string(best_parameters)}")
        else:
            print("❌ Optimization failed - no successful trials")

        return OptimizationResult(
            best_parameters=best_parameters or ModelParameters(),
            best_score=best_score,
            all_trials=all_trials,
            convergence_history=convergence_history,
            optimization_time=optimization_time,
            league=league,
            success=best_parameters is not None,
            error_message=None if best_parameters else "No successful parameter trials",
        )

    def _generate_parameter_combinations(self) -> List[ModelParameters]:
        """Generate parameter combinations for testing"""

        # Create all possible combinations
        param_names = list(self._parameter_bounds.keys())
        param_values = [self._parameter_bounds[name] for name in param_names]

        all_combinations = list(itertools.product(*param_values))

        # Sample a subset for efficiency
        max_combinations = min(self._max_trials, len(all_combinations))
        selected_combinations = random.sample(all_combinations, max_combinations)

        # Convert to ModelParameters objects
        parameter_objects = []
        for combo in selected_combinations:
            param_dict = dict(zip(param_names, combo))

            parameters = ModelParameters(
                l1_regularization=Decimal(str(param_dict["l1_regularization"])),
                l2_regularization=Decimal(str(param_dict["l2_regularization"])),
                team_regularization=Decimal(str(param_dict["team_regularization"])),
                decay_rate=Decimal(str(param_dict["decay_rate"])),
            )
            parameter_objects.append(parameters)

        return parameter_objects

    def _evaluate_parameters(
        self,
        parameters: ModelParameters,
        training_matches: List[Match],
        validation_matches: List[Match],
    ) -> ModelPerformance:
        """
        Evaluate model parameters using domain services.

        This replaces complex model training with domain service orchestration.
        """
        from domain.models.simple_prediction_model import SimplePredictionModel

        # Create and train model with parameters
        model = SimplePredictionModel(
            parameters=parameters,
            ppi_calculator=self._ppi_calculator,
            edge_calculator=self._edge_calculator,
        )

        model.train(training_matches)

        # Generate predictions for validation set
        predictions = []
        actual_outcomes = []
        betting_opportunities = []

        for match in validation_matches:
            if match.result is None:
                continue

            # Generate prediction using domain services
            prediction = model.predict_match(match.home_team, match.away_team)
            predictions.append(
                [
                    float(prediction.prob_home_win),
                    float(prediction.prob_draw),
                    float(prediction.prob_away_win),
                ]
            )

            # Get actual outcome
            actual_outcome = match.get_outcome()
            outcome_index = {"Home": 0, "Draw": 1, "Away": 2}[actual_outcome.value]
            actual_outcomes.append(outcome_index)

            # Check for betting opportunities if odds available
            if match.market_odds:
                # This would use edge calculator to find opportunities
                pass

        # Calculate performance metrics
        if not predictions:
            raise ValueError("No valid predictions generated")

        return self._calculate_performance_metrics(
            predictions, actual_outcomes, betting_opportunities
        )

    def _calculate_performance_metrics(
        self,
        predictions: List[List[float]],
        actual_outcomes: List[int],
        betting_opportunities: List,
    ) -> ModelPerformance:
        """Calculate comprehensive performance metrics"""

        import numpy as np
        from sklearn.metrics import accuracy_score, log_loss

        predictions_array = np.array(predictions)

        # Basic metrics
        predicted_outcomes = np.argmax(predictions_array, axis=1)
        accuracy = accuracy_score(actual_outcomes, predicted_outcomes)
        logloss = log_loss(actual_outcomes, predictions_array)

        # Brier score
        y_true_onehot = np.eye(3)[actual_outcomes]
        brier = np.mean(np.sum((predictions_array - y_true_onehot) ** 2, axis=1))

        # Calibration error (simplified)
        calibration_error = abs(np.mean(predictions_array) - np.mean(y_true_onehot))

        # Betting metrics (simplified for now)
        betting_roi = Decimal("0.0")  # Would calculate from betting_opportunities

        return ModelPerformance(
            accuracy=Decimal(str(accuracy)),
            log_loss=Decimal(str(logloss)),
            brier_score=Decimal(str(brier)),
            calibration_error=Decimal(str(calibration_error)),
            betting_roi=betting_roi,
            num_predictions=len(predictions),
            num_betting_opportunities=len(betting_opportunities),
        )

    def _calculate_optimization_score(
        self, performance: ModelPerformance, objective: str
    ) -> Decimal:
        """
        Calculate optimization score based on objective.

        Lower scores are better for minimization objectives.
        """
        if objective == "accuracy":
            return -performance.accuracy  # Negative because we minimize

        elif objective == "betting_roi":
            return -performance.betting_roi  # Negative because we minimize

        elif objective == "log_loss":
            return performance.log_loss

        elif objective == "combined":
            # Weighted combination of metrics
            accuracy_component = -performance.accuracy * Decimal("0.3")
            logloss_component = performance.log_loss * Decimal("0.4")
            betting_component = -performance.betting_roi * Decimal("0.3")

            return accuracy_component + logloss_component + betting_component

        else:
            raise ValueError(f"Unknown objective: {objective}")

    def _params_to_string(self, params: ModelParameters) -> str:
        """Convert parameters to readable string"""
        return f"L1={params.l1_regularization}, L2={params.l2_regularization}, Team={params.team_regularization}, Decay={params.decay_rate}"


class ParameterScheduler:
    """
    Manages parameter optimization schedules and triggers.

    Determines when models need re-optimization.
    """

    def __init__(self, optimization_interval_days: int = 90):
        self.optimization_interval = optimization_interval_days

    def should_optimize(
        self, league: LeagueName, last_optimization: Optional[datetime] = None
    ) -> bool:
        """Determine if league parameters should be re-optimized"""

        if last_optimization is None:
            return True  # Never optimized

        days_since = (datetime.now() - last_optimization).days
        return days_since >= self.optimization_interval

    def get_optimization_priority(
        self, leagues: List[LeagueName], last_optimizations: Dict[LeagueName, datetime]
    ) -> List[LeagueName]:
        """Get leagues sorted by optimization priority"""

        priority_list = []
        for league in leagues:
            last_opt = last_optimizations.get(league)
            if self.should_optimize(league, last_opt):
                days_since = (
                    (datetime.now() - last_opt).days if last_opt else float("inf")
                )
                priority_list.append((league, days_since))

        # Sort by days since optimization (descending)
        priority_list.sort(key=lambda x: x[1], reverse=True)

        return [league for league, _ in priority_list]
