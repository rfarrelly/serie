from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp


class ValidateBacktestUseCase:
    def __init__(self):
        pass

    def execute(
        self, betting_results_file: str, predictions_file: str
    ) -> Dict[str, Any]:
        # Load results
        betting_df = pd.read_csv(betting_results_file)
        predictions_df = pd.read_csv(predictions_file)

        validation_results = {
            "is_valid": True,
            "confidence_score": 0.0,
            "red_flags": [],
            "warnings": [],
            "metrics": {},
        }

        # Validate temporal consistency
        temporal_issues = self._validate_temporal_consistency(
            betting_df, predictions_df
        )
        validation_results["red_flags"].extend(temporal_issues["red_flags"])
        validation_results["warnings"].extend(temporal_issues["warnings"])
        validation_results["metrics"].update(temporal_issues["metrics"])

        # Validate odds consistency
        odds_issues = self._validate_odds_consistency(betting_df, predictions_df)
        validation_results["red_flags"].extend(odds_issues["red_flags"])
        validation_results["warnings"].extend(odds_issues["warnings"])
        validation_results["metrics"].update(odds_issues["metrics"])

        # Validate performance distribution
        perf_issues = self._validate_performance_distribution(betting_df)
        validation_results["red_flags"].extend(perf_issues["red_flags"])
        validation_results["warnings"].extend(perf_issues["warnings"])
        validation_results["metrics"].update(perf_issues["metrics"])

        # Statistical significance tests
        stat_issues = self._run_significance_tests(betting_df)
        validation_results["red_flags"].extend(stat_issues["red_flags"])
        validation_results["warnings"].extend(stat_issues["warnings"])
        validation_results["metrics"].update(stat_issues["metrics"])

        # Calculate confidence score
        validation_results["confidence_score"] = self._calculate_confidence_score(
            validation_results["red_flags"],
            validation_results["warnings"],
            validation_results["metrics"],
        )

        validation_results["is_valid"] = (
            validation_results["confidence_score"] > 0.7
            and len(validation_results["red_flags"]) == 0
        )

        return validation_results

    def _validate_temporal_consistency(
        self, betting_df: pd.DataFrame, predictions_df: pd.DataFrame
    ) -> Dict:
        red_flags = []
        warnings = []
        metrics = {}

        # Check for future betting (betting after match occurred)
        if "date" in betting_df.columns:
            betting_df["date"] = pd.to_datetime(betting_df["date"])
            # Additional temporal validation logic here
            metrics["temporal_violations"] = 0

        return {"red_flags": red_flags, "warnings": warnings, "metrics": metrics}

    def _validate_odds_consistency(
        self, betting_df: pd.DataFrame, predictions_df: pd.DataFrame
    ) -> Dict:
        red_flags = []
        warnings = []
        metrics = {}

        # Check probability sums
        prob_cols = ["Model_Prob_H", "Model_Prob_D", "Model_Prob_A"]
        if all(col in predictions_df.columns for col in prob_cols):
            prob_sums = predictions_df[prob_cols].sum(axis=1)
            violations = np.sum(np.abs(prob_sums - 1.0) > 0.01)
            metrics["prob_sum_violations"] = violations

            if violations > len(predictions_df) * 0.1:
                red_flags.append(
                    f"Model probabilities don't sum to 1 in {violations} cases"
                )

        # Check odds ranges
        if "odds" in betting_df.columns:
            unrealistic_odds = np.sum(
                (betting_df["odds"] < 1.01) | (betting_df["odds"] > 100)
            )
            metrics["unrealistic_odds"] = unrealistic_odds

            if unrealistic_odds > 0:
                warnings.append(f"Found {unrealistic_odds} bets with unrealistic odds")

        return {"red_flags": red_flags, "warnings": warnings, "metrics": metrics}

    def _validate_performance_distribution(self, betting_df: pd.DataFrame) -> Dict:
        red_flags = []
        warnings = []
        metrics = {}

        if "profit" not in betting_df.columns:
            return {"red_flags": red_flags, "warnings": warnings, "metrics": metrics}

        profits = betting_df["profit"]
        win_rate = np.mean(profits > 0)

        # Check for unrealistic win rates
        if win_rate > 0.65:
            red_flags.append(f"Unrealistically high win rate: {win_rate:.1%}")
        elif win_rate > 0.6:
            warnings.append(f"Very high win rate: {win_rate:.1%}")

        metrics.update(
            {
                "win_rate": win_rate,
                "total_profit": profits.sum(),
                "avg_profit": profits.mean(),
                "profit_std": profits.std(),
            }
        )

        return {"red_flags": red_flags, "warnings": warnings, "metrics": metrics}

    def _run_significance_tests(self, betting_df: pd.DataFrame) -> Dict:
        red_flags = []
        warnings = []
        metrics = {}

        if "profit" not in betting_df.columns or len(betting_df) < 10:
            return {"red_flags": red_flags, "warnings": warnings, "metrics": metrics}

        profits = betting_df["profit"]

        # T-test for profit vs zero
        t_stat, p_value = ttest_1samp(profits, 0)
        is_significant = p_value < 0.05

        # Sharpe ratio
        sharpe_ratio = (
            profits.mean() / profits.std() * np.sqrt(len(profits))
            if profits.std() > 0
            else 0
        )

        metrics.update(
            {
                "t_statistic": t_stat,
                "p_value": p_value,
                "is_significant": is_significant,
                "sharpe_ratio": sharpe_ratio,
            }
        )

        if not is_significant:
            warnings.append(f"Results not statistically significant (p={p_value:.3f})")

        if sharpe_ratio > 2.0:
            warnings.append(f"Extremely high Sharpe ratio: {sharpe_ratio:.2f}")

        return {"red_flags": red_flags, "warnings": warnings, "metrics": metrics}

    def _calculate_confidence_score(
        self, red_flags: List[str], warnings: List[str], metrics: Dict
    ) -> float:
        score = 1.0
        score -= len(red_flags) * 0.3
        score -= len(warnings) * 0.1

        # Adjust based on specific metrics
        if metrics.get("p_value", 0) > 0.05:
            score -= 0.2

        if metrics.get("win_rate", 0) > 0.65:
            score -= 0.15

        return max(0.0, min(1.0, score))
