from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from domains.betting.services import BettingAnalysisService
from sklearn.metrics import log_loss

from .entities import BacktestConfig, BacktestResult, BacktestSummary


class BacktestingService:
    """Enhanced backtesting service with proper temporal validation."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.betting_service = BettingAnalysisService(
            min_edge=config.min_edge, min_prob=config.min_prob, max_odds=config.max_odds
        )

    def run_cross_season_backtest(
        self,
        model_class,
        data: pd.DataFrame,
        train_season: str,
        test_season: str,
        league: str,
        model_params: Optional[Dict] = None,
    ) -> BacktestSummary:
        """Run a temporally valid cross-season backtest."""

        # Prepare data with proper temporal ordering
        train_data, test_data = self._prepare_backtest_data(
            data, train_season, test_season, league
        )

        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError(
                f"Insufficient data for {league} {train_season}/{test_season}"
            )

        # Sort test data by date to ensure temporal order
        test_data = test_data.sort_values(["Date", "Wk"]).reset_index(drop=True)

        all_predictions = []
        betting_results = []
        bankroll = 100.0

        print(f"Backtesting {league}: {train_season} -> {test_season}")
        print(f"Training matches: {len(train_data)}, Test matches: {len(test_data)}")

        # Process matches chronologically
        for idx, (_, match_row) in enumerate(test_data.iterrows()):
            if idx < self.config.min_training_weeks:
                continue

            try:
                # Use only data available before this match
                available_train_data = train_data.copy()
                available_test_data = test_data.iloc[:idx].copy()  # Only past matches

                combined_training = pd.concat(
                    [available_train_data, available_test_data], ignore_index=True
                )

                # Fit model on available data only
                model = self._create_and_fit_model(
                    model_class, combined_training, model_params
                )

                # Generate prediction for current match
                prediction_result = self._predict_single_match(model, match_row)

                if prediction_result:
                    all_predictions.append(prediction_result)

                    # Evaluate betting opportunity
                    betting_result, bankroll = self._evaluate_betting_opportunity(
                        pd.Series(prediction_result),
                        prediction_result["Actual_Outcome"],
                        bankroll,
                    )
                    if betting_result:
                        betting_results.append(betting_result)

            except Exception as e:
                print(f"Error in match {idx}: {str(e)}")
                continue

        # Calculate summary
        predictions_df = pd.DataFrame(all_predictions)
        summary = self._calculate_backtest_summary(
            predictions_df, betting_results, bankroll
        )

        return summary

    def _prepare_backtest_data(
        self, data: pd.DataFrame, train_season: str, test_season: str, league: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data ensuring no temporal leakage."""

        # Filter by league and season
        train_data = (
            data[(data["Season"] == train_season) & (data["League"] == league)]
            .copy()
            .sort_values(["Date", "Wk"])
        )

        test_data = (
            data[(data["Season"] == test_season) & (data["League"] == league)]
            .copy()
            .sort_values(["Date", "Wk"])
        )

        # Ensure teams exist in both seasons (remove promoted/relegated teams)
        train_teams = set(pd.concat([train_data["Home"], train_data["Away"]]))
        test_teams = set(pd.concat([test_data["Home"], test_data["Away"]]))
        common_teams = train_teams & test_teams

        if len(common_teams) < 10:  # Need reasonable number of teams
            print(f"Warning: Only {len(common_teams)} common teams between seasons")

        train_data = train_data[
            train_data["Home"].isin(common_teams)
            & train_data["Away"].isin(common_teams)
        ]

        test_data = test_data[
            test_data["Home"].isin(common_teams) & test_data["Away"].isin(common_teams)
        ]

        return train_data, test_data

    def _create_and_fit_model(
        self, model_class, training_data: pd.DataFrame, model_params: Optional[Dict]
    ):
        """Create and fit model ensuring temporal validity."""
        model_params = model_params or {}

        if isinstance(model_class, type):
            from models.core import ModelConfig

            model = model_class(model_params.get("config", ModelConfig()))
        else:
            model = model_class(**model_params)

        # Ensure training data is sorted chronologically
        training_data = training_data.sort_values(["Date", "Wk"]).reset_index(drop=True)
        model.fit(training_data)
        return model

    def _predict_single_match(self, model, match_row: pd.Series) -> Optional[Dict]:
        """Generate prediction for a single match."""
        try:
            home_team = match_row["Home"]
            away_team = match_row["Away"]

            # Check if teams are known to model
            if (
                not hasattr(model, "team_index")
                or home_team not in model.team_index
                or away_team not in model.team_index
            ):
                return None

            # Get predictions for all methods
            predictions = self._get_all_method_predictions(model, home_team, away_team)

            # Extract market odds with validation
            market_data = self._extract_market_data(match_row)
            if not market_data:
                return None

            # Calculate fair probabilities
            fair_probs = self._calculate_fair_probabilities(market_data["sharp_odds"])

            # Determine actual outcome
            actual_outcome = self._get_actual_outcome(match_row)

            return {
                "Date": match_row["Date"],
                "Season": match_row["Season"],
                "Week": match_row["Wk"],
                "Home": home_team,
                "Away": away_team,
                "FTHG": match_row["FTHG"],
                "FTAG": match_row["FTAG"],
                "Actual_Outcome": actual_outcome,
                # Model probabilities
                "Poisson_Prob_H": predictions["poisson"]["prob_home_win"],
                "Poisson_Prob_D": predictions["poisson"]["prob_draw"],
                "Poisson_Prob_A": predictions["poisson"]["prob_away_win"],
                "ZIP_Prob_H": predictions["zip"]["prob_home_win"],
                "ZIP_Prob_D": predictions["zip"]["prob_draw"],
                "ZIP_Prob_A": predictions["zip"]["prob_away_win"],
                "MOV_Prob_H": predictions["mov"]["prob_home_win"],
                "MOV_Prob_D": predictions["mov"]["prob_draw"],
                "MOV_Prob_A": predictions["mov"]["prob_away_win"],
                # Market data
                "PSH": market_data["sharp_odds"][0],
                "PSD": market_data["sharp_odds"][1],
                "PSA": market_data["sharp_odds"][2],
                "B365H": market_data["soft_odds"][0],
                "B365D": market_data["soft_odds"][1],
                "B365A": market_data["soft_odds"][2],
                "Fair_Prob_H": fair_probs[0] if fair_probs else np.nan,
                "Fair_Prob_D": fair_probs[1] if fair_probs else np.nan,
                "Fair_Prob_A": fair_probs[2] if fair_probs else np.nan,
                # Model outputs
                "Lambda_Home": predictions["zip"]["lambda_home"],
                "Lambda_Away": predictions["zip"]["lambda_away"],
            }

        except Exception as e:
            print(
                f"Error predicting match {match_row.get('Home', 'Unknown')} vs {match_row.get('Away', 'Unknown')}: {e}"
            )
            return None

    def _extract_market_data(self, match_row: pd.Series) -> Optional[Dict]:
        """Extract and validate market odds data."""
        sharp_cols = ["PSH", "PSD", "PSA"]
        soft_cols = ["B365H", "B365D", "B365A"]

        # Check if required columns exist and have valid values
        sharp_odds = []
        soft_odds = []

        for col in sharp_cols:
            val = match_row.get(col)
            if pd.isna(val) or val <= 1.0:
                return None
            sharp_odds.append(val)

        for col in soft_cols:
            val = match_row.get(col)
            if pd.isna(val) or val <= 1.0:
                return None
            soft_odds.append(val)

        return {"sharp_odds": sharp_odds, "soft_odds": soft_odds}

    def _get_all_method_predictions(
        self, model, home_team: str, away_team: str
    ) -> Dict[str, Dict]:
        """Get predictions for all methods from the model."""
        try:
            predictions = {}
            methods = ["poisson", "zip", "mov"]

            for method in methods:
                try:
                    pred = model.predict_match(home_team, away_team, method=method)
                    predictions[method] = {
                        "prob_home_win": pred.prob_home_win,
                        "prob_draw": pred.prob_draw,
                        "prob_away_win": pred.prob_away_win,
                        "lambda_home": pred.lambda_home,
                        "lambda_away": pred.lambda_away,
                    }
                except Exception:
                    # Fallback to basic calculation
                    predictions[method] = self._calculate_basic_prediction(
                        model, home_team, away_team
                    )

            return predictions

        except Exception as e:
            print(f"Error in _get_all_method_predictions: {e}")
            # Return default predictions
            default_prediction = {
                "prob_home_win": 0.33,
                "prob_draw": 0.33,
                "prob_away_win": 0.34,
                "lambda_home": 1.5,
                "lambda_away": 1.2,
            }
            return {
                method: default_prediction.copy()
                for method in ["poisson", "zip", "mov"]
            }

    def _calculate_basic_prediction(
        self, model, home_team: str, away_team: str
    ) -> Dict:
        """Calculate basic prediction when advanced methods fail."""
        try:
            home_idx = model.team_index[home_team]
            away_idx = model.team_index[away_team]

            # Calculate expected goals
            home_strength = (
                model.home_advantage
                + model.attack_ratings[home_idx]
                - model.defense_ratings[away_idx]
            )
            away_strength = (
                model.away_adjustment
                + model.attack_ratings[away_idx]
                - model.defense_ratings[home_idx]
            )

            lambda_home = max(0.1, 1.5 + 0.5 * home_strength)
            lambda_away = max(0.1, 1.2 + 0.5 * away_strength)

            # Simple Poisson outcome calculation
            from scipy.stats import poisson

            max_goals = getattr(model.config, "max_goals", 10)
            prob_matrix = np.zeros((max_goals + 1, max_goals + 1))

            for h_goals in range(max_goals + 1):
                for a_goals in range(max_goals + 1):
                    prob_matrix[h_goals, a_goals] = poisson.pmf(
                        h_goals, lambda_home
                    ) * poisson.pmf(a_goals, lambda_away)

            # Calculate outcome probabilities
            prob_home_win = np.sum(np.tril(prob_matrix, -1))
            prob_draw = np.sum(np.diag(prob_matrix))
            prob_away_win = np.sum(np.triu(prob_matrix, 1))

            # Normalize
            total = prob_home_win + prob_draw + prob_away_win
            if total > 0:
                prob_home_win /= total
                prob_draw /= total
                prob_away_win /= total

            return {
                "prob_home_win": prob_home_win,
                "prob_draw": prob_draw,
                "prob_away_win": prob_away_win,
                "lambda_home": lambda_home,
                "lambda_away": lambda_away,
            }

        except Exception:
            return {
                "prob_home_win": 0.33,
                "prob_draw": 0.33,
                "prob_away_win": 0.34,
                "lambda_home": 1.5,
                "lambda_away": 1.2,
            }

    def _calculate_fair_probabilities(self, odds: List[float]) -> Optional[List[float]]:
        """Calculate fair probabilities from odds."""
        try:
            if any(np.isnan(odds)) or any(o <= 1.0 for o in odds):
                return None

            from utils.odds_helpers import get_no_vig_odds_multiway

            fair_odds_tuple = get_no_vig_odds_multiway(odds)
            return [1 / o for o in fair_odds_tuple]

        except Exception:
            return None

    def _get_actual_outcome(self, match_row: pd.Series) -> int:
        """Get actual match outcome (0=Home, 1=Draw, 2=Away)."""
        home_goals = match_row["FTHG"]
        away_goals = match_row["FTAG"]

        if home_goals > away_goals:
            return 0  # Home win
        elif home_goals == away_goals:
            return 1  # Draw
        else:
            return 2  # Away win

    def _evaluate_betting_opportunity(
        self, prediction_row: pd.Series, actual_outcome: int, bankroll: float
    ) -> Tuple[Optional[BacktestResult], float]:
        """Evaluate betting opportunity with proper edge calculation."""
        try:
            # Use betting service to analyze the prediction
            betting_metrics = self.betting_service.analyze_prediction_row(
                prediction_row
            )

            if (
                not betting_metrics
                or betting_metrics["edge"] < self.config.betting_threshold
            ):
                return None, bankroll

            # Calculate stake using Kelly criterion with limits
            kelly_fraction = betting_metrics.get("kelly_fraction", 0)
            if kelly_fraction <= 0:
                return None, bankroll

            # Apply conservative Kelly sizing
            conservative_kelly = min(
                kelly_fraction * 0.5, self.config.max_stake_fraction
            )
            stake = min(self.config.stake_size, bankroll * conservative_kelly)

            if stake <= 0:
                return None, bankroll

            # Determine if bet won
            bet_outcome_map = {"Home": 0, "Draw": 1, "Away": 2}
            bet_outcome_idx = bet_outcome_map[betting_metrics["bet_type"]]

            if bet_outcome_idx == actual_outcome:
                profit = stake * (betting_metrics["soft_odds"] - 1)
            else:
                profit = -stake

            new_bankroll = bankroll + profit

            result = BacktestResult(
                match_id=f"{prediction_row['Home']}_{prediction_row['Away']}_{prediction_row['Date']}",
                date=prediction_row["Date"],
                home_team=prediction_row["Home"],
                away_team=prediction_row["Away"],
                bet_type=betting_metrics["bet_type"],
                stake=stake,
                odds=betting_metrics["soft_odds"],
                profit=profit,
                model_prob=betting_metrics["model_prob"],
                market_prob=betting_metrics["market_prob"],
                edge=betting_metrics["edge"],
                expected_value=betting_metrics["edge"] * stake,
            )

            return result, new_bankroll

        except Exception as e:
            print(f"Error evaluating bet: {e}")
            return None, bankroll

    def _calculate_backtest_summary(
        self,
        predictions_df: pd.DataFrame,
        betting_results: List[BacktestResult],
        final_bankroll: float,
    ) -> BacktestSummary:
        """Calculate comprehensive backtest summary with proper metrics."""

        # Model performance metrics
        accuracy = 0.0
        logloss = 10.0

        if len(predictions_df) > 0:
            try:
                y_true = predictions_df["Actual_Outcome"].values
                y_prob = predictions_df[
                    ["ZIP_Prob_H", "ZIP_Prob_D", "ZIP_Prob_A"]
                ].values

                # Ensure probabilities are valid
                valid_mask = ~np.isnan(y_prob).any(axis=1)
                if valid_mask.sum() > 0:
                    y_true_valid = y_true[valid_mask]
                    y_prob_valid = y_prob[valid_mask]

                    # Normalize probabilities
                    row_sums = y_prob_valid.sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0] = 1  # Avoid division by zero
                    y_prob_valid = y_prob_valid / row_sums

                    # Calculate metrics
                    y_pred = np.argmax(y_prob_valid, axis=1)
                    accuracy = np.mean(y_true_valid == y_pred)

                    # Calculate log loss with epsilon for numerical stability
                    epsilon = 1e-15
                    y_prob_clipped = np.clip(y_prob_valid, epsilon, 1 - epsilon)
                    logloss = log_loss(y_true_valid, y_prob_clipped, labels=[0, 1, 2])

            except Exception as e:
                print(f"Error calculating model metrics: {e}")

        # Betting performance metrics
        if betting_results:
            profits = [bet.profit for bet in betting_results]
            stakes = [bet.stake for bet in betting_results]
            edges = [bet.edge for bet in betting_results]
            wins = [bet.profit > 0 for bet in betting_results]

            total_profit = sum(profits)
            total_staked = sum(stakes)
            roi_percent = 100 * total_profit / total_staked if total_staked > 0 else 0.0
            win_rate = 100 * sum(wins) / len(wins) if wins else 0.0
            avg_edge = np.mean(edges)
        else:
            total_profit = 0.0
            total_staked = 0.0
            roi_percent = 0.0
            win_rate = 0.0
            avg_edge = 0.0

        return BacktestSummary(
            total_bets=len(betting_results),
            total_profit=total_profit,
            total_staked=total_staked,
            roi_percent=roi_percent,
            win_rate=win_rate,
            avg_edge=avg_edge,
            final_bankroll=final_bankroll,
            predictions_df=predictions_df,
            betting_results=betting_results,
        )
