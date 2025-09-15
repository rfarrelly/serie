from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from domains.betting.services import BettingAnalysisService
from sklearn.metrics import accuracy_score, log_loss

from .entities import BacktestConfig, BacktestResult, BacktestSummary


class BacktestingService:
    """Domain service for backtesting model performance and betting strategies."""

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
        """Run a cross-season backtest for a specific league."""

        # Prepare data
        train_data, test_data = self._prepare_backtest_data(
            data, train_season, test_season, league
        )

        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError(
                f"Insufficient data for {league} {train_season}/{test_season}"
            )

        # Initialize results storage
        all_predictions = []
        betting_results = []
        bankroll = 100.0

        print(f"Backtesting {league}: {train_season} -> {test_season}")
        print(f"Training matches: {len(train_data)}, Test matches: {len(test_data)}")

        # Get unique weeks for testing
        test_weeks = sorted(test_data["Wk"].unique())

        for week_num, week in enumerate(test_weeks, 1):
            if week_num < self.config.min_training_weeks:
                continue

            try:
                # Prepare training data (previous season + previous weeks in test season)
                past_test_data = test_data[test_data["Wk"] < week]
                combined_training = pd.concat(
                    [train_data, past_test_data], ignore_index=True
                )

                # Fit model
                model = self._create_and_fit_model(
                    model_class, combined_training, model_params
                )

                # Get current week matches
                week_matches = test_data[test_data["Wk"] == week].copy()

                # Generate predictions for the week
                week_predictions, week_betting_results, bankroll = self._process_week(
                    model, week_matches, bankroll
                )

                all_predictions.extend(week_predictions)
                betting_results.extend(week_betting_results)

                print(
                    f"Week {week}: {len(week_predictions)} predictions, {len(week_betting_results)} bets"
                )

            except Exception as e:
                print(f"Error in week {week}: {str(e)}")
                continue

        # Calculate final metrics
        predictions_df = pd.DataFrame(all_predictions)
        summary = self._calculate_backtest_summary(
            predictions_df, betting_results, bankroll
        )

        return summary

    def _prepare_backtest_data(
        self, data: pd.DataFrame, train_season: str, test_season: str, league: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare and filter data for backtesting."""

        # Filter and prepare data
        train_data = (
            data[(data["Season"] == train_season) & (data["League"] == league)]
            .copy()
            .sort_values(["Date"])
        )

        test_data = (
            data[(data["Season"] == test_season) & (data["League"] == league)]
            .copy()
            .sort_values(["Wk", "Date"])
        )

        # Remove promoted and relegated teams
        teams_train_data = set(pd.concat([train_data["Home"], train_data["Away"]]))
        teams_test_data = set(pd.concat([test_data["Home"], test_data["Away"]]))
        common_teams = teams_train_data & teams_test_data

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
        """Create and fit a model instance."""
        model_params = model_params or {}

        if isinstance(model_class, type):
            from models.core import ModelConfig

            model = model_class(model_params.get("config", ModelConfig()))
        else:
            model = model_class(**model_params)

        model.fit(training_data)
        return model

    def _process_week(
        self, model, week_matches: pd.DataFrame, bankroll: float
    ) -> Tuple[List[Dict], List[BacktestResult], float]:
        """Process predictions and betting for a single week."""

        week_predictions = []
        week_betting_results = []

        for _, match in week_matches.iterrows():
            try:
                # Generate prediction
                prediction_result = self._predict_single_match(model, match)
                if prediction_result:
                    week_predictions.append(prediction_result)

                    # Evaluate betting opportunity
                    betting_result, bankroll = self._evaluate_betting_opportunity(
                        pd.Series(prediction_result),
                        prediction_result["Actual_Outcome"],
                        bankroll,
                    )
                    if betting_result:
                        week_betting_results.append(betting_result)

            except Exception as e:
                print(f"Error processing match {match['Home']} vs {match['Away']}: {e}")
                continue

        return week_predictions, week_betting_results, bankroll

    def _predict_single_match(self, model, match_row: pd.Series) -> Optional[Dict]:
        """Generate prediction for a single match."""
        try:
            home_team = match_row["Home"]
            away_team = match_row["Away"]

            # Get predictions for all methods
            predictions = self._get_all_method_predictions(model, home_team, away_team)

            # Extract odds
            sharp_odds = [
                match_row.get("PSH", np.nan),
                match_row.get("PSD", np.nan),
                match_row.get("PSA", np.nan),
            ]

            soft_odds = [
                match_row.get("B365H", np.nan),
                match_row.get("B365D", np.nan),
                match_row.get("B365A", np.nan),
            ]

            # Calculate fair probabilities from sharp odds
            fair_probs = self._calculate_fair_probabilities(sharp_odds)

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
                "PSH": sharp_odds[0],
                "PSD": sharp_odds[1],
                "PSA": sharp_odds[2],
                "B365H": soft_odds[0],
                "B365D": soft_odds[1],
                "B365A": soft_odds[2],
                "Fair_Prob_H": fair_probs[0],
                "Fair_Prob_D": fair_probs[1],
                "Fair_Prob_A": fair_probs[2],
                # Model outputs
                "Lambda_Home": predictions["zip"]["lambda_home"],
                "Lambda_Away": predictions["zip"]["lambda_away"],
            }

        except Exception as e:
            print(
                f"Error predicting match {match_row.get('Home', 'Unknown')} vs {match_row.get('Away', 'Unknown')}: {e}"
            )
            return None

    def _get_all_method_predictions(
        self, model, home_team: str, away_team: str
    ) -> Dict[str, Dict]:
        """Get predictions for all methods from the model."""
        try:
            if home_team not in model.team_index or away_team not in model.team_index:
                # Return default predictions for unknown teams
                default_prediction = {
                    "prob_home_win": 0.33,
                    "prob_draw": 0.33,
                    "prob_away_win": 0.34,
                    "lambda_home": 1.5,
                    "lambda_away": 1.2,
                }
                return {
                    "poisson": default_prediction.copy(),
                    "zip": default_prediction.copy(),
                    "mov": default_prediction.copy(),
                }

            predictions = {}
            for method in ["poisson", "zip", "mov"]:
                try:
                    pred = model.predict_match(home_team, away_team, method=method)
                    predictions[method] = {
                        "prob_home_win": pred.prob_home_win,
                        "prob_draw": pred.prob_draw,
                        "prob_away_win": pred.prob_away_win,
                        "lambda_home": pred.lambda_home,
                        "lambda_away": pred.lambda_away,
                    }
                except:
                    predictions[method] = self._calculate_basic_prediction(
                        model, home_team, away_team
                    )

            return predictions

        except Exception as e:
            print(f"Error in _get_all_method_predictions: {e}")
            default_prediction = {
                "prob_home_win": 0.33,
                "prob_draw": 0.33,
                "prob_away_win": 0.34,
                "lambda_home": 1.5,
                "lambda_away": 1.2,
            }
            return {
                "poisson": default_prediction.copy(),
                "zip": default_prediction.copy(),
                "mov": default_prediction.copy(),
            }

    def _calculate_basic_prediction(
        self, model, home_team: str, away_team: str
    ) -> Dict:
        """Calculate basic prediction when advanced methods aren't available."""
        try:
            home_idx = model.team_index[home_team]
            away_idx = model.team_index[away_team]

            # Calculate expected goals using model parameters
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

            # Simple outcome probability calculation
            from scipy.stats import poisson

            max_goals = getattr(model.config, "max_goals", 15)
            prob_matrix = np.zeros((max_goals + 1, max_goals + 1))

            for h_goals in range(max_goals + 1):
                for a_goals in range(max_goals + 1):
                    prob_matrix[h_goals, a_goals] = poisson.pmf(
                        h_goals, lambda_home
                    ) * poisson.pmf(a_goals, lambda_away)

            prob_home_win = np.sum(
                [
                    prob_matrix[h, a]
                    for h in range(max_goals + 1)
                    for a in range(max_goals + 1)
                    if h > a
                ]
            )
            prob_draw = np.sum([prob_matrix[h, h] for h in range(max_goals + 1)])
            prob_away_win = np.sum(
                [
                    prob_matrix[h, a]
                    for h in range(max_goals + 1)
                    for a in range(max_goals + 1)
                    if h < a
                ]
            )

            # Normalize
            total_prob = prob_home_win + prob_draw + prob_away_win
            if total_prob > 0:
                prob_home_win /= total_prob
                prob_draw /= total_prob
                prob_away_win /= total_prob

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

    def _calculate_fair_probabilities(self, odds: List[float]) -> List[float]:
        """Calculate fair (no-vig) probabilities from bookmaker odds."""
        try:
            if any(np.isnan(odds)) or any(o <= 1.0 for o in odds):
                return [np.nan, np.nan, np.nan]

            from utils.odds_helpers import get_no_vig_odds_multiway

            fair_odds_tuple = get_no_vig_odds_multiway(odds)
            fair_probs = [1 / o for o in fair_odds_tuple]
            return fair_probs

        except Exception:
            return [np.nan, np.nan, np.nan]

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
        """Evaluate betting opportunity and return result."""
        try:
            betting_metrics = self.betting_service.analyze_prediction_row(
                prediction_row
            )

            if (
                betting_metrics is None
                or betting_metrics["edge"] < self.config.betting_threshold
            ):
                return None, bankroll

            # Kelly criterion for stake sizing
            kelly_fraction = betting_metrics["edge"] / (
                betting_metrics["soft_odds"] - 1
            )
            stake = min(
                self.config.stake_size,
                bankroll * min(kelly_fraction, self.config.max_stake_fraction),
            )

            # Calculate profit based on actual outcome
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
        """Calculate comprehensive backtest summary."""

        # Model performance metrics
        if len(predictions_df) > 0:
            y_true = predictions_df["Actual_Outcome"].values
            y_prob = predictions_df[["ZIP_Prob_H", "ZIP_Prob_D", "ZIP_Prob_A"]].values

            # Ensure probabilities sum to 1
            y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

            y_pred = np.argmax(y_prob, axis=1)
            accuracy = accuracy_score(y_true, y_pred)
            logloss = log_loss(y_true, y_prob, labels=[0, 1, 2])
        else:
            accuracy = 0.0
            logloss = 10.0

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
