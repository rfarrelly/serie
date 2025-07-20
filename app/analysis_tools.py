import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import log_loss, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from zsd_poisson_model import ZSDPoissonModel


def tune_decay_rate(df, decay_rates, n_splits=5, scoring="sse"):
    """
    Parameters:
        df: pd.DataFrame — must include ['Date', 'Home', 'Away', 'FTHG', 'FTAG']
        decay_rates: list of float — values to try for exponential time decay
        n_splits: int — number of CV folds
        scoring: str — "sse" or "log_likelihood"

    Returns:
        best_decay_rate: float
        results: list of (decay_rate, avg_score) tuples
    """
    assert scoring in ["sse", "log_likelihood"]

    teams = sorted(list(set(df["Home"]).union(df["Away"])))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for decay in decay_rates:
        fold_scores = []

        for train_idx, val_idx in tscv.split(df):
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()

            if val_df.empty:
                continue

            try:
                model = ZSDPoissonModel(teams, train_df, decay_rate=decay)
            except RuntimeError as e:
                print(f"[decay {decay:.5f}] fold failed: {e}")
                continue

            score = 0
            skipped = 0
            for _, row in val_df.iterrows():
                pred = model.predict_match_mov(row["Home"], row["Away"])

                if scoring == "sse":
                    home_error = (pred["home_goals_est"] - row["FTHG"]) ** 2
                    away_error = (pred["away_goals_est"] - row["FTAG"]) ** 2
                    score += home_error + away_error

                elif scoring == "log_likelihood":
                    lambda_home = pred["home_goals_est"]
                    lambda_away = pred["away_goals_est"]
                    if (
                        np.isfinite(lambda_home)
                        and np.isfinite(lambda_away)
                        and lambda_home > 0
                        and lambda_away > 0
                    ):
                        score += poisson.logpmf(row["FTHG"], lambda_home)
                        score += poisson.logpmf(row["FTAG"], lambda_away)
                    else:
                        print(f"Row skipped: \n{row}")
                        skipped += 1

            if skipped > 0:
                print(f"[decay {decay:.5f}] skipped {skipped} invalid matches")

            fold_scores.append(score / len(val_df))  # average per match

        avg_score = np.mean(fold_scores)
        results.append((decay, avg_score))

    if scoring == "sse":
        best = min(results, key=lambda x: x[1])  # lower is better
    else:
        best = max(results, key=lambda x: x[1])  # higher LL is better

    return best[0], results


def benchmark_model(
    df, model_class: ZSDPoissonModel, teams=None, decay_rate=None, n_splits=5
):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_metrics = {"log_loss": [], "rmse_home": [], "rmse_away": [], "brier": []}

    for train_idx, val_idx in tscv.split(df):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx]

        if teams:
            model = model_class(
                teams=teams, played_matches=train_df, decay_rate=decay_rate
            )
        else:
            model = model_class(played_matches=train_df, decay_rate=decay_rate)

        y_true_probs = []
        y_pred_probs = []
        y_true_goals = []
        y_pred_goals = []

        for _, row in val_df.iterrows():
            pred = model.predict_match_mov(row["Home"], row["Away"])
            lambda_h = pred["home_goals_est"]
            lambda_a = pred["away_goals_est"]

            # Store predictions
            y_pred_goals.append((lambda_h, lambda_a))
            y_true_goals.append((row["FTHG"], row["FTAG"]))

            # Build full probability matrix
            prob_matrix = model.poisson_prob_matrix(lambda_h, lambda_a, max_goals=10)
            p_hw = np.tril(prob_matrix, -1).sum()
            p_aw = np.triu(prob_matrix, 1).sum()
            p_draw = np.trace(prob_matrix)

            y_pred_probs.append([p_hw, p_draw, p_aw])

            # One-hot true outcome
            if row["FTHG"] > row["FTAG"]:
                y_true_probs.append([1, 0, 0])
            elif row["FTHG"] < row["FTAG"]:
                y_true_probs.append([0, 0, 1])
            else:
                y_true_probs.append([0, 1, 0])

        # Metrics
        true_goals = np.array(y_true_goals)
        pred_goals = np.array(y_pred_goals)
        all_metrics["rmse_home"].append(
            root_mean_squared_error(true_goals[:, 0], pred_goals[:, 0])
        )
        all_metrics["rmse_away"].append(
            root_mean_squared_error(true_goals[:, 1], pred_goals[:, 1])
        )

        pred_probs = np.array(y_pred_probs)
        true_probs = np.array(y_true_probs)

        eps = 1e-15
        logloss = -np.mean(
            np.sum(true_probs * np.log(np.clip(pred_probs, eps, 1)), axis=1)
        )
        all_metrics["log_loss"].append(logloss)

        brier = np.mean(np.sum((pred_probs - true_probs) ** 2, axis=1))
        all_metrics["brier"].append(brier)

    return {k: np.mean(v) for k, v in all_metrics.items()}
