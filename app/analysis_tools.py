import numpy as np
import pandas as pd
from scipy.stats import poisson
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


files = [
    "DATA/FBREF/Premier-League/Premier-League_2024-2025.csv",
    "DATA/FBREF/Premier-League/Premier-League_2023-2024.csv",
    "DATA/FBREF/Premier-League/Premier-League_2022-2023.csv",
]
matches = pd.concat([pd.read_csv(file, dtype={"Wk": int}) for file in files])
decays_to_test = np.linspace(0.01, 0.1, 10)
best_decay, cv_results = tune_decay_rate(
    df=matches, decay_rates=decays_to_test, scoring="log_likelihood"
)

print("Best decay rate:", best_decay)
for d, score in cv_results:
    print(f"Decay {d:.5f} => Score: {score:.4f}")
