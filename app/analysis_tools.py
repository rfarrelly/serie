import numpy as np
from scipy.stats import poisson
from sklearn.metrics import root_mean_squared_error
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


def benchmark_model(model_class, df, max_goals=15):
    log_losses = []
    brier_scores = []
    home_goals_true = []
    away_goals_true = []
    home_goals_pred = []
    away_goals_pred = []

    model = model_class(played_matches=df)

    for _, row in df.iterrows():
        home_team = row["Home"]
        away_team = row["Away"]
        actual_fthg = row["FTHG"]
        actual_ftag = row["FTAG"]

        try:
            pred = model.predict_zip_adjusted_outcomes(
                home_team=home_team, away_team=away_team, max_goals=max_goals
            )
        except KeyError:
            continue  # skip unknown teams

        # Outcome classification
        if actual_fthg > actual_ftag:
            actual_result = "P(Home Win)"
            actual_vector = np.array([1, 0, 0])
        elif actual_fthg == actual_ftag:
            actual_result = "P(Draw)"
            actual_vector = np.array([0, 1, 0])
        else:
            actual_result = "P(Away Win)"
            actual_vector = np.array([0, 0, 1])

        predicted_vector = np.array(
            [
                pred["P(Home Win)"],
                pred["P(Draw)"],
                pred["P(Away Win)"],
            ]
        )

        # Log loss
        prob = pred.get(actual_result, 1e-10)
        log_losses.append(-np.log(prob))

        # Brier score
        brier_scores.append(np.sum((actual_vector - predicted_vector) ** 2))

        # Goal RMSE (expected goals from ZIP matrix)
        goal_matrix = pred["poisson_matrix"]
        est_home = np.sum(goal_matrix * np.arange(goal_matrix.shape[0])[:, None])
        est_away = np.sum(goal_matrix * np.arange(goal_matrix.shape[1])[None, :])

        home_goals_true.append(actual_fthg)
        away_goals_true.append(actual_ftag)
        home_goals_pred.append(est_home)
        away_goals_pred.append(est_away)

    return {
        "log_loss": np.mean(log_losses),
        "brier": np.mean(brier_scores),
        "rmse_home": root_mean_squared_error(home_goals_true, home_goals_pred),
        "rmse_away": root_mean_squared_error(away_goals_true, away_goals_pred),
    }
