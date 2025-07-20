import numpy as np
import pandas as pd
from analysis_tools import benchmark_model, tune_decay_rate
from utils.datetime_helpers import format_date
from zsd_poisson_model import ZSDPoissonModel


def decay_rate_tuning():
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


def run_benchmarking():
    # matches = pd.read_csv("zsd_poisson_test_data.csv", dtype={"Wk": int})
    # matches = format_date(matches)
    files = [
        "DATA/FBREF/Premier-League/Premier-League_2024-2025.csv",
        "DATA/FBREF/Premier-League/Premier-League_2023-2024.csv",
        "DATA/FBREF/Premier-League/Premier-League_2022-2023.csv",
    ]
    matches = pd.concat([pd.read_csv(file, dtype={"Wk": int}) for file in files])
    teams = sorted(list(set(matches["Home"]).union(matches["Away"])))
    results = benchmark_model(
        teams=teams, df=matches, model_class=ZSDPoissonModel, decay_rate=0.001
    )
    print(results)


def sanity_check():
    matches = pd.read_csv("zsd_poisson_test_data.csv", dtype={"Wk": int})
    matches = format_date(matches)
    played_matches = matches[matches["Wk"] <= 20].copy()
    unplayed_matches = matches[matches["Wk"] > 20]
    model = ZSDPoissonModel(played_matches=played_matches)

    results = []
    for fixture in unplayed_matches.itertuples(index=False):
        week, date, time, home_team, away_team = (
            fixture.Wk,
            fixture.Date,
            fixture.Time,
            fixture.Home,
            fixture.Away,
        )

        # Core predictions from the model
        result = model.predict_match_mov(home_team, away_team)

        # Raw goal estimates
        lambda_home = result["home_goals_est"]
        lambda_away = result["away_goals_est"]

        # Get outcome probabilities from logistic-MOV model
        probs = model.outcome_probabilities(
            lambda_home - lambda_away, lambda_away - lambda_home
        )
        result |= probs

        # Generate Poisson and ZIP-adjusted matrices
        poisson_matrix = model.poisson_prob_matrix(
            lambda_home, lambda_away, max_goals=10
        )
        zip_adj_matrix = model.zip_adjustment_matrix(max_goals=10)
        zip_poisson_matrix = poisson_matrix * zip_adj_matrix.values

        # Add fixture metadata
        result["Wk"] = week
        result["Date"] = date
        result["Time"] = time
        result["Home"] = home_team
        result["Away"] = away_team

        # Collapse to outcome probabilities
        result["P_Poisson(Home Win)"] = np.tril(poisson_matrix, -1).sum()
        result["P_Poisson(Draw)"] = np.trace(poisson_matrix)
        result["P_Poisson(Away Win)"] = np.triu(poisson_matrix, 1).sum()

        result["P_ZIP(Home Win)"] = np.tril(zip_poisson_matrix, -1).sum()
        result["P_ZIP(Draw)"] = np.trace(zip_poisson_matrix)
        result["P_ZIP(Away Win)"] = np.triu(zip_poisson_matrix, 1).sum()

        results.append(result)


def main():
    run_benchmarking()


if __name__ == "__main__":
    main()
