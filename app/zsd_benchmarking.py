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
    files = [
        "DATA/FBREF/Premier-League/Premier-League_2024-2025.csv",
        "DATA/FBREF/Premier-League/Premier-League_2023-2024.csv",
        "DATA/FBREF/Premier-League/Premier-League_2022-2023.csv",
    ]
    matches = pd.concat([pd.read_csv(file, dtype={"Wk": int}) for file in files])
    results = benchmark_model(df=matches, model_class=ZSDPoissonModel)
    print(results)


def sanity_check():
    ### Single Match ###
    matches = pd.read_csv("zsd_poisson_test_data.csv", dtype={"Wk": int})
    matches = format_date(matches)
    played_matches = matches[:206].copy()
    model = ZSDPoissonModel(played_matches=played_matches, decay_rate=0.001)

    # Core predictions from the model
    result = model.predict_match_mov("Bournemouth", "Everton")

    # Raw goal estimates
    lambda_home = result["home_goals_est"]
    lambda_away = result["away_goals_est"]

    # Get outcome probabilities from logistic-MOV model
    probs = model.outcome_probabilities(
        lambda_home - lambda_away, lambda_away - lambda_home
    )
    result |= probs

    # Generate Poisson and ZIP-adjusted matrices
    poisson_matrix = model.poisson_prob_matrix(lambda_home, lambda_away, max_goals=15)
    zip_adj_matrix = model.zip_adjustment_matrix(max_goals=15)
    zip_poisson_matrix = poisson_matrix * zip_adj_matrix.values
    zip_adj_outcomes = model.predict_zip_adjusted_outcomes(
        home_team="Bournemouth", away_team="Everton", max_goals=15
    )

    # Collapse to outcome probabilities
    result["P_Poisson(H)"] = round(np.tril(poisson_matrix, -1).sum(), 2)
    result["P_Poisson(D)"] = round(np.trace(poisson_matrix), 2)
    result["P_Poisson(A)"] = round(np.triu(poisson_matrix, 1).sum(), 2)

    result["P_ZIP(H)"] = round(np.tril(zip_poisson_matrix, -1).sum(), 2)
    result["P_ZIP(D)"] = round(np.trace(zip_poisson_matrix), 2)
    result["P_ZIP(A)"] = round(np.triu(zip_poisson_matrix, 1).sum(), 2)

    result["P_ZIP_ADJ(H)"] = round(zip_adj_outcomes["P(Home Win)"], 2)
    result["P_ZIP_ADJ(D)"] = round(zip_adj_outcomes["P(Draw)"], 2)
    result["P_ZIP_ADJ(A)"] = round(zip_adj_outcomes["P(Away Win)"], 2)

    preds_df = pd.DataFrame(data=result, index=[0])
    preds_df.to_csv("temp.csv", index=False)
    print(preds_df)


def main():
    run_benchmarking()
    sanity_check()


if __name__ == "__main__":
    main()
