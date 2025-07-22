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
    preds = model.predict_match(
        home_team="Bournemouth", away_team="Everton", max_goals=15
    )
    pd.DataFrame(data=preds, index=[0]).to_csv("temp.csv", index=False)


def main():
    run_benchmarking()
    sanity_check()


if __name__ == "__main__":
    main()
