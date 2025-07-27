import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.odds_helpers import get_no_vig_odds_multiway
from zsd_poisson_model import ZSDPoissonModel


def backtest_model_cross_season(
    data, league, train_season, test_season, model_class, decay_rate=0.001
):
    all_predictions = []

    # Sort and split data
    data = data.sort_values(["Season", "Wk", "Date"])
    train_data = data[
        (data["Season"] == train_season) & (data["League"] == league)
    ].copy()
    test_data = data[
        (data["Season"] == test_season) & (data["League"] == league)
    ].copy()

    teams = sorted(list(set(train_data["Home"]).union(test_data["Home"])))

    max_week = test_data["Wk"].max()

    for week in range(1, max_week + 1):
        # Training = previous season + test season up to previous week
        past_test_matches = test_data[test_data["Wk"] < week]
        training_data = pd.concat([train_data, past_test_matches])

        # Fit the model on available history
        try:
            model = model_class(
                teams=teams, played_matches=training_data, decay_rate=decay_rate
            )
        except RuntimeError as e:
            print(f"Week {week}: Model failed to converge. Skipping.")
            continue

        # Predict current week
        current_week_matches = test_data[test_data["Wk"] == week].copy()
        for _, row in current_week_matches.iterrows():
            home = row["Home"]
            away = row["Away"]
            date = row["Date"]
            odds = {"PSCH": row["PSCH"], "PSCD": row["PSCD"], "PSCA": row["PSCA"]}

            # Predictions
            mov_pred = model.predict_match_mov(home, away)
            probs_mov = model.outcome_probabilities(
                mov_pred["predicted_mov"], -mov_pred["predicted_mov"]
            )
            zip_pred = model.predict_zip_adjusted_outcomes(home, away)
            # Compute fair (no-vig) probabilities from closing odds

            try:
                fair_odds = get_no_vig_odds_multiway(
                    [odds["PSCH"], odds["PSCD"], odds["PSCA"]]
                )
                fair_probs = [1 / o for o in fair_odds]
                fair_probs_sum = sum(fair_probs)
                fair_probs = [
                    p / fair_probs_sum for p in fair_probs
                ]  # Normalize to sum to 1
            except Exception as e:
                print(f"Error computing fair odds for match {home} vs {away}: {e}")
                fair_probs = [np.nan, np.nan, np.nan]

            all_predictions.append(
                {
                    "Date": date,
                    "Season": row["Season"],
                    "Wk": row["Wk"],
                    "Home": home,
                    "Away": away,
                    "FTHG": row["FTHG"],
                    "FTAG": row["FTAG"],
                    "PSCH": odds["PSCH"],
                    "PSCD": odds["PSCD"],
                    "PSCA": odds["PSCA"],
                    "FairProb_H": fair_probs[0],
                    "FairProb_D": fair_probs[1],
                    "FairProb_A": fair_probs[2],
                    "PPI_H": row["hPPI"],
                    "PPI_A": row["aPPI"],
                    "PPI_Diff": row["PPI_Diff"],
                    **probs_mov,
                    "P_ZIP(H)": round(zip_pred["P(Home Win)"], 4),
                    "P_ZIP(D)": round(zip_pred["P(Draw)"], 4),
                    "P_ZIP(A)": round(zip_pred["P(Away Win)"], 4),
                }
            )

    return pd.DataFrame(all_predictions)


def evaluate_predictions(
    df, pred_cols=("P_ZIP(H)", "P_ZIP(D)", "P_ZIP(A)"), label="Raw ZIP"
):
    actual = []
    for _, row in df.iterrows():
        if row["FTHG"] > row["FTAG"]:
            actual.append(0)
        elif row["FTHG"] == row["FTAG"]:
            actual.append(1)
        else:
            actual.append(2)

    y_prob = df[list(pred_cols)].values
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    y_true = np.array(actual)

    logloss = log_loss(y_true, y_prob)
    brier = np.mean(np.sum((y_prob - np.eye(3)[y_true]) ** 2, axis=1))
    y_pred = np.argmax(y_prob, axis=1)
    accuracy = accuracy_score(y_true, y_pred)

    profits = []
    for _, row in df.iterrows():
        probs = [row[pred_cols[0]], row[pred_cols[1]], row[pred_cols[2]]]
        fair_probs = [row["FairProb_H"], row["FairProb_D"], row["FairProb_A"]]
        odds = [row["PSCH"], row["PSCD"], row["PSCA"]]

        edges = [p - fp for p, fp in zip(probs, fair_probs)]
        max_edge = max(edges)
        bet_index = edges.index(max_edge)
        if max_edge > 0.02:
            outcome = (
                0
                if row["FTHG"] > row["FTAG"]
                else 1
                if row["FTHG"] == row["FTAG"]
                else 2
            )
            stake = 1.0
            if bet_index == outcome:
                profit = stake * (odds[bet_index] - 1)
            else:
                profit = -stake
            profits.append(profit)

    return {
        "Label": label,
        "Log Loss": round(logloss, 4),
        "Brier Score": round(brier, 4),
        "Accuracy": round(accuracy, 4),
        "Total Bets": len(profits),
        "Total Profit": round(sum(profits), 2),
        "ROI (%)": round(100 * sum(profits) / len(profits), 2) if profits else 0.0,
    }


def calibrate_with_ppi_holdout(df, test_size=0.2, random_state=42):
    # 1. Split the prediction results
    calibration_df, holdout_df = train_test_split(
        df.dropna(subset=["PPI_H", "PPI_A", "PPI_Diff"]),
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
    )

    # 2. Fit logistic regression on calibration set
    X_train = calibration_df[["P_ZIP(H)", "P_ZIP(D)", "P_ZIP(A)", "PPI_A", "PPI_Diff"]]
    y_train = calibration_df.apply(
        lambda row: (
            0 if row["FTHG"] > row["FTAG"] else 1 if row["FTHG"] == row["FTAG"] else 2
        ),
        axis=1,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(multi_class="multinomial", max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # 3. Apply calibration to holdout
    X_holdout = holdout_df[["P_ZIP(H)", "P_ZIP(D)", "P_ZIP(A)", "PPI_A", "PPI_Diff"]]
    X_holdout_scaled = scaler.transform(X_holdout)
    calibrated_probs = model.predict_proba(X_holdout_scaled)

    holdout_df = holdout_df.copy()
    holdout_df["P_CAL(H)"] = calibrated_probs[:, 0]
    holdout_df["P_CAL(D)"] = calibrated_probs[:, 1]
    holdout_df["P_CAL(A)"] = calibrated_probs[:, 2]

    return holdout_df


matches = pd.read_csv("historical_ppi_and_odds.csv", dtype={"Wk": int})
results_df = backtest_model_cross_season(
    data=matches,
    league="Ekstraklasa",
    train_season="2023-2024",
    test_season="2024-2025",
    model_class=ZSDPoissonModel,
)

print(results_df.head(30))

# Calibrate using 80% of data and test on 20%
calibrated_holdout = calibrate_with_ppi_holdout(results_df)

# Evaluate calibrated predictions
metrics_cal = evaluate_predictions(
    calibrated_holdout, pred_cols=("P_CAL(H)", "P_CAL(D)", "P_CAL(A)")
)

print("\n📊 Calibrated ZIP + PPI (Holdout) Evaluation:")
for k, v in metrics_cal.items():
    print(f"{k}: {v}")
