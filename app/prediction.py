import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils.odds_helpers import get_no_vig_odds_multiway

# A value > 1 indicates potential value bet
VALUE_THRESHOLD = 1.05  # Typically use >1.05 or >1.1 to account for model uncertainty
FEATURE_COLUMNS = ["hRPI", "aRPI", "RPI_Diff"]


def convert_odds_to_probability(odds):
    return 1 / odds


def get_fair_odds_future(r):
    odds = get_no_vig_odds_multiway(list(r[["PSH", "PSD", "PSA"]]))
    r["PSH_fair_odds"] = odds[0]
    r["PSD_fair_odds"] = odds[1]
    r["PSA_fair_odds"] = odds[2]

    return r


def get_fair_odds_past(r):
    odds = get_no_vig_odds_multiway(list(r[["PSCH", "PSCD", "PSCA"]]))
    r["PSCH_fair_odds"] = odds[0]
    r["PSCD_fair_odds"] = odds[1]
    r["PSCA_fair_odds"] = odds[2]

    return r


# Load historical data with non-zero RPI values
historical_df = pd.read_csv("historical_rpi_and_odds.csv")
print(f"Historical Data Size: {historical_df.shape[0]}")
historical_df = historical_df.dropna(how="any")
print(f"Historical Data Size After removing NaNs: {historical_df.shape[0]}")

# Filter out matches with zero RPI values
valid_data = historical_df[
    historical_df["Wk"] >= 10
].copy()  # historical_df[(historical_df["hRPI"] != 0) & (historical_df["aRPI"] != 0)]

valid_data["PSCH_fair_odds"] = 0
valid_data["PSCD_fair_odds"] = 0
valid_data["PSCA_fair_odds"] = 0

valid_data = valid_data.apply(get_fair_odds_past, axis="columns")

# Features and targets
X = valid_data[FEATURE_COLUMNS].values
y_home = valid_data["PSCH_fair_odds"].values
y_draw = valid_data["PSCD_fair_odds"].values
y_away = valid_data["PSCA_fair_odds"].values

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_home_train, y_home_test = train_test_split(
    X, y_home, test_size=0.2, random_state=42
)
_, _, y_draw_train, y_draw_test = train_test_split(
    X, y_draw, test_size=0.2, random_state=42
)
_, _, y_away_train, y_away_test = train_test_split(
    X, y_away, test_size=0.2, random_state=42
)

# Train separate models for each outcome
home_model = RandomForestRegressor(n_estimators=100, random_state=42)
draw_model = RandomForestRegressor(n_estimators=100, random_state=42)
away_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit models on training data
home_model.fit(X_train, y_home_train)
draw_model.fit(X_train, y_draw_train)
away_model.fit(X_train, y_away_train)

# Evaluate model performance on test data
home_score = home_model.score(X_test, y_home_test)
draw_score = draw_model.score(X_test, y_draw_test)
away_score = away_model.score(X_test, y_away_test)

print(f"Home model R² score: {home_score:.4f}")
print(f"Draw model R² score: {draw_score:.4f}")
print(f"Away model R² score: {away_score:.4f}")

# After evaluation, you can retrain on the full dataset if the models perform well
home_model.fit(X, y_home)
draw_model.fit(X, y_draw)
away_model.fit(X, y_away)

# Load future matches
future_matches = pd.read_csv("latest_rpi_and_odds.csv")

future_matches["PSH_fair_odds"] = 0
future_matches["PSD_fair_odds"] = 0
future_matches["PSA_fair_odds"] = 0

future_matches = future_matches.apply(get_fair_odds_future, axis="columns")

# Normalize probabilities to remove the overround
future_matches["PSH_fair_prob"] = convert_odds_to_probability(
    future_matches["PSH_fair_odds"]
)
future_matches["PSD_fair_prob"] = convert_odds_to_probability(
    future_matches["PSD_fair_odds"]
)
future_matches["PSA_fair_prob"] = convert_odds_to_probability(
    future_matches["PSA_fair_odds"]
)

# Prepare features for prediction
X_future = future_matches[FEATURE_COLUMNS].values

# Predict odds
future_matches["pred_PSH"] = home_model.predict(X_future)
future_matches["pred_PSD"] = draw_model.predict(X_future)
future_matches["pred_PSA"] = away_model.predict(X_future)

# Convert predicted odds to probabilities
future_matches["pred_PSH_prob"] = convert_odds_to_probability(
    future_matches["pred_PSH"]
)
future_matches["pred_PSD_prob"] = convert_odds_to_probability(
    future_matches["pred_PSD"]
)
future_matches["pred_PSA_prob"] = convert_odds_to_probability(
    future_matches["pred_PSA"]
)

# Normalize predicted probabilities to ensure they sum to 1
prob_sum = (
    future_matches["pred_PSH_prob"]
    + future_matches["pred_PSD_prob"]
    + future_matches["pred_PSA_prob"]
)
future_matches["pred_PSH_prob_norm"] = future_matches["pred_PSH_prob"] / prob_sum
future_matches["pred_PSD_prob_norm"] = future_matches["pred_PSD_prob"] / prob_sum
future_matches["pred_PSA_prob_norm"] = future_matches["pred_PSA_prob"] / prob_sum

# Calculate value (your probability / bookmaker probability)
future_matches["PSH_value"] = (
    future_matches["pred_PSH_prob_norm"] / future_matches["PSH_fair_prob"]
)
future_matches["PSD_value"] = (
    future_matches["pred_PSD_prob_norm"] / future_matches["PSD_fair_prob"]
)
future_matches["PSA_value"] = (
    future_matches["pred_PSA_prob_norm"] / future_matches["PSA_fair_prob"]
)

# Filter value bets
home_value_bets: pd.DataFrame = future_matches[
    future_matches["PSH_value"] > VALUE_THRESHOLD
]
draw_value_bets = future_matches[future_matches["PSD_value"] > VALUE_THRESHOLD]
away_value_bets = future_matches[future_matches["PSA_value"] > VALUE_THRESHOLD]

print("Home Value Bets:")
print(home_value_bets[["Home", "Away", "PSH", "pred_PSH", "PSH_value"]])

print("\nDraw Value Bets:")
print(draw_value_bets[["Home", "Away", "PSD", "pred_PSD", "PSD_value"]])

print("\nAway Value Bets:")
print(away_value_bets[["Home", "Away", "PSA", "pred_PSA", "PSA_value"]])

# Plot value vs. odds
plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.scatter(future_matches["PSH"], future_matches["PSH_value"])
plt.axhline(y=VALUE_THRESHOLD, color="r", linestyle="--")
plt.title("Home Win Value")
plt.xlabel("Odds")
plt.ylabel("Value Ratio")

plt.subplot(1, 3, 2)
plt.scatter(future_matches["PSD"], future_matches["PSD_value"])
plt.axhline(y=VALUE_THRESHOLD, color="r", linestyle="--")
plt.title("Draw Value")
plt.xlabel("Odds")
plt.ylabel("Value Ratio")

plt.subplot(1, 3, 3)
plt.scatter(future_matches["PSA"], future_matches["PSA_value"])
plt.axhline(y=VALUE_THRESHOLD, color="r", linestyle="--")
plt.title("Away Win Value")
plt.xlabel("Odds")
plt.ylabel("Value Ratio")

plt.tight_layout()
plt.show()
