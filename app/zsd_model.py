import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm, poisson
from utils.datetime_helpers import format_date

data = pd.read_csv("zsd_poisson_test_data.csv")
data = format_date(data)
data["Game Total"] = data["FTHG"] + data["FTAG"]
data["Home MOV"] = data["FTHG"] - data["FTAG"]

teams = sorted(list(set(data["Home"]).union(data["Away"])))
team_index = {team: i for i, team in enumerate(teams)}
N = len(teams)

played_matches = data[:206].copy()

# Constants
avg_home_goals = played_matches["FTHG"].mean()
std_home_goals = played_matches["FTHG"].std()
avg_away_goals = played_matches["FTAG"].mean()
std_away_goals = played_matches["FTAG"].std()


def get_params_vec():
    return np.zeros(2 * N + 2)


def unpack_params(params):
    home_ratings = params[:N]
    away_ratings = params[N : 2 * N]
    home_adj = params[-2]
    away_adj = params[-1]
    return home_ratings, away_ratings, home_adj, away_adj


def compute_sse_total(params):
    home_ratings, away_ratings, home_adj, away_adj = unpack_params(params)

    # Match-level vectors
    hrh = played_matches["Home"].map(lambda team: home_ratings[team_index[team]]).values
    hra = played_matches["Away"].map(lambda team: home_ratings[team_index[team]]).values
    arh = played_matches["Home"].map(lambda team: away_ratings[team_index[team]]).values
    ara = played_matches["Away"].map(lambda team: away_ratings[team_index[team]]).values

    # Home estimation
    param_h = home_adj + hrh - hra
    p_h = np.exp(param_h) / (1 + np.exp(param_h))
    z_h = norm.ppf(p_h)
    est_home_goals = avg_home_goals + z_h * std_home_goals
    error_sq_home = (est_home_goals - played_matches["FTHG"].values) ** 2

    # Away estimation
    param_a = away_adj + arh - ara
    p_a = np.exp(param_a) / (1 + np.exp(param_a))
    z_a = norm.ppf(p_a)
    est_away_goals = avg_away_goals + z_a * std_away_goals
    error_sq_away = (est_away_goals - played_matches["FTAG"].values) ** 2

    return np.sum(error_sq_home) + np.sum(error_sq_away)


# Initial guess
initial_params = get_params_vec()

# Minimize
result = minimize(compute_sse_total, initial_params, method="L-BFGS-B")

if result.success:
    print(f"Optimized SSE_TOTAL: {result.fun:.3f}")
    optimized_home_ratings, optimized_away_ratings, opt_home_adj, opt_away_adj = (
        unpack_params(result.x)
    )
else:
    print("Optimization failed:", result.message)

# Linear Regression
# Recompute estimated goals with optimized parameters
home_ratings, away_ratings = optimized_home_ratings, optimized_away_ratings

# Re-map ratings for each match
hrh = played_matches["Home"].map(lambda team: home_ratings[team_index[team]]).values
hra = played_matches["Away"].map(lambda team: home_ratings[team_index[team]]).values
arh = played_matches["Home"].map(lambda team: away_ratings[team_index[team]]).values
ara = played_matches["Away"].map(lambda team: away_ratings[team_index[team]]).values

# Compute final estimated goals using optimized parameters
param_h = opt_home_adj + hrh - hra
p_h = np.exp(param_h) / (1 + np.exp(param_h))
z_h = norm.ppf(p_h)
est_home_goals = avg_home_goals + z_h * std_home_goals

param_a = opt_away_adj + arh - ara
p_a = np.exp(param_a) / (1 + np.exp(param_a))
z_a = norm.ppf(p_a)
est_away_goals = avg_away_goals + z_a * std_away_goals

# Compute Raw MOV (predicted margin of victory)
raw_mov = est_home_goals - est_away_goals

# Linear regression: Home MOV ~ Raw MOV
X = sm.add_constant(raw_mov)  # Add intercept term
y = played_matches["Home MOV"].values

model = sm.OLS(y, X).fit()

print("\nLinear Regression: Home MOV ~ Raw MOV")
print(model.summary())

# Predicted values from the regression model
predicted_mov = model.predict(X)

intercept = model.params[0]
slope = model.params[1]

std_errors = model.bse  # standard errors of the coefficients
intercept_se = std_errors[0]
slope_se = std_errors[1]

model_error = np.sqrt(model.mse_resid)

print(f"Intercept: {intercept:.4f} ± {intercept_se:.4f}")
print(f"Raw MOV Coeff (Slope):     {slope:.4f} ± {slope_se:.4f}")
print(f"Model error (RMSE): {model_error:.4f}")
# Plot: Actual Home MOV vs Raw MOV (with regression line)
# plt.figure(figsize=(10, 6))
# plt.scatter(raw_mov, y, alpha=0.6, label="Actual Data")
# plt.plot(raw_mov, predicted_mov, color="red", label="Regression Line")

# plt.title(f"Linear Regression: Home MOV ~ Raw MOV\n$R^2$ = {model.rsquared:.3f}")
# plt.xlabel("Raw MOV (Predicted Margin)")
# plt.ylabel("Home MOV (Actual Margin)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Predict Future Odds
unplayed_matches = data[206:].copy()


def predict_match_mov(home_team, away_team):
    # Get team ratings
    hrh = optimized_home_ratings[team_index[home_team]]
    hra = optimized_home_ratings[team_index[away_team]]
    arh = optimized_away_ratings[team_index[home_team]]
    ara = optimized_away_ratings[team_index[away_team]]

    # Compute expected home goals
    param_h = opt_home_adj + hrh - hra
    p_h = np.exp(param_h) / (1 + np.exp(param_h))
    z_h = norm.ppf(p_h)
    est_home_goals = avg_home_goals + z_h * std_home_goals

    # Compute expected away goals
    param_a = opt_away_adj + arh - ara
    p_a = np.exp(param_a) / (1 + np.exp(param_a))
    z_a = norm.ppf(p_a)
    est_away_goals = avg_away_goals + z_a * std_away_goals

    # Predicted raw MOV (normal space)
    raw_mov = est_home_goals - est_away_goals

    # Adjusted MOV using regression slope and intercept
    predicted_mov = intercept + slope * raw_mov

    return {
        "home_goals_est": est_home_goals,
        "away_goals_est": est_away_goals,
        "raw_mov": raw_mov,
        "predicted_mov": predicted_mov,
        "model_error": model_error,  # standard deviation of residuals (used for simulating outcomes)
    }


def outcome_probabilities(spread_h, spread_a, model_error):
    # Home win: MOV > 0
    p_home_win = 1 - norm.cdf(0.5, loc=spread_h, scale=model_error)
    # Away win: MOV < 0
    p_away_win = 1 - norm.cdf(0.5, loc=spread_a, scale=model_error)
    # Draw
    p_draw = 1 - p_home_win - p_away_win
    return {
        "P(Home Win)": p_home_win,
        "P(Draw)": p_draw,
        "P(Away Win)": p_away_win,
    }


result = predict_match_mov("Bournemouth", "Everton")
print(f"\nExpected Home Goals: {result['home_goals_est']:.2f}")
print(f"Expected Away Goals: {result['away_goals_est']:.2f}")
print(f"Predicted MOV: {result['predicted_mov']:.2f}")
print(f"Model Error (RMSE): {result['model_error']:.2f}")

spread_home = result["home_goals_est"] - result["away_goals_est"]
spread_away = result["away_goals_est"] - result["home_goals_est"]
print(f"Home Spread: {spread_home:.2f}")
print(f"Away Spread: {spread_away:.2f}")

probs = outcome_probabilities(spread_home, spread_away, result["model_error"])
for outcome, prob in probs.items():
    print(f"{outcome}: {prob:.2%}")

# Basic Poisson
max_goals = 15

# Set Poisson means for home and away
lambda_home = result["home_goals_est"]
lambda_away = result["away_goals_est"]

# Define goal range (0 to 6 goals)
home_goals = np.arange(0, max_goals + 1)
away_goals = np.arange(0, max_goals + 1)

# Compute Poisson PMFs
home_probs = poisson.pmf(home_goals, lambda_home)
away_probs = poisson.pmf(away_goals, lambda_away)

# Compute joint probabilities matrix
basic_prob_matrix = np.outer(home_probs, away_probs)

# Plot heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     hypo_prob_matrix,
#     annot=True,
#     fmt=".3f",
#     cmap="Reds",
#     xticklabels=away_goals,
#     yticklabels=home_goals,
# )

# plt.title("Home vs. Away Goal Probability Heatmap")
# plt.xlabel("Away Goals")
# plt.ylabel("Home Goals")
# plt.show()

home_win_prob = np.sum(
    np.tril(basic_prob_matrix, -1)
)  # Lower triangle, excluding diagonal
draw_prob = np.sum(np.diag(basic_prob_matrix))  # Diagonal
away_win_prob = np.sum(
    np.triu(basic_prob_matrix, 1)
)  # Upper triangle, excluding diagonal

print(f"Home Win Probability: {home_win_prob:.4f}")
print(f"Draw Probability:     {draw_prob:.4f}")
print(f"Away Win Probability: {away_win_prob:.4f}")

##########################################################################
### Hypothetical Expected Goal Frequencies ###
# Define goal range (0 to 6 goals)
home_goals = np.arange(0, max_goals + 1)
away_goals = np.arange(0, max_goals + 1)

# Compute Poisson PMFs
home_probs = poisson.pmf(home_goals, avg_home_goals)
away_probs = poisson.pmf(away_goals, avg_away_goals)

# Compute joint probabilities matrix
hypo_prob_matrix = np.outer(home_probs, away_probs)

# Plot heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     hypo_prob_matrix,
#     annot=True,
#     fmt=".3f",
#     cmap="Reds",
#     xticklabels=away_goals,
#     yticklabels=home_goals,
# )

# plt.title("Home vs. Away Goal Probability Heatmap")
# plt.xlabel("Away Goals")
# plt.ylabel("Home Goals")
# plt.show()

##########################################################################
# Actual Observed Goal Frequencies

# Get frequencies
home_goal_counts = data["FTHG"].value_counts().sort_index()
away_goal_counts = data["FTAG"].value_counts().sort_index()

# Get percentages
home_goal_percentages = data["FTHG"].value_counts(normalize=True).sort_index()
away_goal_percentages = data["FTAG"].value_counts(normalize=True).sort_index()

# Combine into a DataFrame
goal_stats = pd.DataFrame(
    index=np.arange(0, max_goals + 1),
    data={
        "Home Frequency": home_goal_counts,
        "Home Percentage": home_goal_percentages.round(3),
        "Away Frequency": away_goal_counts,
        "Away Percentage": away_goal_percentages.round(3),
    },
).fillna(0)

# Observed Poisson matrix using actual observed goal frequencies
observed_prob_matrix = np.outer(
    goal_stats["Home Percentage"], goal_stats["Away Percentage"]
)

# Plot heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     observed_prob_matrix,
#     annot=True,
#     fmt=".3f",
#     cmap="Reds",
#     xticklabels=away_goals,
#     yticklabels=home_goals,
# )

# plt.title("Home vs. Away Goal Probability Heatmap")
# plt.xlabel("Away Goals")
# plt.ylabel("Home Goals")
# plt.show()

# Create adjustments for zero-inflation

# Divide theoretical poisson probs by the basic poisson probs

adj_poisson_matrix = pd.DataFrame(observed_prob_matrix) / pd.DataFrame(hypo_prob_matrix)

# Plot heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     adj_poisson_matrix,
#     annot=True,
#     fmt=".3f",
#     cmap="Reds",
#     xticklabels=away_goals,
#     yticklabels=home_goals,
# )

# plt.title("Home vs. Away Goal Probability Heatmap")
# plt.xlabel("Away Goals")
# plt.ylabel("Home Goals")
# plt.show()

# ZIP Adjusted

poisson_adj_prob_matrix = basic_prob_matrix * adj_poisson_matrix

# Plot heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     poisson_adj_prob_matrix,
#     annot=True,
#     fmt=".3f",
#     cmap="Reds",
#     xticklabels=away_goals,
#     yticklabels=home_goals,
# )

# plt.title("Home vs. Away Goal Probability Heatmap")
# plt.xlabel("Away Goals")
# plt.ylabel("Home Goals")
# plt.show()

home_win_prob = np.sum(
    np.tril(poisson_adj_prob_matrix, -1)
)  # Lower triangle, excluding diagonal
draw_prob = np.sum(np.diag(poisson_adj_prob_matrix))  # Diagonal
away_win_prob = np.sum(
    np.triu(poisson_adj_prob_matrix, 1)
)  # Upper triangle, excluding diagonal

print(f"Home Win Probability: {home_win_prob:.4f}")
print(f"Draw Probability:     {draw_prob:.4f}")
print(f"Away Win Probability: {away_win_prob:.4f}")
