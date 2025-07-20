import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm, poisson
from utils.datetime_helpers import format_date


class ZSDPoissonModel:
    def __init__(
        self, teams=None, played_matches: pd.DataFrame = None, decay_rate=0.001
    ):
        # When testing we do splitting which may miss teams
        if teams:
            self.teams = teams
        else:
            self.teams = sorted(
                list(set(played_matches["Home"]).union(played_matches["Away"]))
            )
        self.decay_rate = decay_rate
        self.played_matches = self._load_and_prepare_data(
            played_matches, decay_rate=self.decay_rate
        )

        self.team_index = {team: i for i, team in enumerate(self.teams)}
        self.N = len(self.teams)

        self.played_matches = self.played_matches.copy()

        self._init_constants()
        self._fit_model()
        self._fit_regression()

    def _load_and_prepare_data(self, df, decay_rate=0.0015):
        df["Game Total"] = df["FTHG"] + df["FTAG"]
        df["Home MOV"] = df["FTHG"] - df["FTAG"]

        if not np.issubdtype(df["Date"].dtype, np.datetime64):
            df["Date"] = pd.to_datetime(df["Date"])

        df = df.sort_values("Date").reset_index(drop=True)
        self.latest_date = df["Date"].max()

        # Use recency index instead of days
        match_index = np.arange(len(df))[::-1]  # Most recent = 0
        df["Weight"] = np.exp(-decay_rate * match_index)

        # Normalize to keep total weight constant
        df["Weight"] *= len(df) / df["Weight"].sum()

        return df

    def _init_constants(self):
        self.avg_home_goals = self.played_matches["FTHG"].mean()
        self.std_home_goals = self.played_matches["FTHG"].std()
        self.avg_away_goals = self.played_matches["FTAG"].mean()
        self.std_away_goals = self.played_matches["FTAG"].std()

    def _get_params_vec(self):
        return np.zeros(2 * self.N + 2)

    def _unpack_params(self, params):
        home_ratings = params[: self.N]
        away_ratings = params[self.N : 2 * self.N]
        home_adj, away_adj = params[-2:]
        return home_ratings, away_ratings, home_adj, away_adj

    def _compute_sse_total(self, params):
        home_ratings, away_ratings, home_adj, away_adj = self._unpack_params(params)

        hrh = (
            self.played_matches["Home"]
            .map(lambda t: home_ratings[self.team_index[t]])
            .values
        )
        hra = (
            self.played_matches["Away"]
            .map(lambda t: home_ratings[self.team_index[t]])
            .values
        )
        arh = (
            self.played_matches["Home"]
            .map(lambda t: away_ratings[self.team_index[t]])
            .values
        )
        ara = (
            self.played_matches["Away"]
            .map(lambda t: away_ratings[self.team_index[t]])
            .values
        )

        param_h = home_adj + hrh - hra
        param_a = away_adj + arh - ara

        est_home_goals = (
            self.avg_home_goals
            + self._safe_ppf(self._sigmoid(param_h)) * self.std_home_goals
        )
        est_away_goals = (
            self.avg_away_goals
            + self._safe_ppf(self._sigmoid(param_a)) * self.std_away_goals
        )

        error_sq_home = (est_home_goals - self.played_matches["FTHG"].values) ** 2
        error_sq_away = (est_away_goals - self.played_matches["FTAG"].values) ** 2

        # Apply time decay weights
        weights = self.played_matches["Weight"].values
        weighted_error = weights * (error_sq_home + error_sq_away)

        return np.sum(weighted_error)

    def _fit_model(self):
        initial_params = self._get_params_vec()
        result = minimize(self._compute_sse_total, initial_params, method="L-BFGS-B")

        if not result.success or np.any(np.isnan(result.x)):
            raise RuntimeError(f"Optimization failed: {result.message}")

        self.optimized_params = result.x
        (
            self.optimized_home_ratings,
            self.optimized_away_ratings,
            self.opt_home_adj,
            self.opt_away_adj,
        ) = self._unpack_params(self.optimized_params)

    def _fit_regression(self):
        raw_mov = self._get_raw_mov()
        y = self.played_matches["Home MOV"].values
        X = sm.add_constant(raw_mov)

        self.reg_model = sm.OLS(y, X).fit()
        self.intercept, self.slope = self.reg_model.params
        self.model_error = np.sqrt(self.reg_model.mse_resid)

    def _get_raw_mov(self):
        hrh = (
            self.played_matches["Home"]
            .map(lambda t: self.optimized_home_ratings[self.team_index[t]])
            .values
        )
        hra = (
            self.played_matches["Away"]
            .map(lambda t: self.optimized_home_ratings[self.team_index[t]])
            .values
        )
        arh = (
            self.played_matches["Home"]
            .map(lambda t: self.optimized_away_ratings[self.team_index[t]])
            .values
        )
        ara = (
            self.played_matches["Away"]
            .map(lambda t: self.optimized_away_ratings[self.team_index[t]])
            .values
        )

        param_h = self.opt_home_adj + hrh - hra
        param_a = self.opt_away_adj + arh - ara

        est_home_goals = (
            self.avg_home_goals
            + self._safe_ppf(self._sigmoid(param_h)) * self.std_home_goals
        )
        est_away_goals = (
            self.avg_away_goals
            + self._safe_ppf(self._sigmoid(param_a)) * self.std_away_goals
        )

        return est_home_goals - est_away_goals

    def predict_match_mov(self, home_team, away_team):
        idx_h = self.team_index[home_team]
        idx_a = self.team_index[away_team]

        hrh, hra = (
            self.optimized_home_ratings[idx_h],
            self.optimized_home_ratings[idx_a],
        )
        arh, ara = (
            self.optimized_away_ratings[idx_h],
            self.optimized_away_ratings[idx_a],
        )

        # Clamp the raw inputs before sigmoid → norm.ppf to prevent extreme values
        param_h = np.clip(self.opt_home_adj + hrh - hra, -6, 6)
        param_a = np.clip(self.opt_away_adj + arh - ara, -6, 6)

        est_home_goals = (
            self.avg_home_goals
            + self._safe_ppf(self._sigmoid(param_h)) * self.std_home_goals
        )
        est_away_goals = (
            self.avg_away_goals
            + self._safe_ppf(self._sigmoid(param_a)) * self.std_away_goals
        )

        # Ensure Poisson lambda inputs are valid (positive)
        est_home_goals = max(est_home_goals, 1e-3)
        est_away_goals = max(est_away_goals, 1e-3)

        raw_mov = est_home_goals - est_away_goals
        predicted_mov = self.intercept + self.slope * raw_mov

        return {
            "home_goals_est": est_home_goals,
            "away_goals_est": est_away_goals,
            "raw_mov": raw_mov,
            "predicted_mov": predicted_mov,
            "model_error": self.model_error,
        }

    def outcome_probabilities(self, spread_home, spread_away):
        p_home_win = 1 - norm.cdf(0.5, loc=spread_home, scale=self.model_error)
        p_away_win = 1 - norm.cdf(0.5, loc=spread_away, scale=self.model_error)
        p_draw = 1 - p_home_win - p_away_win

        return {
            "P_MOV(Home Win)": p_home_win,
            "P_MOV(Draw)": p_draw,
            "P_MOV(Away Win)": p_away_win,
        }

    def poisson_prob_matrix(self, lambda_home, lambda_away, max_goals=15):
        home_goals = np.arange(0, max_goals + 1)
        away_goals = np.arange(0, max_goals + 1)

        home_probs = poisson.pmf(home_goals, lambda_home)
        away_probs = poisson.pmf(away_goals, lambda_away)

        return np.outer(home_probs, away_probs)

    def zip_adjustment_matrix(self, max_goals=15):
        home_percent = (
            self.played_matches["FTHG"].value_counts(normalize=True).sort_index()
        )
        away_percent = (
            self.played_matches["FTAG"].value_counts(normalize=True).sort_index()
        )
        idx = np.arange(0, max_goals + 1)

        observed = np.outer(
            home_percent.reindex(idx, fill_value=0),
            away_percent.reindex(idx, fill_value=0),
        )

        expected = self.poisson_prob_matrix(
            self.avg_home_goals, self.avg_away_goals, max_goals
        )
        return pd.DataFrame(observed) / pd.DataFrame(expected)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _safe_ppf(x, eps=1e-6):
        return norm.ppf(np.clip(x, eps, 1 - eps))


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
    poisson_matrix = model.poisson_prob_matrix(lambda_home, lambda_away, max_goals=10)
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

preds_df = pd.DataFrame(results)

### Single Match ###
# matches = pd.read_csv("zsd_poisson_test_data.csv", dtype={"Wk": int})
# matches = format_date(matches)
# played_matches = matches[:206].copy()
# unplayed_matches = matches[206:]
# model = ZSDPoissonModel(played_matches=played_matches, decay_rate=0.001)

# # Core predictions from the model
# result = model.predict_match_mov("Bournemouth", "Everton")

# # Raw goal estimates
# lambda_home = result["home_goals_est"]
# lambda_away = result["away_goals_est"]

# # Get outcome probabilities from logistic-MOV model
# probs = model.outcome_probabilities(
#     lambda_home - lambda_away, lambda_away - lambda_home
# )
# result |= probs

# # Generate Poisson and ZIP-adjusted matrices
# poisson_matrix = model.poisson_prob_matrix(lambda_home, lambda_away, max_goals=10)
# zip_adj_matrix = model.zip_adjustment_matrix(max_goals=10)
# zip_poisson_matrix = poisson_matrix * zip_adj_matrix.values

# # Collapse to outcome probabilities
# result["P_Poisson(Home Win)"] = np.tril(poisson_matrix, -1).sum()
# result["P_Poisson(Draw)"] = np.trace(poisson_matrix)
# result["P_Poisson(Away Win)"] = np.triu(poisson_matrix, 1).sum()

# result["P_ZIP(Home Win)"] = np.tril(zip_poisson_matrix, -1).sum()
# result["P_ZIP(Draw)"] = np.trace(zip_poisson_matrix)
# result["P_ZIP(Away Win)"] = np.triu(zip_poisson_matrix, 1).sum()

# preds_df = pd.DataFrame(data=result, index=[0])
# print(preds_df)
