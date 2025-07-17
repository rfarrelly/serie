import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm, poisson


class ZSDPoissonModel:
    def __init__(self, played_matches: pd.DataFrame = None):
        self.played_matches = self._load_and_prepare_data(played_matches)
        self.teams = sorted(
            list(set(self.played_matches["Home"]).union(self.played_matches["Away"]))
        )
        self.team_index = {team: i for i, team in enumerate(self.teams)}
        self.N = len(self.teams)

        self.played_matches = self.played_matches.copy()

        self._init_constants()
        self._fit_model()
        self._fit_regression()

    def _load_and_prepare_data(self, df):
        df["Game Total"] = df["FTHG"] + df["FTAG"]
        df["Home MOV"] = df["FTHG"] - df["FTAG"]
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
            self.avg_home_goals + norm.ppf(self._sigmoid(param_h)) * self.std_home_goals
        )
        est_away_goals = (
            self.avg_away_goals + norm.ppf(self._sigmoid(param_a)) * self.std_away_goals
        )

        error_sq_home = (est_home_goals - self.played_matches["FTHG"].values) ** 2
        error_sq_away = (est_away_goals - self.played_matches["FTAG"].values) ** 2

        return np.sum(error_sq_home + error_sq_away)

    def _fit_model(self):
        initial_params = self._get_params_vec()
        result = minimize(self._compute_sse_total, initial_params, method="L-BFGS-B")

        if not result.success:
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
            self.avg_home_goals + norm.ppf(self._sigmoid(param_h)) * self.std_home_goals
        )
        est_away_goals = (
            self.avg_away_goals + norm.ppf(self._sigmoid(param_a)) * self.std_away_goals
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

        param_h = self.opt_home_adj + hrh - hra
        param_a = self.opt_away_adj + arh - ara

        est_home_goals = (
            self.avg_home_goals + norm.ppf(self._sigmoid(param_h)) * self.std_home_goals
        )
        est_away_goals = (
            self.avg_away_goals + norm.ppf(self._sigmoid(param_a)) * self.std_away_goals
        )

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
        return np.exp(x) / (1 + np.exp(x))
