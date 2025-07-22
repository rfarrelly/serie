import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm, poisson


class ZSDPoissonModel:
    def __init__(
        self, teams=None, played_matches: pd.DataFrame = None, decay_rate=0.001
    ):
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

        self._init_constants()
        self._fit_model()
        self._fit_regression()
        self.zip_matrix = self.zip_adjustment_matrix()

    def _load_and_prepare_data(self, df, decay_rate=0.0015):
        df = df.copy()
        df["Game Total"] = df["FTHG"] + df["FTAG"]
        df["Home MOV"] = df["FTHG"] - df["FTAG"]

        if not np.issubdtype(df["Date"].dtype, np.datetime64):
            df["Date"] = pd.to_datetime(df["Date"])

        df = df.sort_values("Date").reset_index(drop=True)
        self.latest_date = df["Date"].max()

        match_index = np.arange(len(df))[::-1]
        df["Weight"] = np.exp(-decay_rate * match_index)
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
        attack_ratings = params[: self.N]
        defense_ratings = params[self.N : 2 * self.N]
        home_adj, away_adj = params[-2:]
        return attack_ratings, defense_ratings, home_adj, away_adj

    def _compute_sse_total(self, params):
        attack_ratings, defense_ratings, home_adj, away_adj = self._unpack_params(
            params
        )

        atk_h = (
            self.played_matches["Home"]
            .map(lambda t: attack_ratings[self.team_index[t]])
            .values
        )
        def_a = (
            self.played_matches["Away"]
            .map(lambda t: defense_ratings[self.team_index[t]])
            .values
        )
        atk_a = (
            self.played_matches["Away"]
            .map(lambda t: attack_ratings[self.team_index[t]])
            .values
        )
        def_h = (
            self.played_matches["Home"]
            .map(lambda t: defense_ratings[self.team_index[t]])
            .values
        )

        param_h = np.clip(home_adj + atk_h - def_a, -6, 6)
        param_a = np.clip(away_adj + atk_a - def_h, -6, 6)

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
            self.attack_ratings,
            self.defense_ratings,
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
        atk_h = (
            self.played_matches["Home"]
            .map(lambda t: self.attack_ratings[self.team_index[t]])
            .values
        )
        def_a = (
            self.played_matches["Away"]
            .map(lambda t: self.defense_ratings[self.team_index[t]])
            .values
        )
        atk_a = (
            self.played_matches["Away"]
            .map(lambda t: self.attack_ratings[self.team_index[t]])
            .values
        )
        def_h = (
            self.played_matches["Home"]
            .map(lambda t: self.defense_ratings[self.team_index[t]])
            .values
        )

        param_h = np.clip(self.opt_home_adj + atk_h - def_a, -6, 6)
        param_a = np.clip(self.opt_away_adj + atk_a - def_h, -6, 6)

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

        atk_h = self.attack_ratings[idx_h]
        atk_a = self.attack_ratings[idx_a]
        def_h = self.defense_ratings[idx_h]
        def_a = self.defense_ratings[idx_a]

        param_h = np.clip(self.opt_home_adj + atk_h - def_a, -6, 6)
        param_a = np.clip(self.opt_away_adj + atk_a - def_h, -6, 6)

        est_home_goals = (
            self.avg_home_goals
            + self._safe_ppf(self._sigmoid(param_h)) * self.std_home_goals
        )
        est_away_goals = (
            self.avg_away_goals
            + self._safe_ppf(self._sigmoid(param_a)) * self.std_away_goals
        )

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

    def predict_zip_adjusted_outcomes(self, home_team, away_team, max_goals=15):
        idx_h = self.team_index[home_team]
        idx_a = self.team_index[away_team]

        atk_h = self.attack_ratings[idx_h]
        atk_a = self.attack_ratings[idx_a]
        def_h = self.defense_ratings[idx_h]
        def_a = self.defense_ratings[idx_a]

        param_h = self.opt_home_adj + atk_h - def_a
        param_a = self.opt_away_adj + atk_a - def_h

        lambda_home = (
            self.avg_home_goals + norm.ppf(self._sigmoid(param_h)) * self.std_home_goals
        )
        lambda_away = (
            self.avg_away_goals + norm.ppf(self._sigmoid(param_a)) * self.std_away_goals
        )

        lambda_home = np.clip(lambda_home, 0.01, 10)
        lambda_away = np.clip(lambda_away, 0.01, 10)

        base_poisson = self.poisson_prob_matrix(lambda_home, lambda_away, max_goals)
        zip_matrix = self.zip_matrix.values[: max_goals + 1, : max_goals + 1]
        zip_adjusted = base_poisson * zip_matrix
        zip_adjusted /= zip_adjusted.sum()

        home_win = np.tril(zip_adjusted, -1).sum()
        draw = np.trace(zip_adjusted)
        away_win = np.triu(zip_adjusted, 1).sum()

        return {
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
            "poisson_matrix": zip_adjusted,
            "P(Home Win)": home_win,
            "P(Draw)": draw,
            "P(Away Win)": away_win,
        }

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _safe_ppf(x, eps=1e-6):
        return norm.ppf(np.clip(x, eps, 1 - eps))
