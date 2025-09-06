from typing import Dict, List

import numpy as np
from scipy.optimize import minimize

from ..entities.match import Match
from ..entities.team import TeamRatings


class ModelService:
    def __init__(self, decay_rate: float = 0.001, l2_reg: float = 0.01):
        self.decay_rate = decay_rate
        self.l2_reg = l2_reg
        self.team_ratings: Dict[str, TeamRatings] = {}
        self.home_advantage = 0.2
        self.away_adjustment = -0.1

    def fit_ratings(self, matches: List[Match]) -> Dict[str, TeamRatings]:
        """Fit team ratings from historical matches"""
        # Extract teams and create index
        teams = list(
            set([m.home_team for m in matches] + [m.away_team for m in matches])
        )
        team_index = {team: i for i, team in enumerate(teams)}
        n_teams = len(teams)

        # Prepare data
        home_goals = np.array([m.home_goals for m in matches if m.is_played])
        away_goals = np.array([m.away_goals for m in matches if m.is_played])
        home_indices = np.array(
            [team_index[m.home_team] for m in matches if m.is_played]
        )
        away_indices = np.array(
            [team_index[m.away_team] for m in matches if m.is_played]
        )

        # Time weights
        weights = np.exp(-self.decay_rate * np.arange(len(matches))[::-1])
        weights = weights * len(matches) / weights.sum()

        # Optimize ratings
        initial_params = np.random.normal(0, 0.05, 2 * n_teams + 2)

        def objective(params):
            attack_ratings = params[:n_teams]
            defense_ratings = params[n_teams : 2 * n_teams]
            home_adv, away_adj = params[-2:]

            # Predict goals
            home_strength = (
                home_adv + attack_ratings[home_indices] - defense_ratings[away_indices]
            )
            away_strength = (
                away_adj + attack_ratings[away_indices] - defense_ratings[home_indices]
            )

            home_pred = 1.5 + 0.5 * home_strength
            away_pred = 1.2 + 0.5 * away_strength

            # Loss function
            home_error = (home_pred - home_goals) ** 2
            away_error = (away_pred - away_goals) ** 2
            base_loss = np.sum(weights * (home_error + away_error))

            # Regularization
            reg_loss = self.l2_reg * (
                np.sum(attack_ratings**2) + np.sum(defense_ratings**2)
            )

            return base_loss + reg_loss

        # Optimize
        bounds = [(-6, 6)] * (2 * n_teams) + [(-2, 2), (-2, 2)]
        result = minimize(objective, initial_params, bounds=bounds, method="L-BFGS-B")

        if not result.success:
            raise RuntimeError("Rating optimization failed")

        # Extract results
        attack_ratings = result.x[:n_teams]
        defense_ratings = result.x[n_teams : 2 * n_teams]
        self.home_advantage = result.x[-2]
        self.away_adjustment = result.x[-1]

        # Create team ratings
        self.team_ratings = {}
        for i, team in enumerate(teams):
            self.team_ratings[team] = TeamRatings(
                team_name=team,
                attack_rating=attack_ratings[i],
                defense_rating=defense_ratings[i],
                net_rating=attack_ratings[i] - defense_ratings[i],
            )

        return self.team_ratings

    def get_team_rating(self, team_name: str) -> TeamRatings:
        """Get ratings for a specific team"""
        if team_name not in self.team_ratings:
            # Return default ratings for unknown teams
            return TeamRatings(team_name, 0.0, 0.0, 0.0)
        return self.team_ratings[team_name]
