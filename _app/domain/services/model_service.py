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
        # Filter only played matches
        played_matches = [m for m in matches if m.is_played]

        if len(played_matches) < 10:
            # Not enough data, return default ratings
            teams = list(
                set([m.home_team for m in matches] + [m.away_team for m in matches])
            )
            self.team_ratings = {}
            for team in teams:
                self.team_ratings[team] = TeamRatings(
                    team_name=team,
                    attack_rating=0.0,
                    defense_rating=0.0,
                    net_rating=0.0,
                )
            return self.team_ratings

        # Extract teams and create index
        teams = list(
            set(
                [m.home_team for m in played_matches]
                + [m.away_team for m in played_matches]
            )
        )
        team_index = {team: i for i, team in enumerate(teams)}
        n_teams = len(teams)

        # Prepare data arrays
        home_goals = np.array([m.home_goals for m in played_matches])
        away_goals = np.array([m.away_goals for m in played_matches])
        home_indices = np.array([team_index[m.home_team] for m in played_matches])
        away_indices = np.array([team_index[m.away_team] for m in played_matches])

        # Time weights (most recent matches weighted higher)
        n_matches = len(played_matches)
        weights = np.exp(-self.decay_rate * np.arange(n_matches)[::-1])
        weights = weights * n_matches / weights.sum()

        # Initialize parameters with smaller random values
        np.random.seed(42)  # For reproducibility
        initial_params = np.random.normal(0, 0.01, 2 * n_teams + 2)
        initial_params[-2] = 0.2  # home advantage
        initial_params[-1] = -0.1  # away adjustment

        def objective(params):
            try:
                attack_ratings = params[:n_teams]
                defense_ratings = params[n_teams : 2 * n_teams]
                home_adv, away_adj = params[-2:]

                # Calculate team strengths
                home_strength = (
                    home_adv
                    + attack_ratings[home_indices]
                    - defense_ratings[away_indices]
                )
                away_strength = (
                    away_adj
                    + attack_ratings[away_indices]
                    - defense_ratings[home_indices]
                )

                # Convert to expected goals with bounds
                home_pred = np.clip(1.5 + 0.3 * home_strength, 0.1, 5.0)
                away_pred = np.clip(1.2 + 0.3 * away_strength, 0.1, 5.0)

                # Weighted squared error loss
                home_error = (home_pred - home_goals) ** 2
                away_error = (away_pred - away_goals) ** 2
                base_loss = np.sum(weights * (home_error + away_error))

                # L2 regularization
                reg_loss = self.l2_reg * (
                    np.sum(attack_ratings**2)
                    + np.sum(defense_ratings**2)
                    + home_adv**2
                    + away_adj**2
                )

                total_loss = base_loss + reg_loss

                # Check for invalid values
                if not np.isfinite(total_loss):
                    return 1e10

                return total_loss

            except Exception:
                return 1e10

        try:
            # Use tighter bounds and multiple optimization attempts
            bounds = [(-3, 3)] * (2 * n_teams) + [(-1, 1), (-1, 1)]

            best_result = None
            best_loss = float("inf")

            # Try multiple random starts
            for attempt in range(3):
                try:
                    if attempt > 0:
                        # Add small random perturbation for subsequent attempts
                        start_params = initial_params + np.random.normal(
                            0, 0.001, len(initial_params)
                        )
                    else:
                        start_params = initial_params.copy()

                    result = minimize(
                        objective,
                        start_params,
                        bounds=bounds,
                        method="L-BFGS-B",
                        options={"maxiter": 1000, "ftol": 1e-6},
                    )

                    if result.success and result.fun < best_loss:
                        best_result = result
                        best_loss = result.fun

                except Exception:
                    continue

            if best_result is None or not best_result.success:
                # Fall back to simple average-based ratings
                return self._simple_ratings_fallback(played_matches, teams)

            # Extract optimized parameters
            attack_ratings = best_result.x[:n_teams]
            defense_ratings = best_result.x[n_teams : 2 * n_teams]
            self.home_advantage = best_result.x[-2]
            self.away_adjustment = best_result.x[-1]

            # Create team ratings dictionary
            self.team_ratings = {}
            for i, team in enumerate(teams):
                self.team_ratings[team] = TeamRatings(
                    team_name=team,
                    attack_rating=attack_ratings[i],
                    defense_rating=defense_ratings[i],
                    net_rating=attack_ratings[i] - defense_ratings[i],
                )

            return self.team_ratings

        except Exception as e:
            print(f"Optimization failed: {e}, using fallback ratings")
            return self._simple_ratings_fallback(played_matches, teams)

    def _simple_ratings_fallback(
        self, played_matches: List[Match], teams: List[str]
    ) -> Dict[str, TeamRatings]:
        """Fallback to simple average-based ratings when optimization fails"""
        # Calculate simple stats for each team
        team_stats = {
            team: {
                "goals_for": [],
                "goals_against": [],
                "home_games": 0,
                "away_games": 0,
            }
            for team in teams
        }

        for match in played_matches:
            team_stats[match.home_team]["goals_for"].append(match.home_goals)
            team_stats[match.home_team]["goals_against"].append(match.away_goals)
            team_stats[match.home_team]["home_games"] += 1

            team_stats[match.away_team]["goals_for"].append(match.away_goals)
            team_stats[match.away_team]["goals_against"].append(match.home_goals)
            team_stats[match.away_team]["away_games"] += 1

        # Calculate simple ratings
        self.team_ratings = {}
        league_avg_gf = np.mean(
            [
                np.mean(stats["goals_for"])
                for stats in team_stats.values()
                if stats["goals_for"]
            ]
        )
        league_avg_ga = np.mean(
            [
                np.mean(stats["goals_against"])
                for stats in team_stats.values()
                if stats["goals_against"]
            ]
        )

        for team in teams:
            stats = team_stats[team]
            if not stats["goals_for"]:  # No games played
                attack_rating = 0.0
                defense_rating = 0.0
            else:
                # Simple rating based on deviation from league average
                avg_gf = np.mean(stats["goals_for"])
                avg_ga = np.mean(stats["goals_against"])

                attack_rating = (avg_gf - league_avg_gf) / max(league_avg_gf, 0.1)
                defense_rating = (avg_ga - league_avg_ga) / max(league_avg_ga, 0.1)

            self.team_ratings[team] = TeamRatings(
                team_name=team,
                attack_rating=attack_rating,
                defense_rating=defense_rating,
                net_rating=attack_rating - defense_rating,
            )

        # Set default global parameters
        self.home_advantage = 0.2
        self.away_adjustment = -0.1

        return self.team_ratings

    def get_team_rating(self, team_name: str) -> TeamRatings:
        """Get ratings for a specific team"""
        if team_name not in self.team_ratings:
            # Return default ratings for unknown teams
            return TeamRatings(team_name, 0.0, 0.0, 0.0)
        return self.team_ratings[team_name]
