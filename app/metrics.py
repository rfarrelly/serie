from dataclasses import dataclass
from functools import cached_property
from typing import Union

import numpy as np
import pandas as pd


@dataclass
class TeamMetrics:
    """
    A class for calculating team performance metrics from match data.

    Args:
        team_name: Name of the team to analyze
        matches: Either a CSV file path or a pandas DataFrame containing match data
        team_matches: DataFrame of matches for this specific team (auto-populated)
    """

    team_name: str
    matches: Union[str, pd.DataFrame]
    team_matches: pd.DataFrame = None

    def __post_init__(self):
        """Initialize the TeamMetrics object and validate input data."""
        if isinstance(self.matches, str):
            self.matches = pd.read_csv(self.matches)

        # Input validation
        required_cols = ["Home", "Away", "FTHG", "FTAG", "Wk", "Date"]
        missing_cols = set(required_cols) - set(self.matches.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        self.team_matches = self.matches[
            (self.matches["Home"] == self.team_name)
            | (self.matches["Away"] == self.team_name)
        ].copy()

    def _calculate_points(
        self, home_goals: pd.Series, away_goals: pd.Series, perspective: str
    ) -> pd.Series:
        """
        Calculate points from home/away perspective using vectorized operations.

        Args:
            home_goals: Series of home team goals
            away_goals: Series of away team goals
            perspective: Either "home" or "away" to determine point calculation

        Returns:
            Series of points (3 for win, 1 for draw, 0 for loss)
        """
        if perspective == "home":
            return np.where(
                home_goals > away_goals, 3, np.where(home_goals == away_goals, 1, 0)
            )
        else:  # away
            return np.where(
                away_goals > home_goals, 3, np.where(home_goals == away_goals, 1, 0)
            )

    @property
    def home_played_opposition_teams(self) -> pd.DataFrame:
        """Get away teams that played against this team at home."""
        opposition = self.team_matches[["Wk", "Date", "Away"]]
        return opposition[opposition["Away"] != self.team_name]

    @property
    def away_played_opposition_teams(self) -> pd.DataFrame:
        """Get home teams that this team played against away."""
        opposition = self.team_matches[["Wk", "Date", "Home"]]
        return opposition[opposition["Home"] != self.team_name]

    @cached_property
    def home_points(self) -> pd.DataFrame:
        """Calculate home points for all teams across all weeks."""
        matches_copy = self.matches.copy()
        matches_copy["HP"] = self._calculate_points(
            matches_copy["FTHG"], matches_copy["FTAG"], "home"
        )
        return matches_copy.pivot(index="Home", columns="Wk", values="HP")

    @cached_property
    def away_points(self) -> pd.DataFrame:
        """Calculate away points for all teams across all weeks."""
        matches_copy = self.matches.copy()
        matches_copy["AP"] = self._calculate_points(
            matches_copy["FTHG"], matches_copy["FTAG"], "away"
        )
        return matches_copy.pivot(index="Away", columns="Wk", values="AP")

    def team_total_weekly_ppg(self) -> pd.DataFrame:
        """Calculate team's cumulative points per game by week."""
        home_points = self.home_points[self.home_points.index == self.team_name]
        away_points = self.away_points[self.away_points.index == self.team_name]

        combined = pd.concat([home_points, away_points], axis="columns").dropna(
            how="all", axis="columns"
        )

        result = (
            combined[combined.columns.sort_values()]
            .T.expanding()
            .mean()
            .T.round(3)
            .reset_index(names="Team")
        )

        result.columns.name = None
        return result

    def opposition_away_weekly_ppg(self) -> pd.DataFrame:
        """Calculate opposition away PPG for teams that played at this team's home."""
        result = (
            self.away_points[
                self.away_points.index.isin(self.home_played_opposition_teams["Away"])
            ]
            .T.expanding()
            .mean()
            .T.round(3)
            .reset_index(names="Team")
        )

        result.columns.name = None
        return result

    def opposition_home_weekly_ppg(self) -> pd.DataFrame:
        """Calculate opposition home PPG for teams this team played away against."""
        result = (
            self.home_points[
                self.home_points.index.isin(self.away_played_opposition_teams["Home"])
            ]
            .T.expanding()
            .mean()
            .T.round(3)
            .reset_index(names="Team")
        )

        result.columns.name = None
        return result

    def _get_opposition_home_ppg(self) -> pd.DataFrame:
        """Get opposition home PPG for matches where team played away."""
        return self.away_played_opposition_teams.merge(
            self.opposition_home_weekly_ppg(), left_on="Home", right_on="Team"
        )

    def _get_opposition_away_ppg(self) -> pd.DataFrame:
        """Get opposition away PPG for matches where team played home."""
        return self.home_played_opposition_teams.merge(
            self.opposition_away_weekly_ppg(), left_on="Away", right_on="Team"
        )

    def opposition_home_away_weekly_mean_ppg(self) -> pd.DataFrame:
        """Combine home and away opposition PPG data with cumulative means."""
        home_weekly_mean_ppg = self._get_opposition_home_ppg()
        away_weekly_mean_ppg = self._get_opposition_away_ppg()

        combined_mean_weekly_ppg = (
            pd.concat([home_weekly_mean_ppg, away_weekly_mean_ppg])
            .sort_values("Date")
            .drop(["Home", "Away"], axis="columns")
        ).rename(columns={"Team": "Opposition"})

        valid_weeks = set(combined_mean_weekly_ppg["Wk"].unique())

        keep_cols = ["Wk", "Date", "Opposition"] + [
            c for c in combined_mean_weekly_ppg.columns if c in valid_weeks
        ]

        combined_mean_weekly_ppg = combined_mean_weekly_ppg[keep_cols]

        week_cols = [
            c for c in combined_mean_weekly_ppg.columns if isinstance(c, (int, float))
        ]
        combined_mean_weekly_ppg[week_cols] = (
            combined_mean_weekly_ppg[week_cols].expanding().mean().round(3)
        )

        return combined_mean_weekly_ppg

    def points_performance_index(self) -> pd.DataFrame:
        """Calculate points performance index comparing team PPG to opposition PPG."""
        mean_weekly_opposition_ppg = self.opposition_home_away_weekly_mean_ppg()
        week_cols = [
            c for c in mean_weekly_opposition_ppg.columns if isinstance(c, (int, float))
        ]

        # More robust way to get opposition PPG for the week each match was played
        opposition_ppg_values = []
        for idx, row in mean_weekly_opposition_ppg.iterrows():
            week = row["Wk"]
            if week in week_cols:
                opposition_ppg_values.append(row[week])
            else:
                opposition_ppg_values.append(np.nan)

        mean_weekly_opposition_ppg["OppsPPG"] = opposition_ppg_values

        weekly_team_ppg = self.team_total_weekly_ppg()[
            self.team_total_weekly_ppg().columns[1:]
        ].values[0]

        mean_weekly_opposition_ppg["TeamPPG"] = weekly_team_ppg
        mean_weekly_opposition_ppg["TeamPPI"] = (
            mean_weekly_opposition_ppg["OppsPPG"]
            * mean_weekly_opposition_ppg["TeamPPG"]
        )
        return mean_weekly_opposition_ppg
