import pandas as pd
import numpy as np


class TeamStats:
    """Class to analyze and store team statistics from match history data."""

    REQUIRED_COLUMNS = ["Home", "Away", "Date", "FTHG", "FTAG", "Wk"]

    def __init__(self, team: str, match_history_df: pd.DataFrame):
        """
        Initialize TeamStats with team name and match history data.

        Args:
            team: Team name to analyze
            match_history_df: DataFrame containing match history
        """
        self._validate_inputs(team, match_history_df)

        self.team = team
        self.match_history_df = self._filter_team_matches(match_history_df)

        self._calculate_points()
        self._calculate_ppg()
        self._identify_opponents()

    def _validate_inputs(self, team: str, df: pd.DataFrame) -> None:
        """Validate input parameters."""
        if not isinstance(team, str) or not team:
            raise ValueError("Team must be a non-empty string")

        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Match history must be a non-empty DataFrame")

        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")

        if team not in df["Home"].values and team not in df["Away"].values:
            raise ValueError(f"Team '{team}' not found in match history data")

    def _filter_team_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter matches for the specified team and sort by date."""
        return (
            df[(df["Home"] == self.team) | (df["Away"] == self.team)]
            .copy()
            .reset_index(drop=True)
            .sort_values("Date")
        )

    def _calculate_points(self) -> None:
        """Calculate points earned in each match."""
        home_win = self.match_history_df["FTHG"] > self.match_history_df["FTAG"]
        draw = self.match_history_df["FTHG"] == self.match_history_df["FTAG"]
        away_win = self.match_history_df["FTHG"] < self.match_history_df["FTAG"]

        is_home_team = self.match_history_df["Home"] == self.team
        is_away_team = self.match_history_df["Away"] == self.team

        home_points = np.zeros(len(self.match_history_df))
        home_points[home_win & is_home_team] = 3
        home_points[draw & is_home_team] = 1
        home_points[~is_home_team] = None

        away_points = np.zeros(len(self.match_history_df))
        away_points[away_win & is_away_team] = 3
        away_points[draw & is_away_team] = 1
        away_points[~is_away_team] = None

        self.match_history_df["HomePoints"] = home_points
        self.match_history_df["AwayPoints"] = away_points

        self.match_history_df["TeamPoints"] = np.where(
            is_home_team,
            self.match_history_df["HomePoints"],
            self.match_history_df["AwayPoints"],
        )

    def _calculate_ppg(self) -> None:
        """Calculate the rolling points per game average."""
        self.match_history_df["PPG"] = (
            self.match_history_df["TeamPoints"].expanding().mean().round(2)
        )

        self.match_history_df["Team"] = self.team

    def _identify_opponents(self) -> None:
        """Identify opponents in each match."""
        self.match_history_df["Opponent"] = np.where(
            self.match_history_df["Home"] == self.team,
            self.match_history_df["Away"],
            self.match_history_df["Home"],
        )

        self.match_history_df["HomeOpps"] = np.where(
            self.match_history_df["Home"] != self.team,
            self.match_history_df["Home"],
            None,
        )

        self.match_history_df["AwayOpps"] = np.where(
            self.match_history_df["Away"] != self.team,
            self.match_history_df["Away"],
            None,
        )

    @property
    def stats_df(self) -> pd.DataFrame:
        """Return processed match history dataframe."""
        return self.match_history_df

    @property
    def home_points_pivot_table(self) -> pd.DataFrame:
        """Create pivot table of home points by week."""
        return (
            self.match_history_df[["Wk", "Team", "HomePoints"]]
            .pivot(index="Team", columns="Wk", values="HomePoints")
            .reset_index()
        )

    @property
    def away_points_pivot_table(self) -> pd.DataFrame:
        """Create pivot table of away points by week."""
        return (
            self.match_history_df[["Wk", "Team", "AwayPoints"]]
            .pivot(index="Team", columns="Wk", values="AwayPoints")
            .reset_index()
        )


def compute_opponents_weekly_ppg(
    stats_df: pd.DataFrame,
    opponent_column: str,
    points_pivot_table: str,
    all_teams_stats: dict[str:TeamStats],
    merge_column: str,
):
    opponents_points = pd.concat(
        [
            getattr(all_teams_stats[opp], points_pivot_table)
            for opp in stats_df[opponent_column]
            if opp is not None
        ]
    )

    week_columns = opponents_points.columns[1:]

    weeks_columns_sorted = sorted(opponents_points[week_columns].columns)

    opponents_points = opponents_points[["Team"] + weeks_columns_sorted]

    opponents_points[week_columns] = (
        opponents_points[week_columns].T.expanding().mean().T.round(2)
    )

    return stats_df[["Wk", "Date", "Home", "Away", "PPG"]].merge(
        opponents_points, left_on=merge_column, right_on="Team"
    )


def compute_rpi(
    target_team_stats: TeamStats, all_teams_stats: dict[str:TeamStats]
) -> pd.DataFrame:

    home_opps_points = compute_opponents_weekly_ppg(
        stats_df=target_team_stats.stats_df,
        opponent_column="HomeOpps",
        points_pivot_table="home_points_pivot_table",
        all_teams_stats=all_teams_stats,
        merge_column="Home",
    )

    away_opps_points = compute_opponents_weekly_ppg(
        stats_df=target_team_stats.stats_df,
        opponent_column="AwayOpps",
        points_pivot_table="away_points_pivot_table",
        all_teams_stats=all_teams_stats,
        merge_column="Away",
    )

    home_away_opps_points = (
        pd.concat([home_opps_points, away_opps_points])
        .drop(["Home", "Away"], axis="columns")
        .sort_values("Date")
    )

    week_columns = home_away_opps_points.columns[4:]

    # Compute the combined home & away average ppg of opponents played
    home_away_opps_points[week_columns] = (
        home_away_opps_points[week_columns].expanding().mean().round(2)
    )

    # Average PPG of opponents played spans the diagonal of this dataframe
    opponent_ppg = np.diag(home_away_opps_points[week_columns])
    home_away_opps_points["OppsPPG"] = opponent_ppg.round(2)

    home_away_opps_points["RPI"] = (
        home_away_opps_points["PPG"] * home_away_opps_points["OppsPPG"]
    ).round(2)

    return home_away_opps_points.reset_index(drop=True)
