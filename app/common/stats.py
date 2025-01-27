import pandas as pd
import numpy as np


class TeamStats:
    def __init__(self, team: str, df: pd.DataFrame):
        self.team = team
        self.df = (
            df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]
            .copy()
            .reset_index(drop=True)
            .sort_values("Date")
        )
        self._setup()

    def _setup(self) -> pd.DataFrame:
        self.df["HomePoints"] = self.df.apply(
            lambda x: (
                (3 if x["FTHG"] > x["FTAG"] else 1 if x["FTHG"] == x["FTAG"] else 0)
                if (x["HomeTeam"] == self.team)
                else None
            ),
            axis="columns",
        )

        self.df["AwayPoints"] = self.df.apply(
            lambda x: (
                (3 if x["FTHG"] < x["FTAG"] else 1 if x["FTHG"] == x["FTAG"] else 0)
                if (x["AwayTeam"] == self.team)
                else None
            ),
            axis="columns",
        )

        self.df["PPG"] = (
            pd.Series(
                np.where(
                    self.df["HomeTeam"] == self.team,
                    self.df["HomePoints"],
                    self.df["AwayPoints"],
                )
            )
            .expanding()
            .mean()
            .round(2)
        )

        self.df["Team"] = pd.Series(
            np.where(
                self.df["HomeTeam"] == self.team,
                self.df["HomeTeam"],
                self.df["AwayTeam"],
            )
        )

        self.df["HomeOpps"] = self.df.apply(
            lambda x: x["HomeTeam"] if x["HomeTeam"] != self.team else None,
            axis="columns",
        )
        self.df["AwayOpps"] = self.df.apply(
            lambda x: x["AwayTeam"] if x["AwayTeam"] != self.team else None,
            axis="columns",
        )

    @property
    def stats_df(self):
        return self.df

    @property
    def home_points_pivot_table(self) -> pd.DataFrame:
        return (
            self.df[["Wk", "Team", "HomePoints"]]
            .pivot(index="Team", columns="Wk", values="HomePoints")
            .reset_index()
        )

    @property
    def away_points_pivot_table(self) -> pd.DataFrame:
        return (
            self.df[["Wk", "Team", "AwayPoints"]]
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

    weeks_columns_sorted = sorted(
        opponents_points[opponents_points.columns[1:]].columns
    )

    opponents_points = opponents_points[["Team"] + weeks_columns_sorted]

    opponents_points[opponents_points.columns[1:]] = (
        opponents_points[opponents_points.columns[1:]].T.expanding().mean().T.round(2)
    )

    merged_points = stats_df[["Wk", "Date", "HomeTeam", "AwayTeam", "PPG"]].merge(
        opponents_points, left_on=merge_column, right_on="Team"
    )

    return merged_points


def compute_rpi(target_team_stats: TeamStats, all_teams_stats: dict[str:TeamStats]):

    home_opps_points = compute_opponents_weekly_ppg(
        stats_df=target_team_stats.stats_df,
        opponent_column="HomeOpps",
        points_pivot_table="home_points_pivot_table",
        all_teams_stats=all_teams_stats,
        merge_column="HomeTeam",
    )

    away_opps_points = compute_opponents_weekly_ppg(
        stats_df=target_team_stats.stats_df,
        opponent_column="AwayOpps",
        points_pivot_table="away_points_pivot_table",
        all_teams_stats=all_teams_stats,
        merge_column="AwayTeam",
    )

    home_away_opps_points = (
        pd.concat([home_opps_points, away_opps_points])
        .drop(["HomeTeam", "AwayTeam"], axis="columns")
        .sort_values("Date")
    )

    # Compute the combined home & away average ppg of opponents played
    home_away_opps_points[home_away_opps_points.columns[4:]] = (
        home_away_opps_points[home_away_opps_points.columns[4:]]
        .expanding()
        .mean()
        .round(2)
    )

    # Average PPG of opponents played spans the diagonal of this dataframe
    opponent_ppg = np.diag(home_away_opps_points[home_away_opps_points.columns[4:]])
    home_away_opps_points["OppsPPG"] = opponent_ppg.round(2)

    # Calculate the RPI of the target team
    home_away_opps_points["RPI"] = (
        home_away_opps_points["PPG"] * home_away_opps_points["OppsPPG"]
    ).round(2)

    return home_away_opps_points.reset_index(drop=True)
