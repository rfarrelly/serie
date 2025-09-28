from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TeamMetrics:
    team_name: str
    matches: str | pd.DataFrame
    team_matches: pd.DataFrame = None

    def __post_init__(self):
        if isinstance(self.matches, str):
            self.matches = pd.read_csv(self.matches)

        self.team_matches = self.matches[
            (self.matches["Home"] == self.team_name)
            | (self.matches["Away"] == self.team_name)
        ]

    # Away teams played by the team at home
    @property
    def home_played_opposition_teams(self):
        opposition = self.team_matches[["Wk", "Date", "Away"]]
        opposition = opposition[opposition["Away"] != self.team_name]
        return opposition

    # Home teams played by the team away
    @property
    def away_played_opposition_teams(self):
        opposition = self.team_matches[["Wk", "Date", "Home"]]
        opposition = opposition[opposition["Home"] != self.team_name]
        return opposition

    @property
    def home_points(self) -> pd.DataFrame:
        self.matches["HP"] = self.matches.copy().apply(
            lambda x: (
                3 if x["FTHG"] > x["FTAG"] else 1 if x["FTHG"] == x["FTAG"] else 0
            ),
            axis="columns",
        )

        return self.matches.pivot(index="Home", columns="Wk", values="HP")

    @property
    def away_points(self) -> pd.DataFrame:
        self.matches["AP"] = self.matches.copy().apply(
            lambda x: (
                3 if x["FTHG"] < x["FTAG"] else 1 if x["FTHG"] == x["FTAG"] else 0
            ),
            axis="columns",
        )

        return self.matches.pivot(index="Away", columns="Wk", values="AP")

    def team_total_weekly_ppg(self):
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

    def opposition_away_weekly_ppg(self):
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

    def opposition_home_weekly_ppg(self):
        return (
            self.home_points[
                self.home_points.index.isin(self.away_played_opposition_teams["Home"])
            ]
            .T.expanding()
            .mean()
            .T.round(3)
            .reset_index(names="Team")
        )

    def opposition_home_away_weekly_mean_ppg(self):
        home_weekly_mean_ppg = self.away_played_opposition_teams.merge(
            self.opposition_home_weekly_ppg(), left_on="Home", right_on="Team"
        )

        away_weekly_mean_ppg = self.home_played_opposition_teams.merge(
            self.opposition_away_weekly_ppg(), left_on="Away", right_on="Team"
        )

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

    def points_performance_index(self):
        mean_weekly_opposition_ppg = self.opposition_home_away_weekly_mean_ppg()
        week_cols = [
            c for c in mean_weekly_opposition_ppg.columns if isinstance(c, (int, float))
        ]

        mean_opposistion_ppg = np.diag(mean_weekly_opposition_ppg[week_cols])

        mean_weekly_opposition_ppg["OppsPPG"] = mean_opposistion_ppg
        weekly_team_ppg = self.team_total_weekly_ppg()[
            self.team_total_weekly_ppg().columns[1:]
        ].values[0]
        mean_weekly_opposition_ppg["TeamPPG"] = weekly_team_ppg
        mean_weekly_opposition_ppg["TeamPPI"] = (
            mean_weekly_opposition_ppg["OppsPPG"]
            * mean_weekly_opposition_ppg["TeamPPG"]
        )
        return mean_weekly_opposition_ppg
