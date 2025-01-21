import pandas as pd


class TeamStats:
    def __init__(self, team: str, df: pd.DataFrame):
        self.team = team
        self.df = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy()
        self._ppg = self._compute_ppg()

    def _compute_home_and_away_points(self) -> pd.DataFrame:
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

    def _compute_ppg(self):
        self._compute_home_and_away_points()
        self.df["PPG"] = (
            self.df["HomePoints"]
            .combine_first(self.df["AwayPoints"])
            .expanding()
            .mean()
        )

    @property
    def get_stats_df(self):
        return self.df.reset_index(drop=True)
