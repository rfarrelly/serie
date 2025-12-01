from typing import List, TypeAlias

import numpy as np
import pandas as pd

DF: TypeAlias = pd.DataFrame

df = pd.read_csv("DATA/FBREF/National-League/National-League_2025-2026.csv")


def compute_points(df: DF) -> DF:
    draw_or_loss = np.where(df["FTHG"] == df["FTAG"], 1, 0)
    df["HP"] = np.where(df["FTHG"] > df["FTAG"], 3, draw_or_loss)
    df["AP"] = np.where(df["FTHG"] < df["FTAG"], 3, draw_or_loss)
    return df


def compute_ppg(df: DF) -> DF:
    teams = set(df["Home"]).union(df["Away"])

    home_points = df.pivot(index=["Home"], columns="Date", values="HP")
    away_points = df.pivot(index=["Away"], columns="Date", values="AP")

    ppg_list: List[DF] = []

    for team in teams:
        h_ppg = home_points.loc[team].sort_index().dropna().reset_index()
        h_ppg["Home"] = team
        h_ppg["HomePPG"] = h_ppg[team].expanding().mean().round(2)
        h_ppg = h_ppg.rename(columns={team: "HP"})
        h_ppg = h_ppg[["Date", "Home", "HP", "HomePPG"]]

        a_ppg = away_points.loc[team].sort_index().dropna().reset_index()
        a_ppg["Away"] = team
        a_ppg["AwayPPG"] = a_ppg[team].expanding().mean().round(2)
        a_ppg = a_ppg.rename(columns={team: "AP"})
        a_ppg = a_ppg[["Date", "Away", "AP", "AwayPPG"]]

        ppg = pd.concat([h_ppg, a_ppg], axis=0).sort_values("Date")
        ppg["HA_PPG"] = ppg["HomePPG"].fillna(ppg["AwayPPG"])
        ppg["TotalPPG"] = ppg["HP"].fillna(ppg["AP"]).expanding().mean().round(2)
        ppg["Team"] = team
        ppg_list.append(ppg)

    return (
        pd.concat(ppg_list)[["Date", "Team", "HA_PPG", "TotalPPG"]]
        .pivot(index="Team", columns="Date", values="HA_PPG")
        .sort_index()
        .reset_index()
    )


def opponents_ppg(team: str, df: DF, ppg_df: DF) -> List[str]:
    team_df = df[(df["Home"] == team) | (df["Away"] == team)].copy()

    team_df["Opp"] = np.where(
        team_df["Home"] != team,
        team_df["Home"],
        np.where(team_df["Away"] != team, team_df["Away"], np.nan),
    )

    ppg_df = ppg_df.merge(team_df, left_on="Team", right_on="Opp")[
        ["Date"] + ["Team"] + team_df["Date"].to_list()
    ]

    ppg_df.iloc[:, 2:] = ppg_df.iloc[:, 2:].expanding().mean()

    diags = np.diag(ppg_df.iloc[:, 2:])
    breakpoint()


def main():
    points_df = compute_points(df)
    ppg_df = compute_ppg(points_df)
    opps = opponents_ppg("Aldershot Town", df, ppg_df)

    # compute_ppi("Aldershot Town", ppg_df, opps)


if __name__ == "__main__":
    main()
