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


def compute_ppg(df: DF, shift: bool = 0) -> DF:
    df = df.sort_values(["Home", "Date"])

    df["HomePPG"] = (
        df.groupby("Home")["HP"]
        .expanding()
        .mean()
        .shift(shift)
        .round(2)
        .reset_index(level=0, drop=True)
    )

    df = df.sort_values(["Away", "Date"])

    df["AwayPPG"] = (
        df.groupby("Away")["AP"]
        .expanding()
        .mean()
        .shift(shift)
        .round(2)
        .reset_index(level=0, drop=True)
    )

    home_df = df[["Date", "Home", "HP"]].rename(
        columns={"Home": "Team", "HP": "Points"}
    )

    away_df = df[["Date", "Away", "AP"]].rename(
        columns={"Away": "Team", "AP": "Points"}
    )

    long_df = pd.concat([home_df, away_df], ignore_index=True)

    long_df = long_df.sort_values(["Team", "Date"])

    long_df["TPPG"] = (
        long_df.groupby("Team")["Points"]
        .expanding()
        .mean()
        .shift(shift)
        .round(2)
        .reset_index(level=0, drop=True)
    )

    df = (
        df.merge(
            long_df, left_on=["Date", "Home", "HP"], right_on=["Date", "Team", "Points"]
        )
        .rename(columns={"TPPG": "HomeTotalPPG"})
        .drop(["Team", "Points"], axis=1)
    )

    df = (
        df.merge(
            long_df, left_on=["Date", "Away", "AP"], right_on=["Date", "Team", "Points"]
        )
        .rename(columns={"TPPG": "AwayTotalPPG"})
        .drop(["Team", "Points"], axis=1)
    ).sort_values("Date")

    return df


def opposition_ppg(team: str, df: DF) -> DF:
    def build_ppg(side: str, opp_side: str, ppg_col: str):
        opp = df.loc[df[side] == team, ["Date", opp_side]].rename(
            columns={opp_side: "Opponent"}
        )

        opp_ppg = (
            df.pivot(index=opp_side, columns="Date", values=ppg_col)
            .ffill(axis=1)
            .reset_index()
            .rename(columns={opp_side: "Team"})
        )

        merged = (
            opp_ppg.merge(opp, left_on="Team", right_on="Opponent")
            .drop("Team", axis=1)
            .sort_values("Date")
        )

        front_cols = ["Date", "Opponent"]
        date_cols = [c for c in merged.columns if c not in front_cols]
        date_cols_sorted = sorted(date_cols, key=pd.to_datetime)
        return merged[front_cols + date_cols_sorted]

    home_ppg = build_ppg("Home", "Away", "AwayPPG")
    away_ppg = build_ppg("Away", "Home", "HomePPG")

    combined = (
        pd.concat([home_ppg, away_ppg]).sort_values("Date").reset_index(drop=True)
    )

    combined.iloc[:, 2:] = (
        combined.iloc[:, 2:].ffill(axis=1).expanding().mean().round(2).fillna(0)
    )

    return combined


def main():
    points_df = compute_points(df)

    ppg_df = compute_ppg(points_df, shift=0)

    opposition_ppg("York City", ppg_df)


if __name__ == "__main__":
    main()
