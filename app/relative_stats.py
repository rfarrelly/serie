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

    df["HPPG"] = (
        df.groupby("Home")["HP"]
        .expanding()
        .mean()
        .shift(shift)
        .round(2)
        .reset_index(level=0, drop=True)
    )

    df = df.sort_values(["Away", "Date"])

    df["APPG"] = (
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

    breakpoint()


def main():
    points_df = compute_points(df)

    ppg_df = compute_ppg(points_df, shift=0)


if __name__ == "__main__":
    main()
