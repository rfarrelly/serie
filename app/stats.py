import numpy as np
import pandas as pd


def compute_ppg(df: pd.DataFrame) -> tuple[pd.DataFrame]:
    df["HP"] = df.apply(
        lambda x: 3 if x["FTHG"] > x["FTAG"] else 1 if x["FTHG"] == x["FTAG"] else 0,
        axis="columns",
    )

    df["AP"] = df.apply(
        lambda x: 3 if x["FTHG"] < x["FTAG"] else 1 if x["FTHG"] == x["FTAG"] else 0,
        axis="columns",
    )

    df = week_continuity(df)

    home_points = df.pivot(index="Home", columns="Wk", values="HP")
    away_points = df.pivot(index="Away", columns="Wk", values="AP")

    home_ppg = home_points.T.expanding().mean().T.round(3).reset_index()
    away_ppg = away_points.T.expanding().mean().T.round(3).reset_index()
    total_ppg = (
        home_points.combine_first(away_points)
        .T.expanding()
        .mean()
        .T.round(3)
        .reset_index()
        .rename({"Home": "Team"}, axis="columns")
    )

    return home_ppg, away_ppg, total_ppg


def compute_points_performance_index(
    team: str,
    df: pd.DataFrame,
    hppg: pd.DataFrame,
    appg: pd.DataFrame,
    tppg: pd.DataFrame,
) -> pd.DataFrame:
    home_games = df[(df["Home"] == team)]
    away_games = df[(df["Away"] == team)]
    home_games = home_games.merge(appg, left_on="Away", right_on="Away")
    away_games = away_games.merge(hppg, left_on="Home", right_on="Home")

    combined = (
        pd.concat([home_games, away_games]).reset_index(drop=True).sort_values("Date")
    )

    weeks_columns = [x for x in combined.columns if isinstance(x, int)]

    combined[weeks_columns] = combined[weeks_columns].expanding().mean().round(3)

    combined["OppPPG"] = np.diag(combined[weeks_columns])

    combined["PPG"] = tppg[tppg["Team"] == team][weeks_columns].values[0][
        : combined.shape[0]
    ]

    # NOTE:
    # OppPPG = average PPG of ALL opposition teams played at home OR away
    # TeamPPG = combined home AND away PPG for the target team
    combined["PPI"] = round(combined["OppPPG"] * combined["PPG"], 3)
    combined["TeamType"] = combined.apply(
        lambda x: "h" if team == x["Home"] else "a", axis=1
    )
    return combined.drop(weeks_columns, axis=1)


def week_continuity(df: pd.DataFrame):
    week = df["Wk"].astype(int)

    # Create a cumulative offset that increases by the last known max week
    # each time a reset is detected
    offsets = np.zeros(len(df), dtype=int)
    current_offset = 0
    last_max = 0

    for i in range(len(df)):
        if i > 0 and week[i] < week[i - 1]:
            current_offset += last_max
        offsets[i] = current_offset
        last_max = max(last_max, week[i])

    # Apply offset to fix week numbers
    df["Wk_adj"] = week + offsets

    df = df.drop("Wk", axis=1).rename({"Wk_adj": "Wk"}, axis=1)
    cols = ["Wk"] + [col for col in df.columns if col != "Wk"]
    df = df[cols]
    # if "Ekstraklasa" in df["League"].values:
    #     breakpoint()
    return df
