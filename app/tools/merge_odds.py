import pandas as pd
from pathlib import Path

# Translate team names
team_dict = pd.read_csv("team_dict.csv")


def standardise_team_names(row):
    if row["Home"] in team_dict["fbduk"].to_list():
        row["Home"] = team_dict.loc[team_dict["fbduk"] == row["Home"]]["fbref"].values[
            0
        ]
    else:
        print(f"{row["Home"]} is missing")

    if row["Away"] in team_dict["fbduk"].to_list():
        row["Away"] = team_dict.loc[team_dict["fbduk"] == row["Away"]]["fbref"].values[
            0
        ]
    else:
        print(f"{row["Away"]} is missing")
    return row


def merge_fbduk_odds():
    fbduk_data_path = Path("DATA/FBDUK")
    fbduk_files = [
        str(file) for file in fbduk_data_path.rglob("*.csv") if file.is_file()
    ]

    fbref_df = pd.read_csv("historical_candidates.csv")

    df_list = []
    for file in fbduk_files:
        df_list.append(
            pd.read_csv(file)[
                [
                    "Date",
                    "Time",
                    "HomeTeam",
                    "AwayTeam",
                    "B365CH",
                    "B365CD",
                    "B365CA",
                    "PSCH",
                    "PSCD",
                    "PSCA",
                ]
            ]
        )

    fbduk_all = pd.concat(df_list).rename(
        columns={"HomeTeam": "Home", "AwayTeam": "Away"}
    )

    fbduk_all["Date"] = pd.to_datetime(
        fbduk_all["Date"], format="%d/%m/%Y"
    ).dt.strftime("%Y-%m-%d")

    print(f"FBDUK number of mathes: {fbduk_all.shape[0]}")
    print(f"FBREF number of mathes: {fbref_df.shape[0]}")

    fbduk_all = fbduk_all.apply(standardise_team_names, axis="columns")

    merged_df = fbref_df.merge(fbduk_all, on=["Date", "Home", "Away"])

    cols = [
        "Wk",
        "Date",
        "Time",
        "League",
        "Home",
        "Away",
        "hRPI",
        "aRPI",
        "RPI_Diff",
        "FTHG",
        "FTAG",
        "FTR",
        "B365CH",
        "B365CD",
        "B365CA",
        "PSCH",
        "PSCD",
        "PSCA",
    ]

    merged_df = merged_df[cols]

    merged_df.to_csv("hf_odds.csv", index=False)


merge_fbduk_odds()
