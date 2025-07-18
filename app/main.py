import numpy as np
import pandas as pd
from config import DEFAULT_CONFIG, END_DATE, GET_DATA, TODAY, Leagues
from processing import LeagueProcessor, get_historical_ppi
from utils.datetime_helpers import format_date
from utils.team_name_dict_builder import TeamNameManagerCLI


def build_team_name_dictionary():
    data_sources = ["fbref", "fbduk"]
    csv_path = "team_name_dictionary.csv"

    # Create manager
    manager = TeamNameManagerCLI(csv_path, data_sources)

    fbduk_teams = np.unique(
        pd.concat(
            [
                pd.read_csv(str(file))[["Home", "Away"]]
                for file in DEFAULT_CONFIG.fbduk_data_dir.rglob("*.csv")
                if file.is_file()
            ]
        )
        .to_numpy()
        .flatten()
    )

    fbref_teams = np.unique(
        pd.concat(
            [
                pd.read_csv(str(file))[["Home", "Away"]]
                for file in DEFAULT_CONFIG.fbref_data_dir.rglob("*.csv")
                if file.is_file()
            ]
        )
        .to_numpy()
        .flatten()
    )

    manager.import_team_list(
        fbref_teams, "fbref", auto_match=True, auto_threshold=0.7, interactive=True
    )

    manager.import_team_list(
        fbduk_teams, "fbduk", auto_match=True, auto_threshold=0.7, interactive=True
    )

    print(f"Dictionary saved to {csv_path}")


def merge_historical_odds_data():
    fbduk_odds_data = pd.concat(
        [
            pd.read_csv(str(file))
            for file in DEFAULT_CONFIG.fbduk_data_dir.rglob("*.csv")
            if file.is_file()
        ]
    )

    fbref_historical_rpi_data = pd.read_csv("historical_rpi.csv")

    print(f"fbduk input matches (odds): {fbduk_odds_data.shape[0]}")
    print(f"fbref imput matches: {fbref_historical_rpi_data.shape[0]}")

    team_name_dict = pd.read_csv("team_name_dictionary.csv")

    fbduk_to_fbref = dict(zip(team_name_dict["fbduk"], team_name_dict["fbref"]))

    def map_team_name(team_name):
        return fbduk_to_fbref.get(team_name, team_name)

    fbduk_odds_data["Home"] = fbduk_odds_data["Home"].apply(map_team_name)
    fbduk_odds_data["Away"] = fbduk_odds_data["Away"].apply(map_team_name)

    merged_df = (
        pd.merge(
            fbref_historical_rpi_data,
            fbduk_odds_data,
            on=["Date", "Home", "Away"],
            how="left",
        )
        .drop(["Time_x", "Season_y", "Wk_y"], axis="columns")
        .sort_values("Date")
    ).rename({"Time_y": "Time", "Season_x": "Season", "Wk_x": "Wk"}, axis="columns")

    print(f"Merged historical odds size: {merged_df.shape[0]}")
    print(f"{merged_df[merged_df['PSCH'].isna()].shape[0]} unmerged historical odds")

    merged_df.to_csv("historical_ppi_and_odds.csv", index=False)


def merge_future_odds_data():
    latest_ppi = pd.read_csv("latest_ppi.csv")
    fbduk_main_odds_data = pd.read_csv("fixtures.csv").rename(
        {"HomeTeam": "Home", "AwayTeam": "Away"}, axis="columns"
    )[
        [
            "Date",
            "Time",
            "Home",
            "Away",
            "PSH",
            "PSD",
            "PSA",
        ]
    ]
    fbduk_extra_odds_data = pd.read_csv("new_league_fixtures.csv").rename(
        {"HomeTeam": "Home", "AwayTeam": "Away"}, axis="columns"
    )[
        [
            "Date",
            "Time",
            "Home",
            "Away",
            "PSH",
            "PSD",
            "PSA",
        ]
    ]

    fbduk_odds_data = pd.concat([fbduk_main_odds_data, fbduk_extra_odds_data])

    fbduk_odds_data = format_date(fbduk_odds_data)

    team_name_dict = pd.read_csv("team_name_dictionary.csv")

    fbduk_to_fbref = dict(zip(team_name_dict["fbduk"], team_name_dict["fbref"]))

    def map_team_name(team_name):
        return fbduk_to_fbref.get(team_name, team_name)

    fbduk_odds_data["Home"] = fbduk_odds_data["Home"].apply(map_team_name)
    fbduk_odds_data["Away"] = fbduk_odds_data["Away"].apply(map_team_name)

    merged_df = (
        pd.merge(
            latest_ppi,
            fbduk_odds_data,
            on=["Date", "Home", "Away"],
            how="left",
        )
        .drop(["Time_x"], axis="columns")
        .sort_values("Date")
    ).rename({"Time_y": "Time"}, axis="columns")

    columns = [
        "Wk",
        "Date",
        "Time",
        "League",
        "Home",
        "Away",
        "aOppPPG",
        "hOppPPG",
        "aPPG",
        "hPPG",
        "hPPI",
        "aPPI",
        "PPI_Diff",
        "PSH",
        "PSD",
        "PSA",
    ]
    merged_df = merged_df[columns].sort_values(by="PPI_Diff")
    print(f"Merged future odds size: {merged_df.shape[0]}")

    merged_df.to_csv("latest_ppi_and_odds.csv", index=False)


def main():
    ppi_all_leagues = []
    zsd_all_leagues = []

    for league in Leagues:
        print(f"Processing {league.name} ({league.value['fbref_name']})")

        processor = LeagueProcessor(league, DEFAULT_CONFIG)

        if GET_DATA == "1":
            processor.get_fbref_data()
            processor.get_fbduk_data()

        ppi = processor.get_points_performance_index()

        if ppi:
            ppi_all_leagues.extend(ppi)

        zsd_poisson = processor.get_zsd_poisson()

        if zsd_poisson:
            zsd_all_leagues.extend(zsd_poisson)

    if ppi_all_leagues:
        print(f"Getting betting candidates for the period {TODAY} to {END_DATE}")
        ppi_latest = pd.DataFrame(ppi_all_leagues).sort_values(by="PPI_Diff")

        ppi_latest = ppi_latest[
            ppi_latest["PPI_Diff"] <= DEFAULT_CONFIG.ppi_diff_threshold
        ]

        ppi_latest.to_csv("latest_ppi.csv", index=False)

    if zsd_all_leagues:
        print(f"Getting zsd poisson data for the period {TODAY} to {END_DATE}")
        zsd_latest = pd.DataFrame(zsd_all_leagues)
        zsd_latest.to_csv("latest_zsd.csv", index=False)

    merge_future_odds_data()
    get_historical_ppi(DEFAULT_CONFIG).to_csv("historical_rpi.csv", index=False)
    merge_historical_odds_data()
    # build_team_name_dictionary()


if __name__ == "__main__":
    main()
