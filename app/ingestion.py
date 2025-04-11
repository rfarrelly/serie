import pandas as pd
from curl_cffi import requests
from config import League


def write_files(
    df: pd.DataFrame, dir: str, league_name: str, season: str, prefix: str = None
):
    filename = f"{league_name}_{season}.csv"
    if prefix:
        filename = prefix + filename
    df.to_csv(f"{dir}/{filename}", index=False)
    print(f"File '{filename}' downloaded and saved to '{dir}'")


def fbref_url_builder(base_url: str, league: League, season: str = None):

    league_name = league.fbref_name
    league_id = league.fbref_id
    return f"{base_url}/{league_id}/{season}/schedule/{season}-{league_name}-Scores-and-Fixtures"


def fbduk_url_builder(base_url: str, league: League, season: str):

    league_id = league.fbduk_id
    season = season[2:-2].replace("-", "")

    return f"{base_url}/{season}/{league_id}.csv"


def get_fbref_data(url: str, league_name: str, season: str, dir: str):
    """
    Fetches and processes football match data from the given FBref URL.
    """
    try:
        print(f"Getting data for url: {url}")
        response = requests.get(url, impersonate="safari_ios")
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred: {req_err}")

    columns = ["Wk", "Day", "Date", "Time", "Home", "Score", "Away"]

    data_df = pd.read_html(response.content)[0]

    data_df = data_df[~data_df["Notes"].isin(["Match Suspended", "Match Cancelled"])]

    unplayed_fixtures_df = (
        data_df[data_df["Score"].isna()][columns]
        .drop("Score", axis="columns")
        .dropna(how="any", axis="index")
    )

    played_fixtures_df = data_df.dropna(how="any", subset="Score", axis="index")[
        columns
    ]

    def parse_score(score: str) -> tuple:
        if score:
            goals = score.split("–")
            return int(goals[0]), int(goals[1])
        return 0, 0

    played_fixtures_df[["FTHG", "FTAG"]] = played_fixtures_df["Score"].apply(
        lambda x: pd.Series(parse_score(x))
    )
    played_fixtures_df = played_fixtures_df.drop("Score", axis="columns")

    write_files(played_fixtures_df, dir, league_name, season)
    write_files(unplayed_fixtures_df, dir, league_name, season, prefix="unplayed_")


def get_fbduk_data(url: str, league_name: str, season: str, dir: str):

    data_df = pd.read_csv(url, encoding="latin-1")[
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

    data_df = data_df.rename(columns={"HomeTeam": "Home", "AwayTeam": "Away"})

    data_df["Date"] = pd.to_datetime(data_df["Date"], format="%d/%m/%Y").dt.strftime(
        "%Y-%m-%d"
    )

    write_files(data_df, dir, league_name, season)
