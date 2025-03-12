import pandas as pd
from curl_cffi import requests
from config import League


def fbref_url_builder(base_url: str, league: League, season: str = "current"):

    league_name = league.fbref_name
    league_id = league.id

    if season == "current":
        return f"{base_url}/{league_id}/schedule/{league_name}-Scores-and-Fixtures"

    return f"{base_url}/{league_id}/{season}/schedule/{season}-{league_name}-Scores-and-Fixtures"


def get_fbref_data(url: str, league_name: str, season: str, dir: str) -> pd.DataFrame:
    """
    Fetches and processes football match data from the given FBref URL.
    """

    response = requests.get(url, impersonate="safari_ios")

    columns = ["Wk", "Day", "Date", "Time", "Home", "Score", "Away"]

    data_df = pd.read_html(response.content)[0]

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

    played_fixtures_filename = f"{league_name}_{season}.csv"
    played_fixtures_df.to_csv(f"{dir}/{played_fixtures_filename}", index=False)
    print(f"File '{played_fixtures_filename}' downloaded and saved to '{dir}'")

    unplayed_fixtures_filename = f"unplayed_{league_name}_{season}.csv"
    unplayed_fixtures_df.to_csv(f"{dir}/{unplayed_fixtures_filename}", index=False)
    print(f"File '{unplayed_fixtures_filename}' downloaded and saved to '{dir}'")
