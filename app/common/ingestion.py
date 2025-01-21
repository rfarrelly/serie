import pandas as pd
from curl_cffi import requests
from app.common.config import FbrefLeagueName


def get_fbref_data(
    url: str, league: FbrefLeagueName, season: str, dir: str
) -> pd.DataFrame:
    """
    Fetches and processes football match data from the given FBref URL.
    """

    response = requests.get(url, impersonate="safari_ios")

    columns = ["Wk", "Day", "Date", "Time", "HomeTeam", "Score", "AwayTeam"]

    # Only get played games (i.e games with scores)
    data_df = pd.read_html(response.content)[0].dropna(
        how="any", subset="Score", axis="index"
    )[["Wk", "Day", "Date", "Time", "Home", "Score", "Away"]]

    data_df.columns = columns

    data_df["FTHG"] = data_df["Score"].apply(lambda x: int(x.split("–")[0]))
    data_df["FTAG"] = data_df["Score"].apply(lambda x: int(x.split("–")[1]))

    data_df = data_df.drop("Score", axis="columns")

    filename = f"{league.value}_{season}.csv".replace("-", "_")
    data_df.to_csv(f"{dir}/{filename}", index=False)
    print(f"File '{filename}' downloaded and saved to '{dir}'")
