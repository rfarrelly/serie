from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from curl_cffi import requests
from utils.datetime_helpers import filter_date_range

leagues = [
    "england",
    "england2",
    "england3",
    "england4",
    "england5",
]

DATA_DIRECTORY = Path("DATA/SOCCERSTATS")
TODAY = datetime.now().date()
TIME_DELTA = 3
END_DATE = TODAY + timedelta(days=TIME_DELTA)


def get_ppi_tables(league):
    url = f"https://www.soccerstats.com/table.asp?league={league}&tid=rp"

    res = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
        impersonate="safari_ios",
    )
    ppi_table = (
        pd.read_html(res.content)[11]
        .drop(
            ["Unnamed: 0", "Points Performance Index (Team PPG x Opponents PPG)"],
            axis=1,
        )
        .rename(
            columns={
                "Unnamed: 1": "Team",
                "Team PPG": "TeamPPG",
                "Opponents PPG": "OppsPPG",
                "Points Performance Index (Team PPG x Opponents PPG).1": "PPI",
            }
        )
    )

    ppi_table["OppsPPG"] = (
        ppi_table["OppsPPG"].str.extract(r"^\s*([0-9]+(?:\.[0-9]+)?)").astype(float)
    )

    path = DATA_DIRECTORY / league
    path.mkdir(parents=True, exist_ok=True)

    ppi_table.head(len(ppi_table) - 1).to_csv(f"{path}/ppi.csv", index=False)


def get_fixtures(league):
    url = f"https://www.soccerstats.com/results.asp?league={league}"
    res = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
        impersonate="safari_ios",
    )
    df = (
        pd.read_html(res.content)[10]
        .dropna(how="all")
        .iloc[1:]
        .rename(columns={0: "Date", 1: "Home", 2: "Time", 3: "Away"})[
            ["Date", "Home", "Time", "Away"]
        ]
    )

    df = df[~df["Time"].str.contains(r"\d+\s*-\s*\d+", regex=True, na=False)]

    current_year = datetime.now().year
    df["Date"] = pd.to_datetime(df["Date"] + f" {current_year}", format="%a %d %b %Y")

    df = filter_date_range(df, TODAY, END_DATE)
    df.to_csv("fixtures.csv", index=False)


def merge_metrics(league):
    fixtures = pd.read_csv("fixtures.csv")
    ppi_table = pd.read_csv(f"{DATA_DIRECTORY}/{league}/ppi.csv")
    combined = fixtures.merge(ppi_table, left_on="Home", right_on="Team")
    combined = combined.merge(ppi_table, left_on="Away", right_on="Team").rename(
        columns={"PPI_x": "hPPI", "PPI_y": "aPPI"}
    )
    combined = combined[["Date", "Time", "Home", "Away", "hPPI", "aPPI"]]

    combined["PPI_DIFF"] = abs(combined["hPPI"] - combined["aPPI"])
    return combined.reset_index(drop=True).sort_values("PPI_DIFF")


# get_fixtures("england2")
# get_ppi_tables("england2")
# merge_metrics("england2")
