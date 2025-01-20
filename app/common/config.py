from enum import Enum

FBREF_BASE_URL = "https://fbref.com/en/comps"


class FbrefLeagueId(Enum):
    EPL = 9
    ECH = 10
    ENL = 34
    EL1 = 15
    EL2 = 16
    SP1 = 12
    IT1 = 11


class FbrefLeagueName(Enum):
    EPL = "Premier-League"
    ECH = "Championship"
    ENL = "National-League"
    EL1 = "League-One"
    EL2 = "League-Two"
    SP1 = "La-Liga"
    IT1 = "Serie-A"


def fbref_url_builder(league: FbrefLeagueName, season: str = "current"):

    league_name = league.value
    league_id = FbrefLeagueId[league.name].value

    if season == "current":
        return (
            f"{FBREF_BASE_URL}/{league_id}/schedule/{league_name}-Scores-and-Fixtures"
        )

    return f"{FBREF_BASE_URL}/{league_id}/{season}/schedule/{season}-{league_name}-Scores-and-Fixtures"
