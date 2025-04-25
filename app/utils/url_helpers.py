from config import Leagues


def fbref_url_builder(base_url: str, league: Leagues, season: str = None) -> str:
    return f"{base_url}/{league.fbref_id}/{season}/schedule/{season}-{league.fbref_name}-Scores-and-Fixtures"


def fbduk_main_url_builder(base_url: str, league: Leagues, season: str) -> str:
    season = season[2:-2].replace("-", "")
    return f"{base_url}/{season}/{league.fbduk_id}.csv"


def fbduk_extra_url_builder(base_url: str, league: Leagues) -> str:
    return f"{base_url}/{league.fbduk_id}.csv"
