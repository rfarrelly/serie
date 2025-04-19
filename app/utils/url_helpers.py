from config import Leagues


def fbref_url_builder(base_url: str, league: Leagues, season: str = None) -> str:
    return f"{base_url}/{league.fbref_id}/{season}/schedule/{season}-{league.fbref_name}-Scores-and-Fixtures"
