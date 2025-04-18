from config import Leagues


def fbref_url_builder(base_url: str, league: Leagues, season: str = None):
    league_name = league.fbref_name
    league_id = league.fbref_id
    return f"{base_url}/{league_id}/{season}/schedule/{season}-{league_name}-Scores-and-Fixtures"
