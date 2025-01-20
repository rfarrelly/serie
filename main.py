from app.common import ingestion, config

FBREF_LEAGUE_NAME = config.FbrefLeagueName.EPL
FBREF_DATA_DIRECTORY = "app/DATA/FBREF"
SEASON = "2024-2025"


def main():
    ingestion.get_fbref_data(
        url=config.fbref_url_builder(league=FBREF_LEAGUE_NAME, season=SEASON),
        league=FBREF_LEAGUE_NAME,
        season=SEASON,
        dir=FBREF_DATA_DIRECTORY,
    )


if __name__ == "__main__":
    main()
