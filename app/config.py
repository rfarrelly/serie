from enum import Enum
import os
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv, dotenv_values

load_dotenv()

env_values = dotenv_values()


class Leagues(Enum):
    EPL = {
        "fbref_id": 9,
        "fbref_name": "Premier-League",
    }
    ECH = {
        "fbref_id": 10,
        "fbref_name": "Championship",
    }
    EL1 = {
        "fbref_id": 15,
        "fbref_name": "League-One",
    }
    EL2 = {
        "fbref_id": 16,
        "fbref_name": "League-Two",
    }
    ENL = {
        "fbref_id": 34,
        "fbref_name": "National-League",
    }
    SP1 = {
        "fbref_id": 12,
        "fbref_name": "La-Liga",
    }
    SP2 = {
        "fbref_id": 17,
        "fbref_name": "Segunda-Division",
    }
    D1 = {
        "fbref_id": 20,
        "fbref_name": "Bundesliga",
    }
    D2 = {
        "fbref_id": 33,
        "fbref_name": "2-Bundesliga",
    }
    IT1 = {
        "fbref_id": 11,
        "fbref_name": "Serie-A",
    }
    IT2 = {
        "fbref_id": 18,
        "fbref_name": "Serie-B",
    }
    FR1 = {
        "fbref_id": 13,
        "fbref_name": "Ligue-1",
    }
    FR2 = {
        "fbref_id": 60,
        "fbref_name": "Ligue-2",
    }
    POR = {
        "fbref_id": 32,
        "fbref_name": "Primeira-Liga",
    }
    # BEL = {
    #     "fbref_id": 37,
    #     "fbref_name": "Belgian-Pro-League",
    # }
    NED = {
        "fbref_id": 23,
        "fbref_name": "Eredivisie",
    }
    # AUT = {
    #     "fbref_id": 56,
    #     "fbref_name": "Austrian-Bundesliga",
    # }
    POL = {
        "fbref_id": 36,
        "fbref_name": "Ekstraklasa",
    }

    @property
    def fbref_id(self):
        return self.value["fbref_id"]

    @property
    def fbref_name(self):
        return self.value["fbref_name"]


@dataclass
class AppConfig:
    fbref_base_url: str = os.getenv("FBREF_BASE_URL")
    current_season: str = os.getenv("CURRENT_SEASON")
    rpi_diff_threshold: float = float(os.getenv("RPI_DIFF_THRESHOLD"))
    data_dir: Path = Path(os.getenv("DATA_DIR"))

    @property
    def fbref_data_dir(self):
        return self.data_dir / "FBREF"

    def get_fbref_league_dir(self, league_name):
        path = self.fbref_data_dir / league_name
        path.mkdir(parents=True, exist_ok=True)
        return path


TODAY = datetime.now().date()
TIME_DELTA = int(env_values.get("TIME_DELTA"))
END_DATE = TODAY + timedelta(days=TIME_DELTA)

GET_DATA = env_values.get("GET_DATA")

# Default app configuration
DEFAULT_CONFIG = AppConfig()
