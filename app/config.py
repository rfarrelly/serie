from enum import Enum
import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


class Leagues(Enum):
    EPL = {
        "fbref_id": 9,
        "fbref_name": "Premier-League",
        "fbduk_id": "E0",
    }
    ECH = {
        "fbref_id": 10,
        "fbref_name": "Championship",
        "fbduk_id": "E1",
    }
    EL1 = {
        "fbref_id": 15,
        "fbref_name": "League-One",
        "fbduk_id": "E2",
    }
    EL2 = {
        "fbref_id": 16,
        "fbref_name": "League-Two",
        "fbduk_id": "E3",
    }
    ENL = {
        "fbref_id": 34,
        "fbref_name": "National-League",
        "fbduk_id": "EC",
    }
    SP1 = {
        "fbref_id": 12,
        "fbref_name": "La-Liga",
        "fbduk_id": "SP1",
    }
    SP2 = {
        "fbref_id": 17,
        "fbref_name": "Segunda-Division",
        "fbduk_id": "SP2",
    }
    D1 = {
        "fbref_id": 20,
        "fbref_name": "Bundesliga",
        "fbduk_id": "D1",
    }
    D2 = {
        "fbref_id": 33,
        "fbref_name": "2-Bundesliga",
        "fbduk_id": "D2",
    }
    IT1 = {
        "fbref_id": 11,
        "fbref_name": "Serie-A",
        "fbduk_id": "I1",
    }
    IT2 = {
        "fbref_id": 18,
        "fbref_name": "Serie-B",
        "fbduk_id": "I2",
    }
    FR1 = {
        "fbref_id": 13,
        "fbref_name": "Ligue-1",
        "fbduk_id": "F1",
    }
    FR2 = {
        "fbref_id": 60,
        "fbref_name": "Ligue-2",
        "fbduk_id": "F2",
    }

    @property
    def fbref_id(self):
        return self.value["fbref_id"]

    @property
    def fbref_name(self):
        return self.value["fbref_name"]

    @property
    def fbduk_id(self):
        return self.value["fbduk_id"]


@dataclass
class AppConfig:
    fbref_base_url: str = os.getenv("FBREF_BASE_URL")
    fbduk_base_url: str = os.getenv("FBDUK_BASE_URL")
    current_season: str = os.getenv("CURRENT_SEASON")
    rpi_diff_threshold: float = float(os.getenv("RPI_DIFF_THRESHOLD"))
    data_dir: Path = Path(os.getenv("DATA_DIR"))

    @property
    def fbref_data_dir(self):
        return self.data_dir / "FBREF"

    @property
    def fbduk_data_dir(self):
        return self.data_dir / "FBDUK"

    def get_fbref_league_dir(self, league_name):
        path = self.fbref_data_dir / league_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_fbduk_league_dir(self, league_name):
        path = self.fbduk_data_dir / league_name
        path.mkdir(parents=True, exist_ok=True)
        return path


# Function to parse comma-separated weeks into a list of integers
def parse_weeks(env_var, default=None):
    if env_var and env_var.strip():
        return [int(week) for week in env_var.split(",")]
    return default or []


# League weeks configuration loaded from environment
LEAGUE_WEEKS = {
    Leagues.EPL: parse_weeks(os.getenv("EPL_WEEKS"), []),
    Leagues.ECH: parse_weeks(os.getenv("ECH_WEEKS"), []),
    Leagues.EL1: parse_weeks(os.getenv("EL1_WEEKS"), []),
    Leagues.EL2: parse_weeks(os.getenv("EL2_WEEKS"), []),
    Leagues.ENL: parse_weeks(os.getenv("ENL_WEEKS"), []),
    Leagues.SP1: parse_weeks(os.getenv("SP1_WEEKS"), []),
    Leagues.SP2: parse_weeks(os.getenv("SP2_WEEKS"), []),
    Leagues.D1: parse_weeks(os.getenv("D1_WEEKS"), []),
    Leagues.D2: parse_weeks(os.getenv("D2_WEEKS"), []),
    Leagues.IT1: parse_weeks(os.getenv("IT1_WEEKS"), []),
    Leagues.IT2: parse_weeks(os.getenv("IT2_WEEKS"), []),
    Leagues.FR1: parse_weeks(os.getenv("FR1_WEEKS"), []),
    Leagues.FR2: parse_weeks(os.getenv("FR2_WEEKS"), []),
}

TIME_DELTA = int(os.getenv("TIME_DELTA"))

# Default app configuration
DEFAULT_CONFIG = AppConfig()
