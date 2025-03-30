from enum import Enum


class League(Enum):
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

    @property
    def fbref_id(self):
        return self.value["fbref_id"]

    @property
    def fbref_name(self):
        return self.value["fbref_name"]

    @property
    def fbduk_id(self):
        return self.value["fbduk_id"]
