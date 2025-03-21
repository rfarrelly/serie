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

    @property
    def fbref_id(self):
        return self.value["fbref_id"]

    @property
    def fbref_name(self):
        return self.value["fbref_name"]

    @property
    def fbduk_id(self):
        return self.value["fbduk_id"]
