from enum import Enum


class League(Enum):
    EPL = {"id": 9, "fbref_name": "Premier-League"}
    ECH = {"id": 10, "fbref_name": "Championship"}
    ENL = {"id": 34, "fbref_name": "National-League"}
    EL1 = {"id": 15, "fbref_name": "League-One"}
    EL2 = {"id": 16, "fbref_name": "League-Two"}
    SP1 = {"id": 12, "fbref_name": "La-Liga"}
    IT1 = {"id": 11, "fbref_name": "Serie-A"}

    @property
    def id(self):
        return self.value["id"]

    @property
    def fbref_name(self):
        return self.value["fbref_name"]
