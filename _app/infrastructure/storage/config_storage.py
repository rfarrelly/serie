import json
from pathlib import Path
from typing import Any, Dict


class ConfigStorage:
    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = config_dir
        self.config_dir.mkdir(exist_ok=True)

    def load_config(self, name: str) -> Dict[str, Any]:
        config_file = self.config_dir / f"{name}.json"
        if not config_file.exists():
            return {}

        with open(config_file, "r") as f:
            return json.load(f)

    def save_config(self, name: str, config: Dict[str, Any]) -> None:
        config_file = self.config_dir / f"{name}.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
