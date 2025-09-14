from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class CSVFileAdapter:
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)

    def read_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(self.base_path / filename, **kwargs)

    def write_csv(self, data: pd.DataFrame, filename: str, **kwargs) -> None:
        data.to_csv(self.base_path / filename, index=False, **kwargs)

    def write_dict_list(self, data: List[Dict[str, Any]], filename: str) -> None:
        df = pd.DataFrame(data)
        self.write_csv(df, filename)

    def file_exists(self, filename: str) -> bool:
        return (self.base_path / filename).exists()
