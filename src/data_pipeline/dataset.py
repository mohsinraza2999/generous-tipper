import pandas as pd
from pathlib import Path
from typing import Tuple, List
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.data_pipeline.splitter import DataSpliter
from src.utils.schemas import DataConfig
from src.utils.logging import logger


class ReadData:

    def __init__(self, root: Path):
        if not root.exists():
            (logger.error if logger else print)(f"Directory not found: {root}")
            raise FileNotFoundError(f"Directory not found: {root}")
        self.root=root

    def read(self, name: str)-> pd.DataFrame:
        path=self.root/name
        if not path.exists():
            (logger.error if logger else print)(f"Directory not found: {path}")
            raise FileNotFoundError(f"Data path not exists or not found on: {path}")
        return pd.read_csv(path)



def train_val_data(config:DataConfig
                   )->Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    (logger.info if logger else print)("Processed Data loading")
    read_object=ReadData(Path(config['processed_path']))
    (logger.info if logger else print)("Processed Data loaded")
    splitter_object=DataSpliter(read_object.read(config['name']), config)

    return splitter_object.split_data()

def data_pipeline(cat_cols: list[str], numeric_cols: List[str]):

    (logger.info if logger else print)("Building Data Pipeline")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols or []),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols or [])
        ],
        remainder="drop"
    )
    return preprocessor

