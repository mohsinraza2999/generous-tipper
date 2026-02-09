import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from src.utils.schemas import DataConfig

class DataSpliter:

    def __init__(self, df: pd.DataFrame, config: DataConfig):
        self.data_configs=config
        self.X, self.y= self._target_predictors(df)

    def split_data(self):
        return train_test_split(self.X, self.y, test_size=self.data_configs['test_size'],
                                random_state=self.data_configs['seed'], stratify=self.y)
    
    def _target_predictors(self, df: pd.DataFrame)->Tuple[pd.DataFrame, pd.Series]:

        return (df.drop(columns=self.data_configs['target'], axis=1), df[self.data_configs['target']])
