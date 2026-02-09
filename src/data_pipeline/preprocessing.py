import pandas as pd
import numpy as np
from typing import Tuple, List
from pathlib import Path
from src.data_pipeline.feature_engineering import FeatureEngineering
from src.data_pipeline.dataset import ReadData
from src.utils.configs import Configurations
from src.utils.logging import logger


class DataPreprocessing:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df=df

    def _upper_lower_quantile(self, col: str, threshold_variable:float)-> Tuple[float]:

        percentile25 = self.df[col].quantile(0.25)

        percentile75 = self.df[col].quantile(0.75)

        iqr = percentile75 - percentile25

        upper_limit = percentile75 + threshold_variable * iqr

        lower_limit = percentile25 - 1.5 * iqr
        if lower_limit < 0:
            lower_limit = 0
        return upper_limit, lower_limit


    def outliers_handler(self, outlier_config: List[list])-> None:

                
        for col, thres_var in outlier_config:
            upper_limit, lower_limit = self._upper_lower_quantile(col,thres_var)

            self.df[col] = np.where(self.df[col] < lower_limit,lower_limit, self.df[col] )
            
            self.df[col] = np.where(self.df[col] > upper_limit,upper_limit, self.df[col] )

    def null_values_handler(self)-> None:
        self.df=self.df.dropna(axis=0)

    def to_datetime(self, datetime_col: List[str])-> None:
        for col in datetime_col:
            self.df[col] = pd.to_datetime(self.df[col])


    def duplicate_handler(self)-> None:
        self.df=self.df.drop_duplicates()

    def feature_engineer(self, fe_configs)-> None:
        FE_object=FeatureEngineering()
        self.df[fe_configs['tip_col_eng']['name']]= FE_object.tip_percent(self.df,fe_configs['tip_col_eng']['use_col'])
        
        self.df[fe_configs['target_col_eng']['name']]=self.df[fe_configs['target_col_eng']['use_col']].apply(FE_object.generous_tipper)
        
        self.df[fe_configs['day_col_eng']['name']]= FE_object.day_of_week(self.df,fe_configs['day_col_eng']['use_col'])
        
        self.df[fe_configs['hours_col_eng']['name']]=FE_object.time_of_day(self.df,fe_configs['hours_col_eng']['use_col'])
        
    def drop_col(self, drop_list: List[str])-> None:
        self.df=self.df.drop(columns=drop_list,axis=1) 
        
    def save_data(self, root: Path)-> None:
        
        path=root/"processed_data.csv"
        self.df.to_csv(path, index= False)
        


def main():
    (logger.info if logger else print)("Data Processing Start")
    config_object=Configurations()
    data_configs=config_object.data_configs()
    
    (logger.info if logger else print)("Raw Data loading")
    read_object= ReadData(Path(data_configs['raw_path']))
    (logger.info if logger else print)("Raw Data Loaded")

    processor_object= DataPreprocessing(read_object.read(data_configs['name']))

    processor_object.null_values_handler()
    (logger.info if logger else print)("Handles Null Values")

    processor_object.duplicate_handler()
    (logger.info if logger else print)("Drop Duplicates")

    processor_object.outliers_handler(data_configs['outlier'])
    (logger.info if logger else print)("Handles Outliers")

    processor_object.to_datetime(data_configs['datetime'])

    (logger.info if logger else print)("Feature Engineering Starts")
    processor_object.feature_engineer(data_configs['feature_engineer'])
    (logger.info if logger else print)("Features Engineered as Required")

    processor_object.drop_col(data_configs['drop'])
    (logger.info if logger else print)("Drop Unnessary Columns")

    processor_object.save_data(Path(data_configs['processed_path']))
    (logger.info if logger else print)("Saved Preprocessed Data")



if __name__=="__main__":
    main()