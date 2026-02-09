import pandas as pd
from typing import Dict
class FeatureEngineering:

    def __init__(self):
        pass

    def tip_percent(self,df: pd.DataFrame, cols_dict: Dict)-> pd.Series:
        return round(100*df[cols_dict['tip_col']]/(df[cols_dict['amount_col']] -df[cols_dict['tip_col']]),3)

    def generous_tipper(self, tip_percent: float)-> int:
        if tip_percent>=20:
            return 1
        return 0

    def day_of_week(self, df: pd.DataFrame,col:str)-> pd.Series:
        return df[col].dt.day_name().str.lower()

    def time_of_day(self, df: pd.DataFrame, col:str)-> pd.Series:

        hours=df[col].dt.hour

        return hours.apply(self._string_label)

    def _string_label(self,hours):
        if 6<= hours <10:
            return "am rush"
        elif 10<= hours <16:
            return "daytime"
        elif 16<= hours <20:
            return "pm rush"
        else:
            return "nighttime"


