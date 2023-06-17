
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Union, List
from ..utils import load_json, load_config


class DailyBarsData:
    def __init__(self,
                 data_path: Optional[str]=None,
                 days_count: Optional[int]=None):
        if data_path is None:
            data_path = load_config()['daily_bars_data_path']

        if days_count is None:
            days_count = int(1e5)

        self.data_path = data_path
        self.days_count = days_count


    def load(self, index: List[str]) -> pd.DataFrame:                     
        result = []
        for ticker in index:
            path = '{}/{}.csv'.format(self.data_path, ticker)
            if not os.path.exists(path):
                continue
            daily_df = pd.read_csv(path)[::-1][:self.days_count]
            daily_df['ticker'] = ticker
            daily_df['return'] = (daily_df['Adj Close'] / 
                                  daily_df['Adj Close'].shift(-1)).fillna(1)
            result.append(daily_df)

        if len(result) == 0:
            return

        if len(result) == 1:
            result = result[0]
        else:    
            result = pd.concat(result, axis=0).reset_index(drop=True)

        result = result.infer_objects()
        result['date'] = result['Date'].astype(np.datetime64) 
        result = result.reset_index(drop=True)

        return result


    def existing_index(self):
        index = [x.split('.csv')[0] for x in os.listdir(self.data_path)]
        return index 



