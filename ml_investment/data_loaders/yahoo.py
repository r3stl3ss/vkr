'''
Loader for dataset provided by yahoo.
Data may be downloaded by script
:func:`~ml_investment.download_scripts.download_yahoo.main`

Expected dataset structure:
    | path to Yahoo data folder with structure
    | Yahoo
    | ├── quarterly
    | │   ├── AAPL.csv
    | │   ├── FB.csv
    | │   └── ...
    | ├── base
    |     ├── AAPL.json
    |     ├── FB.json
    |     └── ...
'''

import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Union, List
from ..utils import load_json


class YahooQuarterlyData:
    '''
    Loader for quartely fundamental information about
    companies(debt, revenue etc)
    '''

    def __init__(self,
                 data_path: str,
                 quarter_count: Optional[int] = None):
        '''
        Parameters
        ----------
        data_path:
            path to :mod:`~ml_investment.data_loaders.yahoo` dataset folder
        quarter_count:
            maximum number of last quarters to return. 
            Resulted number may be less due to short history in some companies
        '''
        self.data_path = data_path
        self.quarter_count = quarter_count

    def load(self, index: List[str]) -> pd.DataFrame:
        '''
        Parameters
        ----------
        index:
            list of tickers to load data for
            
        Returns
        -------
        ``pd.DataFrame``
            quarterly information about companies
        '''
        result = []
        for ticker in index:
            path = '{}/quarterly/{}.csv'.format(self.data_path, ticker)
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df['ticker'] = ticker
            if self.quarter_count is not None:
                df = df[:self.quarter_count]
            result.append(df)

        if len(result) > 0:
            result = pd.concat(result, axis=0).reset_index(drop=True)
        else:
            return None

        result['asOfDate'] = result['asOfDate'].astype(np.datetime64)
        result = result.drop_duplicates(['symbol', 'asOfDate'])
        result.index = range(len(result))

        return result


class YahooBaseData:
    '''
    Loader for base information about company(like sector, industry etc)
    '''

    def __init__(self, data_path: str):
        '''
        Parameters
        ----------
        data_path:
            path to :mod:`~ml_investment.data_loaders.yahoo` dataset folder
        '''
        self.data_path = data_path

    def load(self, index: Optional[List[str]] = None) -> pd.DataFrame:
        '''
        Parameters
        ----------
        index:
            list of tickers to load data for
            OR ``None`` (for loading all possible tickers)
            
        Returns
        -------
        ``pd.DataFrame``
            base companies information
        '''
        result = {}
        base_path = '{}/base'.format(self.data_path)
        if index is None:
            index = [x.split('.json')[0] for x in os.listdir(base_path)]
        for ticker in index:

            path = '{}/{}.json'.format(base_path, ticker)
            if not os.path.exists(path):
                continue
            data = load_json(path)
            #parsed_data = json.loads(data)
            data[ticker]['ticker'] = ticker
            result.update(data[ticker])

        if len(result) == 0:
            return None

        result = pd.DataFrame(columns=['ticker', 'sector'])

        for ticker in index:

            path = '{}/{}.json'.format(base_path, ticker)
            if not os.path.exists(path):
                continue
            data = load_json(path)
            #parsed_data = json.loads(data)
            data['ticker'] = ticker
            addable_dict = {'ticker': ticker,
                            'sector': data[ticker]['sector']}
            addable_dict = {k: [v] for k, v in addable_dict.items()}
            debug = data[ticker]['sector']
            addable_df = pd.DataFrame.from_dict(addable_dict)
            result = pd.concat([result,addable_df])
            result.set_index('ticker', inplace=True)

        return result
