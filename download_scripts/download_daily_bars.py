import argparse
import os
import time
from datetime import datetime
from multiprocessing import Pool
from typing import List, Optional

import numpy as np
import psycopg2
from tqdm import tqdm
from yahooquery import Ticker

from ml_investment.applications.fair_marketcap_yahoo import FairMarketcapYahoo
from ml_investment.utils import load_config, load_tickers

conn = psycopg2.connect(
    host="localhost",
    database="stocks",
    user="postgres",
    password="postgres"
)

cursor = conn.cursor()


def _single_ticker_download(ticker):
    global _data_path
    global _from_date
    global _to_date
    success = False
    for _ in range(3):
        try:
            ht = Ticker(ticker)
            amount = ht.key_stats[ticker]["sharesOutstanding"]
            hdf = ht.history(start='2022-01-01', end=datetime.today().strftime('%Y-%m-%d'))
            hdf.to_csv('{}/{}.csv'.format(_data_path, ticker))
            time.sleep(np.random.uniform(0.2, 1.0))
            fmy = FairMarketcapYahoo()
            result = fmy.execute(ticker)
            result_list = result['marketcap'].to_list()
            for value in result_list:
                predicted = value / amount
                query = f"INSERT INTO {ticker} (predicted) VALUES ('{predicted}')"
                cursor.execute(query)
            success = True
            break
        except:
            None
            
    if not success and _verbose:
        print('Can not download {}'.format(ticker))


def main(data_path: str=load_config()['daily_bars_data_path'], 
         tickers: Optional[List]=load_tickers()['base_us_stocks'],
         from_date: Optional[np.datetime64]=np.datetime64('2022-01-01'),
         to_date: Optional[np.datetime64]=np.datetime64('now'),
         verbose: bool=True):
    global _data_path
    _data_path = data_path

    global _from_date
    _from_date = from_date
    
    global _to_date
    _to_date = to_date

    global _verbose
    _verbose = verbose

    os.makedirs(data_path, exist_ok=True)
    
    print('Start daily bars downloading: {}'.format(
            str(np.datetime64(int(time.time() * 1000), 'ms'))))
    with Pool(4) as p:
        for _ in tqdm(p.imap(_single_ticker_download, tickers),
                      disable=not verbose):
            None




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--data_path', type=str)
    arg('--verbose', type=bool)
    args = vars(parser.parse_args())
    args = {key:args[key] for key in args if args[key] is not None}  
    main(**args)


