import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import psycopg2
import yahooquery as yq
from tqdm import tqdm

from ml_investment.applications.fair_marketcap_yahoo import FairMarketcapYahoo
from ml_investment.utils import load_config, load_tickers, save_json

global _data_path
_data_path = None

conn = psycopg2.connect(
    host="localhost",
    database="stocks",
    user="postgres",
    password="postgres"
)

cursor = conn.cursor()

def _single_ticker_download(ticker):
    try:
        ticker_data = yq.Ticker(ticker)
        amount = ticker_data.key_stats[ticker]["sharesOutstanding"]
        quarterly_df = ticker_data.all_financial_data('q')
        quarterly_df['date'] = quarterly_df.index
        quarterly_df.to_csv('{}/quarterly/{}.csv'.format(_data_path, ticker))

        save_json('{}/base/{}.json'.format(_data_path, ticker),
                  ticker_data.summary_profile)
        time.sleep(np.random.uniform(0.1, 0.5))
        fmy = FairMarketcapYahoo()
        result = fmy.execute(ticker)
        result_list = result['marketcap'].to_list()
        for value in result_list:
            predicted = value/amount
            query = f"INSERT INTO {ticker} (predicted) VALUES ('{predicted}')"
            cursor.execute(query)
    except:
        None


def main(data_path:str=None):
    if data_path is None:
        config = load_config()
        data_path = config['yahoo_data_path']

    global _data_path
    _data_path = data_path
    tickers = load_tickers()['base_us_stocks']
    os.makedirs('{}/quarterly'.format(data_path), exist_ok=True)
    os.makedirs('{}/base'.format(data_path), exist_ok=True)

    p = Pool(1)
    for _ in tqdm(p.imap(_single_ticker_download, tickers)):
        None




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--data_path', type=str)
    args = parser.parse_args()
    main(args.data_path)
 
