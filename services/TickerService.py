import numpy as np
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class TickerService:
    def get_plot_url(self, ticker):
        conn = psycopg2.connect(host="localhost", database="stocks", user="postgres", password="postgres")
        cursor = conn.cursor()
        cursor.execute(f"SELECT date, close, predicted FROM {ticker} ORDER BY date ASC")
        data = cursor.fetchall()
        cursor.close()
        conn.close()
        df = pd.DataFrame(data, columns=['date', 'close', 'predicted'])
        df = df.fillna(0)
        df.set_index('date', inplace=True)
        plt.rcParams["figure.figsize"] = [8, 6]
        fig, ax = plt.subplots()
        ax.plot(df.index, df['close'], label='Price on close', color='green')
        ax.plot(df.index, df['predicted'], label='Predicted', color='red')
        ax.set_title(f'{ticker} Price on close')
        ax.set_ylabel('Price on close')
        ax.set_xlabel('Date')
        ax.legend()
        img = BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        return plot_url