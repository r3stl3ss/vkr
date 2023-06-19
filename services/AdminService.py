import threading
from datetime import datetime

import psycopg2
from download_scripts import download_yahoo_v2, download_daily_bars


class AdminService:
    def refresh(self):
        conn = psycopg2.connect(host="localhost", database="accounts", user="postgres", password="postgres")
        cursor = conn.cursor()
        query = "SELECT last_updated FROM admin_accounts"
        cursor.execute(query)
        t1 = threading.Thread(target=download_yahoo_v2.main())
        t2 = threading.Thread(target=download_daily_bars.main())
        t1.start()
        t2.start()
        return True

    def check_updates(self):
        conn = psycopg2.connect(host="localhost", database="accounts", user="postgres", password="postgres")
        cursor = conn.cursor()
        query = "SELECT last_updated FROM admin_accounts"
        cursor.execute(query)
        result = cursor.fetchall()
        db_date = max(result)
        today = datetime.now().date()
        if today == db_date:
            return True
        return False