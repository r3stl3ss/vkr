import psycopg2


class GetTickerService():
    def get_tickers(self):
        conn = psycopg2.connect(host="localhost", database="stocks", user="postgres", password="postgres")
        cursor = conn.cursor()
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        return table_names