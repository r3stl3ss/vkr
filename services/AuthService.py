import psycopg2

class AuthService:
    def authenticate(self, username, password):
        conn = psycopg2.connect(host="localhost", database="accounts", user="postgres", password="postgres")
        cursor = conn.cursor()


        query = "SELECT username, password FROM admin_accounts WHERE username = %s AND password = %s"
        cursor.execute(query, (username, password))

        result = cursor.fetchone()

        cursor.close()
        conn.close()
        if result is not None:
            return True
        else:
            return False
