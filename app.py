import psycopg2
from flask import Flask, render_template, request, flash, redirect, url_for, Blueprint

from services.AdminService import AdminService
from services.AuthService import AuthService
from services.GetTickersService import GetTickerService
from services.TickerService import TickerService

app = Flask(__name__)
app.secret_key='doesnotmatter'
auth_bp = Blueprint('auth', __name__)
app.register_blueprint(auth_bp)


conn = psycopg2.connect(
    host="localhost",
    database="stocks",
    user="postgres",
    password="postgres",
    port="5432"
)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        auth_service = AuthService()
        if auth_service.authenticate(username, password):
            return redirect(url_for('panel'))
        else:
            flash('Invalid username or password', 'error')

    # Отображение страницы авторизации
    return render_template('register.html')

@app.route('/panel', methods=['GET', 'POST'])
def panel():
    admin_service = AdminService()
    if admin_service.check_updates():
        return redirect(url_for('refresh_data'))
    return render_template('refresh_ready.html')

@app.route('/ticker', methods=['GET', 'POST'])
def ticker():
    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        ticker_service = TickerService()
        plot_url = ticker_service.get_plot_url(ticker)
        return render_template('graph.html', ticker=ticker, plot_url=plot_url)
    else:
        return render_template('graph.html')

@app.route('/info')
def info_page():
    return render_template('info.html')

@app.route('/refresh_data')
def refresh_data():
    admin_service = AdminService()
    admin_service.refresh()
    return render_template('refreshing.html')

@app.route('/tickers_list')
def get_tables():
    get_service = GetTickerService()
    tickers = get_service.get_tickers()
    return render_template('ticker.html', tables=tickers)


@app.errorhandler(500)
def page_not_found(error):
    return render_template('error.html'), 500

if __name__ == '__main__':
    app.run()
