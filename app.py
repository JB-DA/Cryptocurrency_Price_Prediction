
from flask import Flask, jsonify, render_template, request
import sqlite3
from sqlite3 import Error
from sqlalchemy import create_engine
import pandas as pd
import json

database = r'crypto_db.sqlite'


def create_connection(db_file):
    """ create a database connection to the SQLite database specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return conn
# END create_connection


# RUN SQL COMMAND
###
#
def execute_sql_cmd(conn, command):
    """ run a sql command statement
    :param conn: Connection object
    :param execute_sql_cmd: run sql statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(command)
    except Error as e:
        print(e)
# END execute_sql_cmd

# DATABASE SETUP
###
#
# engine = create_engine(f"sqlite:///crypto_db.sqlite")
# Base = automap_base()
# Base.prepare(engine, reflect=True)


app = Flask(__name__, template_folder='static', static_folder='static')


# ROUTES
# API Pages
@app.route("/api/asset")
def api_asset():
    asset = request.args.get('asset')

    conn = create_connection(database)
    query = f"SELECT time_period_start, time_period_end, time_open, price_open, price_high, price_low, price_close FROM historic_data WHERE asset_id='{asset}'"

    if conn is not None:
        df = pd.read_sql_query(query, conn)
    else:
        print("Error! cannot create the database connection.")

    json_overview = json.loads(df.to_json(orient='records'))
    conn.close()
    return jsonify(json_overview)


@app.route("/api/assets")
def api_assets():
    #asset = request.args.get('asset')

    conn = create_connection(database)
    query = f"SELECT DISTINCT asset_id FROM historic_data"

    if conn is not None:
        df = pd.read_sql_query(query, conn)
    else:
        print("Error! cannot create the database connection.")

    df = pd.read_sql(query, conn)

    json_overview = json.loads(df.to_json(orient='records'))
    conn.close()
    return jsonify(json_overview)


@app.route("/api/volume")
def api_volume():
    #asset = request.args.get('asset')

    conn = create_connection(database)
    query = f"SELECT SUM(type_is_crypto) FROM assets WHERE type_is_crypto='1' AND price_usd!='null'"

    if conn is not None:
        df = pd.read_sql_query(query, conn)
    else:
        print("Error! cannot create the database connection.")

    df = pd.read_sql(query, conn)

    json_overview = json.loads(df.to_json(orient='records'))
    conn.close()
    return jsonify(json_overview)


@app.route("/api/predictions")
def api_prediction():
    asset = request.args.get('asset')

    from datetime import datetime, timedelta
    import joblib

    path = f'models/{asset}/'

    MODEL = joblib.load(f'{path}{asset}_model.smd')

    numdays = 365

    conn = create_connection(database)

    query = f"SELECT time_period_end, price_close FROM historic_data WHERE asset_id='{asset}' ORDER BY time_period_end ASC"

    if conn is not None:
        result = pd.read_sql_query(query, conn)
    else:
        print("Error! cannot create the database connection.")
    conn.close()

    result['time_period_end'] = pd.to_datetime(result['time_period_end'])

    plotstart = result['time_period_end'].iloc[-1]

    date = plotstart

    graph = pd.DataFrame({'date': [plotstart]})

    for i in range(numdays - 1):
        date += timedelta(days=1)
        graph = graph.append({'date': date}, ignore_index=True)


    forecast = MODEL.forecast(numdays)[0]
# MODEL.forecast() shows predicted value, std error of forecast, and confidence interval (lower & upper bounds)

    graph['forecast'] = forecast

    graph['forecast'][0] = result['price_close'].iloc[-1].copy()

    json_overview = json.loads(graph.to_json(orient='records'))
    return jsonify(json_overview)

# HTML PAGES


# type_is_crypto
# "volume_1hrs_usd" FLOAT   NOT NULL,
#     "volume_1day_usd" FLOAT   NOT NULL,
#     "volume_1mth_usd" FLOAT   NOT NULL


@app.route('/')
def index():
    return render_template('index.html')


# RUN APP
if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug=True)
