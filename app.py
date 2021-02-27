
#from flask import Flask, jsonify, render_template, url_for, redirect
from flask import Flask, jsonify, render_template, request
import sqlite3
from sqlalchemy import create_engine
import pandas as pd
import json


# DATABASE SETUP
###
#

# FLASK SETUP
###
#
app = Flask(__name__, template_folder='static', static_folder='static')


# ROUTES
###
#
# API Pages
# @app.route("/api/volume1h")
# def api_volume1h():

#     asset = request.args.get('asset')

#     con = sqlite3.connect("crypto_db.sqlite")
#     df = pd.read_sql_query(f"SELECT * from assets WHERE volume_1hrs_usd='{asset}'", con)
#     json_assets1h = json.loads(df.to_json(orient='records'))
#     #df_csv.to_sql("historic_trades", con, if_exists="replace")
#     con.close()
#     return jsonify(json_assets1h)

# @app.route("/api/volume1d")
# def api_volume1d():

#     asset = request.args.get('asset')

#     con = sqlite3.connect("crypto_db.sqlite")
#     df = pd.read_sql_query(f"SELECT * from assets WHERE volume_1day_usd='{asset}'", con)
#     json_assets1d = json.loads(df.to_json(orient='records'))
#     #df_csv.to_sql("historic_trades", con, if_exists="replace")
#     con.close()
#     return jsonify(json_assets1d)

# @app.route("/api/volume1m")
# def api_volume1m():

#     asset = request.args.get('asset')

#     con = sqlite3.connect("crypto_db.sqlite")
#     df = pd.read_sql_query(f"SELECT * from assets WHERE volume_1mth_usd='{asset}'", con)
#     json_assets1m = json.loads(df.to_json(orient='records'))
#     #df_csv.to_sql("historic_trades", con, if_exists="replace")
#     con.close()
#     return jsonify(json_assets1m)

@app.route("/api/assets")
def api_asset_list():

    #asset = request.args.get('asset')

    con = sqlite3.connect("crypto_db.sqlite")
    df = pd.read_sql_query(f"SELECT DISTINCT asset_id FROM historic_data", con)
    json_assets = json.loads(df.to_json(orient='records'))
    #df_csv.to_sql("historic_trades", con, if_exists="replace")
    con.close()
    return jsonify(json_assets)








# HTML PAGES
@app.route('/')
def index():
    return render_template('index.html')


# RUN APP
###
#
if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug=True)
    