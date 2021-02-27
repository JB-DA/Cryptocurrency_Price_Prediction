
#from flask import Flask, jsonify, render_template, url_for, redirect
from flask import Flask, jsonify, render_template, request
import sqlite3
from sqlalchemy import create_engine
import pandas as pd
import json


##### DATABASE SETUP
###
#
engine = create_engine(f"sqlite:///crypto_db.sqlite")
Base = automap_base()
Base.prepare(engine, reflect=True)

app = Flask(__name__, template_folder='static', static_folder='static')


# ROUTES
# API Pages
@app.route("/api/asset")
def api_asset():
    asset = request.args.get('asset')

    dbConnect = engine.connect()
    query = f"SELECT time_period_end, price_close from historic_data WHERE asset_id='{asset}'"
    df = pd.read_sql(query , conn)

    json_overview = json.loads(df.to_json(orient='records'))
    dbConnect.close()
    return jsonify(json_overview)

@app.route("/api/assets")
def api_assets():
    #asset = request.args.get('asset')

    dbConnect = engine.connect()
    query = f"SELECT DISTINCT asset_id FROM historic_data"
    df = pd.read_sql(query , conn).head(29)

    json_overview = json.loads(df.to_json(orient='records'))
    dbConnect.close()
    return jsonify(json_overview)

# HTML PAGES
@app.route('/')
def index():
    return render_template('index.html')

# RUN APP
if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug=True)
    