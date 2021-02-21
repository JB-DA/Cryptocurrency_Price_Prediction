
#from flask import Flask, jsonify, render_template, url_for, redirect
from flask import Flask, jsonify, render_template
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
@app.route("/api/assets")
def api_overview():
    con = sqlite3.connect("crypto_db.sqlite")
    df = pd.read_sql_query("SELECT * from historic_trades WHERE asset_id='brentoil'", con)
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
    app.run(debug=True)
