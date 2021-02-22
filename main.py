#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# IMPORTS
###
#
import requests
import json

#from apikeys import apikey_coinio as key

import sqlite3
from sqlite3 import Error

import glob
import pandas as pd

from sqlalchemy import create_engine
# END IMPORTS




import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

from flask import Flask, jsonify, render_template, request


# In[ ]:


# VARIABLES
###
#
# General
database = r"crypto_db.sqlite"
coins = ['BTC', 'ETH']
apikeys = "B44F0242-E0BA-4C1A-BED2-831A67426480", "1830D89F-A633-4F73-9707-3A7FAFE5C0F0", "200EF4DD-8BF3-4A8A-9FC9-CF9C9D6D1173"

# Database Schema
sql_create_assets_table = """ CREATE TABLE IF NOT EXISTS "assets" (
    "asset_id" VARCHAR   NOT NULL,
    "name" VARCHAR   NOT NULL,
    "type_is_crypto" INT   NOT NULL,
    "data_quote_start" VARCHAR   NOT NULL,
    "data_quote_end" VARCHAR   NOT NULL,
    "data_orderbook_start" VARCHAR   NOT NULL,
    "data_orderbook_end" VARCHAR   NOT NULL,
    "data_trade_start" VARCHAR   NOT NULL,
    "data_trade_end" VARCHAR   NOT NULL,
    "data_quote_count" VARCHAR   NOT NULL,
    "data_trade_count" VARCHAR   NOT NULL,
    "data_symbols_count" INT   NOT NULL,
    "volume_1hrs_usd" FLOAT   NOT NULL,
    "volume_1day_usd" FLOAT   NOT NULL,
    "volume_1mth_usd" FLOAT   NOT NULL,
    "price_usd" FLOAT   NOT NULL,
    PRIMARY KEY ("asset_id"),
    FOREIGN KEY ("asset_id") REFERENCES "historic_trades" ("asset_id")
);"""

sql_create_periods_table = """ CREATE TABLE IF NOT EXISTS "periods" (
    "period_id" VARCHAR   NOT NULL,
    "length_seconds" INT   NOT NULL,
    "length_months" INT   NOT NULL,
    "unit_count" INT   NOT NULL,
    "unit_name" VARCHAR   NOT NULL,
    "display_name" VARCHAR   NOT NULL
);"""

sql_create_current_rates_table = """ CREATE TABLE IF NOT EXISTS "current_rates" (
    "time" VARCHAR   NOT NULL,
    "asset_id_base" VARCHAR   NOT NULL,
    "asset_id_quote" VARCHAR   NOT NULL,
    "rate" FLOAT   NOT NULL,
    FOREIGN KEY ("asset_id_base") REFERENCES "assets" ("asset_id")
);"""

sql_create_exchanges_table = """ CREATE TABLE IF NOT EXISTS "exchanges" (
    "exchange_id" VARCHAR   NOT NULL,
    "website" VARCHAR   NOT NULL,
    "name" VARCHAR   NOT NULL,
    "data_start" VARCHAR   NOT NULL,
    "data_end" VARCHAR   NOT NULL,
    "data_quote_start" VARCHAR   NOT NULL,
    "data_quote_end" VARCHAR   NOT NULL,
    "data_orderbook_start" VARCHAR   NOT NULL,
    "data_orderbook_end" VARCHAR   NOT NULL,
    "data_trade_start" VARCHAR   NOT NULL,
    "data_trade_end" VARCHAR   NOT NULL,
    "data_symbols_count" INT   NOT NULL,
    "volume_1hrs_usd" FLOAT   NOT NULL,
    "volume_1day_usd" FLOAT   NOT NULL,
    "volume_1mth_usd" FLOAT   NOT NULL
);"""

sql_create_historic_trades_table = """ CREATE TABLE IF NOT EXISTS "historic_trades" (
    "asset_id" VARCHAR  NOT NULL,
    "time_period_start" VARCHAR   NOT NULL,
    "time_period_end" VARCHAR   NOT NULL,
    "time_open" VARCHAR   NULL,
    "time_close" VARCHAR   NULL,
    "price_open" FLOAT   NOT NULL,
    "price_high" FLOAT   NOT NULL,
    "price_low" FLOAT   NOT NULL,
    "price_close" FLOAT   NOT NULL,
    "volume_traded" FLOAT   NOT NULL,
    "trades_count" INT   NULL
);"""
# END VARIABLES


# DATABASE CONNECTION
###
##
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


# LOAD SCHEMA
###
#
conn = create_connection(database)

if conn is not None:
    execute_sql_cmd(conn, sql_create_assets_table)
    execute_sql_cmd(conn, sql_create_periods_table)
    execute_sql_cmd(conn, sql_create_current_rates_table)
    execute_sql_cmd(conn, sql_create_exchanges_table)
    execute_sql_cmd(conn, sql_create_historic_trades_table)
else:
    print("Error! cannot create the database connection.")
conn.close()
# END LOAD SCHEMA

# ETL ALL CSVS in 'data_raw'
###
#
def data_csv_load():
    find_csvs = glob.glob("data_raw/*.csv")

    for i in find_csvs:
        df_csv = pd.read_csv(i)
        df_csv['asset_id'] = find_csvs[find_csvs.index(i)].split("\\", 1)[
            1].split("_", 2)[0]
        df_csv.dropna(inplace=True)
        df_csv = df_csv.drop(columns=['Adj Close'])
        df_csv = df_csv.rename(columns={
            'Date': 'time_period_end',
            'Open': 'price_open',
            'High': 'price_high',
            'Low': 'price_low',
            'Close': 'price_close',
            'Volume': 'volume_traded',
        })

        df_csv['time_period_end'] = pd.to_datetime(
            df_csv['time_period_end'], format='%d/%m/%Y')
        df_csv['time_period_start'] = df_csv['time_period_end'].copy()

        with open(f'data_raw/{df_csv.asset_id[0]}.json', 'w') as f:
            f.write(df_csv.to_json())

        conn = create_connection(database)
        df_csv.to_sql("historic_trades", conn, if_exists="append", index=False)
        conn.close()
# END data_csv_load


# ETL ALL API CALL DATA
###
#
def data_api_load():
    # ASSETS
    # https://docs.coinapi.io/#list-all-assets
    url = 'https://rest.coinapi.io/v1/assets'
    headers = {'X-CoinAPI-Key': apikeys[0]}
    response = requests.get(url, headers=headers)

    with open('data_raw/assets.json', 'w') as ii:
        json.dump(response.json(), ii)

    # PERIODS
    endpoint = 'https://rest.coinapi.io'

    furl = f"{endpoint}/v1/ohlcv/periods"

    url = furl
    headers = {'X-CoinAPI-Key': apikeys[0]}
    response = requests.get(url, headers=headers)

    display(f'{response}')

    # save to asset_id_base.json
    with open(f'data_raw/periods.json', 'w') as ii:
        json.dump(response.json(), ii)

    # CURRENT RATES
    asset_id_base = 'USD'
    endpoint = 'https://rest.coinapi.io'

    furl = f"{endpoint}/v1/exchangerate/{asset_id_base}"

    url = furl
    headers = {'X-CoinAPI-Key': apikeys[0]}
    response = requests.get(url, headers=headers)

    display(f'{response}')

    # save to asset_id_base.json
    with open(f'data_raw/current_rates_{asset_id_base}.json', 'w') as ii:
        json.dump(response.json(), ii)

    # ALL EXCHANGES
    endpoint = 'https://rest.coinapi.io'

    furl = f"{endpoint}/v1/exchanges"

    url = furl
    headers = {'X-CoinAPI-Key': apikeys[0]}
    response = requests.get(url, headers=headers)

    display(f'{response}')

    # save to asset_id_base.json
    with open(f'data_raw/exchanges.json', 'w') as ii:
        json.dump(response.json(), ii)

    # COINS
    with open( 'data_raw/assets.json', 'r' ) as jj:
        json_d = json.load( jj )
        global df_assets
        df_assets = pd.DataFrame( json_d )
    
    for coin in coins:
        print(coin)
        print(apikeys[coins.index(coin)])
        asset_id_base = f'{coin}'
        asset_id_quote = 'USD'
        limit = 5000
        # database query on assets [coins.i] in data_Start column
        
        time_start = df_assets[df_assets['asset_id']== asset_id_base]['data_start']
        period_id = '1DAY'
        endpoint = 'https://rest.coinapi.io'
        include_empty_items = False

        furl = f"{endpoint}/v1/ohlcv/{asset_id_base}/{asset_id_quote}/history?period_id={period_id}&time_start={time_start.iloc[0]}&limit={limit}&include_empty_items={include_empty_items}"

        url = furl
        headers = {'X-CoinAPI-Key': apikeys[coins.index(coin)+1]}
        response = requests.get(url, headers=headers)

        # save to asset_id_base.json
        with open(f'data_raw/{asset_id_base}.json', 'w') as ii:
            json.dump(response.json(), ii)
        
        data = pd.DataFrame(response.json()) #load to dataframe
        
        conn = create_connection(database)
        data.to_sql("historic_trades", conn, if_exists="append", index=False)
        conn.close()
        
        display(data)
        
        return(data)
        
# END data_api_load


def model():
    pass
# END data_update


data_csv_load()
data_api_load()

x = data_api_load()


# In[ ]:





# In[ ]:


response
data = pd.DataFrame(response.json()) #load to dataframe


# In[ ]:


data


# In[ ]:


coins = ['BTC', 'ETH']
apikeys = "B44F0242-E0BA-4C1A-BED2-831A67426480", "1830D89F-A633-4F73-9707-3A7FAFE5C0F0", "200EF4DD-8BF3-4A8A-9FC9-CF9C9D6D1173"
i = 0

coins[i], apikeys[i]


# In[ ]:


with open('data_raw/assets.json', 'r') as jj:
    json_d = json.load(jj)
    df_assets = pd.DataFrame(json_d)


# In[ ]:


coin='BTC'
print(coin)
print(apikeys[coins.index(coin)])
asset_id_base = f'{coin}'
asset_id_quote = 'USD'
limit = 5
# database query on assets [coins.i] in data_Start column
time_start = df_assets[df_assets['asset_id']== asset_id_base]['data_start']
period_id = '1DAY'
endpoint = 'https://rest.coinapi.io'
include_empty_items = False

furl = f"{endpoint}/v1/ohlcv/{asset_id_base}/{asset_id_quote}/history?period_id={period_id}&time_start={time_start.iloc[0]}&limit={limit}&include_empty_items={include_empty_items}"

url = furl
headers = {'X-CoinAPI-Key': apikeys[coins.index(coin)+1]}
response = requests.get(url, headers=headers)

display(f'{response}')

# save to asset_id_base.json
with open(f'data_raw/{asset_id_base}.json', 'w') as ii:
    json.dump(response.json(), ii)


# In[ ]:


response.json()


# In[ ]:


# a = execute_sql_cmd(
#     conn, "SELECT * from historic_trades WHERE asset_id='brentoil'")
# conn.close()


con = sqlite3.connect(database)

a = pd.read_sql_query("SELECT * from historic_trades WHERE asset_id='brentoil'", con)
b = pd.read_sql_query(f"SELECT * from historic_trades WHERE asset_id='goldfutures'", con)
c = pd.read_sql_query("SELECT * from historic_trades WHERE asset_id='BTC'", con)
d = pd.read_sql_query(f"SELECT * from historic_trades WHERE asset_id='ETH'", con)
# df_csv.to_sql("historic_trades", con, if_exists="replace")

con.close()

display(a, b, c, d)


# In[ ]:


with open('data_raw/BTC.json', 'r') as jj:  # open api results
    json_d = json.load(jj)
    df_api = pd.DataFrame(json_d)  # save to dataframe
df_api['asset_id'] = 'BTC'
# df = df_api.head(50)
df = df_api

con = sqlite3.connect("crypto_db.sqlite")
df.to_sql("historic_trades", con, if_exists="append", index=False)
con.close()


# In[ ]:


with open('data_raw/ETH.json', 'r') as jj:  # open api results
    json_d = json.load(jj)
    df_api = pd.DataFrame(json_d)  # save to dataframe
df_api['asset_id'] = 'ETH'
# df = df_api.head(50)
df = df_api

con = sqlite3.connect("crypto_db.sqlite")
df.to_sql("historic_trades", con, if_exists="append", index=False)
con.close()


# In[ ]:


con = sqlite3.connect(database)

a = pd.read_sql_query(
    "SELECT * from historic_trades WHERE asset_id='brentoil'", con)
b = pd.read_sql_query(
    f"SELECT * from historic_trades WHERE asset_id='goldfutures'", con)
c = pd.read_sql_query(
    "SELECT * from historic_trades WHERE asset_id='BTC'", con)
d = pd.read_sql_query(
    f"SELECT * from historic_trades WHERE asset_id='ETH'", con)
# df_csv.to_sql("historic_trades", con, if_exists="replace")

con.close()

display(a, b, c, d)


# In[ ]:


con = sqlite3.connect("crypto_db.sqlite")
#pd.read_sql_query("SELECT * from historic_trades WHERE asset_id='brentoil'  max('time_period_end')", con)
con.close()


# WHERE Dates IN (SELECT max(Dates) FROM table);


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


# In[ ]:


df = c.copy()


# In[ ]:


from statsmodels.tsa.stattools import adfuller


def test_stationarity(timeseries, window=12, cutoff=0.01):
    # Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()
    # Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC', maxlag=100)
    dfoutput = pd.Series(dftest[0:4], index=[
                         'Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
        pvalue = dftest[1]
        if pvalue < cutoff:
            print('p-value = %.4f. The series is likely stationary.' % pvalue)
        else:
            print('p-value = %.4f. The series is likely non-stationary.' % pvalue)
    return(dfoutput)
#     print(dfoutput)


outs = test_stationarity(df['price_close'])
outs


# In[ ]:


import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})


# In[ ]:


# # Original Series
# fig, axes = plt.subplots(3, 2, sharex=True)
# axes[0, 0].plot(df["price_close"].values); axes[0, 0].set_title('Original Series')
# plot_acf(df["price_close"].values, ax=axes[0, 1])


# # 1st Differencing
# axes[1, 0].plot(np.diff(df["price_close"].values)); axes[1, 0].set_title('1st Order Differencing')
# df=df.dropna()
# plot_acf(np.diff(df["price_close"].values), ax=axes[1, 1])

# # 2nd Differencing
# axes[2, 0].plot(np.diff(df["price_close"].values)); axes[2, 0].set_title('2nd Order Differencing')
# plot_acf(np.diff(df["price_close"].values), ax=axes[2, 1])

# plt.show()


# In[ ]:


from statsmodels.tsa.arima_model import ARIMAResults
import joblib
train_data, test_data = df[0:int(len(df)*0.8)], df[int(len(df)*0.8):]

filename = f'{df.asset_id[0]}_model.sav'

training_data = train_data['price_close'].values
test_data = test_data['price_close'].values

history = [x for x in training_data]
model_predictions = []
N_test_observations = len(test_data)

for time_point in range(N_test_observations):
    model = ARIMA(history, order=(6, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    true_test_value = test_data[time_point]
    history.append(true_test_value)
model_fit.save(filename)

MSE_error = mean_squared_error(test_data, model_predictions)
print('Testing Mean Squared Error is {}'.format(MSE_error))


# In[ ]:


date_df = pd.DataFrame(df[int(len(df)*0.8):].time_period_end)
df_test = pd.DataFrame(data=test_data)
df_preds = pd.DataFrame(data=model_predictions)

#display(date_df.head(3), df_test.head(3), df_preds.head(3))


# In[ ]:


date_df.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)
df_preds.reset_index(drop=True, inplace=True)

df_preds = df_preds.shift(periods=-3)

# p = p.iloc[:-1]
# t = t.iloc[:-1]

frames = [date_df['time_period_end'], df_test[0], df_preds[0]]
headers = ['Date', 'test', 'preds']
graphdata = pd.concat(frames, axis=1, keys=headers)

# graphdata = df_test.merge(df_preds, how='inner', suffixes=('_actual', '_prediction'))
graphdata


# In[ ]:


import plotly.express as px
import plotly.graph_objects as go

fig = go.Figure([

    go.Scatter(
        name='Actual',
        x=graphdata.Date,
        y=graphdata['test'],
        mode='lines',
        marker=dict(color="#008080"),
        line=dict(width=1),
        showlegend=True
    ),
    go.Scatter(
        name='Prediction',
        x=graphdata.Date,
        y=graphdata['preds'],
        marker=dict(color="#FF8C00"),
        line=dict(width=1),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        showlegend=True
    )
])

fig.update_layout(
    yaxis_title='',
    title='',
    hovermode="x"
)

fig.show()


# In[ ]:


table = graphdata[['test', 'preds']].copy()
table.dropna(inplace=True)
table['delta'] = graphdata['preds'] - graphdata['test']
#table['perc'] = abs(graphdata['delta'] / graphdata['test'])*100

table.nlargest(10, ['delta'])

table


# In[ ]:


from statsmodels.tsa.arima_model import ARIMAResults
loaded = ARIMAResults.load(filename)

