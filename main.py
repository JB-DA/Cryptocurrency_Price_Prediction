
### IMPORTS
#
from flask import Flask, jsonify, render_template, request
from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine
from sqlite3 import Error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from statsmodels.tsa.stattools import adfuller
import dataframe_image as dfi
import datetime
import glob
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
import sqlite3
import time
# END IMPORTS


# if save_json == 'yes' json dump
# else pass









### VARIABLES
#
# General
database = r'crypto_db.sqlite'
coins = ['DOGE', 'BTC', 'ETH', 'LTC']
apikeys = 'EFA335E9-DA1A-42CC-A498-DFC46281CE85','1C124E08-4D3D-45BC-9EB5-AB1AC5206B47','B44F0242-E0BA-4C1A-BED2-831A67426480', '1830D89F-A633-4F73-9707-3A7FAFE5C0F0', '200EF4DD-8BF3-4A8A-9FC9-CF9C9D6D1173'

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
    FOREIGN KEY ("asset_id") REFERENCES "historic_data" ("asset_id")
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

sql_create_exchange_rates_table = """ CREATE TABLE IF NOT EXISTS "exchange_rates" (
    "time" VARCHAR   NOT NULL,
    "asset_id_quote" VARCHAR   NOT NULL,
    "rate" FLOAT   NOT NULL,
    FOREIGN KEY ("asset_id_quote") REFERENCES "assets" ("asset_id")
);"""

sql_create_periods_table = """ CREATE TABLE IF NOT EXISTS "periods" (
    "period_id" VARCHAR   NOT NULL,
    "length_seconds" INT   NOT NULL,
    "length_months" INT   NOT NULL,
    "unit_count" INT   NOT NULL,
    "unit_name" VARCHAR   NOT NULL,
    "display_name" VARCHAR   NOT NULL
);"""

sql_create_historic_data_table = """ CREATE TABLE IF NOT EXISTS "historic_data" (
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
time.sleep(1)
# END VARIABLES


### DATABASE CONNECTION
#
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


### RUN SQL COMMAND
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


### LOAD SCHEMA
#
conn = create_connection(database)

if conn is not None:
    execute_sql_cmd(conn, sql_create_assets_table)
    execute_sql_cmd(conn, sql_create_exchanges_table)
    execute_sql_cmd(conn, sql_create_exchange_rates_table)
    execute_sql_cmd(conn, sql_create_periods_table)
    execute_sql_cmd(conn, sql_create_historic_data_table)
else:
    print("Error! cannot create the database connection.")
conn.close()
# END LOAD SCHEMA


### ETL ALL CSVS in 'data_raw'
#
def data_csv_load():
    
    csvs = glob.glob('data_raw/*.csv')

    for csv in csvs:
        df_csv = pd.read_csv(csv)
        df_csv.dropna(inplace = True)
        df_csv = df_csv.rename(columns = {
            df_csv.columns[df_csv.columns.str.contains(pat = 'date', case = False)][0]: 'time_period_end',
            df_csv.columns[df_csv.columns.str.contains(pat = 'open', case = False)][0]: 'price_open',
            df_csv.columns[df_csv.columns.str.contains(pat = 'high', case = False)][0]: 'price_high',
            df_csv.columns[df_csv.columns.str.contains(pat = 'low', case = False)][0]: 'price_low',
            df_csv.columns[df_csv.columns.str.contains(pat = 'close', case = False)][0]: 'price_close',
            df_csv.columns[df_csv.columns.str.contains(pat = 'vol', case = False)][0]: 'volume_traded',
        })
        df_csv['asset_id'] = csvs[csvs.index(csv)].split('\\', 1)[1].split('_', 2)[0]
        df_csv['time_period_end'] = pd.to_datetime(df_csv['time_period_end'], format = '%d/%m/%Y')
        df_csv['time_period_start'] = df_csv['time_period_end'].copy()

        df_csv = df_csv[['asset_id','time_period_start','time_period_end','price_open','price_high','price_low','price_close','volume_traded']].copy()

        #with open(f'data_raw/{df_csv.asset_id[0]}.json', 'w') as f:
        #    f.write(df_csv.to_json())

        conn = create_connection(database)
        df_csv.to_sql('historic_data', conn, if_exists = 'append', index = False) ##switch to append once dev finished
        conn.close()
# END data_csv_load


### ETL ALL API CALL DATA
#
def data_api_load():
        
    conn = create_connection(database)
    
    endpoint = 'https://rest.coinapi.io/v1'
    asset_id_base = 'USD'
    
    # ASSETS TABLE
    url = f'{endpoint}/assets'
    headers = {'X-CoinAPI-Key': apikeys[0]}
    response = requests.get(url, headers = headers)    
    with open('data_raw/assets.json', 'w') as ii:
        json.dump(response.json(), ii)
    pd.DataFrame(response.json()).to_sql("assets", conn, if_exists = 'replace', index = False)
    
    # EXCHANGES TABLE
    url = f'{endpoint}/exchanges'
    headers = {'X-CoinAPI-Key': apikeys[0]}
    response = requests.get(url, headers = headers)
    #with open(f'data_raw/exchanges.json', 'w') as ii:
    #    json.dump(response.json(), ii)
    pd.DataFrame(response.json()).to_sql("exchanges", conn, if_exists = 'replace', index = False)
    
    # EXCHANGE RATES TABLE
    url = f'{endpoint}/exchangerate/{asset_id_base}'
    headers = {'X-CoinAPI-Key': apikeys[0]}
    response = requests.get(url, headers = headers)
    #with open(f'data_raw/exchange_rates.json', 'w') as ii:
    #    json.dump(response.json(), ii)
    pd.DataFrame(response.json()['rates']).to_sql("exchange_rates", conn, if_exists = 'replace', index = False)

    # TIME PERIODS TABLE
    url = f'{endpoint}/ohlcv/periods'
    headers = {'X-CoinAPI-Key': apikeys[0]}
    response = requests.get(url, headers = headers)
    #with open(f'data_raw/periods.json', 'w') as ii:
    #    json.dump(response.json(), ii)
    pd.DataFrame(response.json()).to_sql("periods", conn, if_exists = 'replace', index = False)

    # CRYPTOCURRENCY    
    for coin in coins:
        query = f"SELECT data_start from assets WHERE asset_id='{coin}'"
        if conn is not None:
            result = pd.read_sql_query(query,conn)
        else:
            print("Error! cannot create the database connection.")
            
        endpoint = 'https://rest.coinapi.io/v1'
        asset_id_base = f'{coin}'
        asset_id_quote = 'USD'
        limit = 5000
        time_start = result['data_start'][0]
        period_id = '1DAY'
        include_empty_items = False

        url = f'{endpoint}/ohlcv/{asset_id_base}/{asset_id_quote}/history?period_id={period_id}&time_start={time_start}&limit={limit}&include_empty_items={include_empty_items}'

        headers = {'X-CoinAPI-Key': apikeys[coins.index(coin) + 2]}
        response = requests.get(url, headers = headers)

        #with open(f'data_raw/{asset_id_base}.json', 'w') as ii:
        #    json.dump(response.json(), ii)
        
        df_coin = pd.DataFrame(response.json())
        df_coin['asset_id'] = coin

        df_coin.to_sql("historic_data", conn, if_exists = "append", index = False)
    conn.close()        
# END data_api_load


def data_model(*assets):
    #MODEL FUNCTION
    plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})
    split = 80
    maxlags = 6

    conn = create_connection(database)
    query = f"SELECT DISTINCT asset_id FROM historic_data"
    if conn is not None:
        result = pd.read_sql_query(query,conn)
    else:
        print("Error! cannot create the database connection.")
    conn.close()

    assets = result.copy()

    for asset in assets['asset_id']:        
        try:
            os.makedirs(f'models/{asset}')
        except FileExistsError:
            pass

        path = f'models/{asset}/'

        conn = create_connection(database)
        query = f"SELECT time_period_end, price_close from historic_data WHERE asset_id='{asset}'"

        if conn is not None:
            result = pd.read_sql_query(query,conn)
        else:
            print("Error! cannot create the database connection.")
        conn.close()

        df = result.copy()

        # DICKEY-FULLER TEST FOR TIME SERIES STATIONARITY, test if data series changes with time
        def test_stationarity(timeseries, window = 12, cutoff = 0.05):
            # Calculates rolling mean & standard deviation
            rolmean = timeseries.rolling(window).mean()
            rolstd = timeseries.rolling(window).std()
            # Plot rolling statistics
            fig = plt.figure(figsize = (12, 8))
            orig = plt.plot(timeseries, color = 'blue', label = 'Original')
            mean = plt.plot(rolmean, color = 'red', label = 'Rolling Mean')
            std = plt.plot(rolstd, color = 'black', label = 'Rolling Std')
            plt.legend(loc = 'best')
            plt.title(f'Rolling Mean & Standard Deviation: {asset}')
            plt.savefig(f'{path}{asset}_mean_std.png')
            plt.close()    
            # Performs dickey-fuller test for time-series statistics
            dftest = adfuller(timeseries, autolag = 'AIC', maxlag = 100) #AIC estimator of prediction error
            dfoutput = pd.Series(dftest[0:4], index = [ #Display results
                                 'Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
            for key, value in dftest[4].items():
                dfoutput['Critical Value (%s)' % key] = value #If critical value < test statistics, dataset is stationary
                pvalue = dftest[1]
            #if pvalue < cutoff: #If pvalue < cutoff (0.05), dataset is stationary
            #    print('p-value = %.4f. The series is likely stationary.' % pvalue)
            #else:
            #    print('p-value = %.4f. The series is likely non-stationary.' % pvalue)
            return(dfoutput, cutoff)

        fuller_out, cutoff = test_stationarity(df['price_close'])

        df_fuller = pd.DataFrame(fuller_out)
        df_fuller = df_fuller.rename(columns = {
            0 : f'Dickey-Fuller Test Scores: {asset}'})

        if df_fuller.iloc[1][0] < cutoff: #If pvalue < 0.05, dataset is stationary
            df_fuller.loc['Data Stationary?'] = 'Yes'
        else:
            df_fuller.loc['Data Stationary?'] = 'No'

        df_fuller.dfi.export(f'{path}{asset}_dickeyfuller.png')
        
        #     Lag plots allow you to check for:
        #     Model suitability
        #     Outliers (data points with extremely high or low values)
        #     Randomness (data without a pattern)
        #     Serial correlation (where error terms in a time series transfer from one period to another)
        #     Seasonality (periodic fluctuations in time series data that happens at regular periods)

        lags = int(df_fuller.loc['#Lags Used'][0])
        
        # plt.figure()
        lag_plot(df['price_close'], lag = lags)
        plt.title(f'Lag of timeseries with lag = {lags}: {asset} (real)')
        plt.savefig(f'{path}{asset}_series_lag_real.png')
        plt.close()

        if lags > maxlags: #Sets maximum lag value
            lags = maxlags
        else:
            lags = int(df_fuller.loc['#Lags Used'][0])

        # plt.figure()
        lag_plot(df['price_close'], lag = lags)
        plt.title(f'Lag of timeseries with lag = {lags}: {asset} (capped)')
        plt.savefig(f'{path}{asset}_series_lag_capped.png')
        plt.close()

        # PLOTS d value first & second differentiation
        # Original Series
        fig, axes = plt.subplots(3, 2, sharex = True)
        axes[0, 0].plot(df["price_close"].values); axes[0, 0].set_title('Original Series')
        plot_acf(df["price_close"].values, ax = axes[0, 1])

        # 1st Differencing
        axes[1, 0].plot(np.diff(df["price_close"].values)); axes[1, 0].set_title('1st Order Differencing')
        df = df.dropna()
        plot_acf(np.diff(df["price_close"].values), ax = axes[1, 1])

        # 2nd Differencing
        axes[2, 0].plot(np.diff(df["price_close"].values)); axes[2, 0].set_title('2nd Order Differencing')
        plot_acf(np.diff(df["price_close"].values), ax = axes[2, 1])

        plt.suptitle(f'Differencing & Correlation: {asset}')
        plt.savefig(f'{path}{asset}_1order_2order.png')
        plt.close()    

        # CREATE TRAIN & TEST DATA FOR MODELLING
        train_data, test_data = df[0:int(len(df) * split / 100)], df[int(len(df) * split / 100):]

        training_data = train_data['price_close'].values
        test_data = test_data['price_close'].values

        history = [x for x in training_data]
        model_predictions = []
        N_test_observations = len(test_data)

        # CREATE MODEL
        for time_point in range(N_test_observations):
            model = ARIMA(history, order = (lags, 1, 0))
            model_fit = model.fit(disp = 0)
            output = model_fit.forecast()
            yhat = output[0]
            model_predictions.append(yhat)
            true_test_value = test_data[time_point]
            history.append(true_test_value)

        model_fit.save(f'{path}{asset}_model.smd')

        MSE_error = mean_squared_error(test_data, model_predictions)

        # CREATE DATAFRAME WITH DATES, ACTUAL & PREDICTED VALUES
        date_df = pd.DataFrame(df[int(len(df) * split / 100):].time_period_end)
        df_test = pd.DataFrame(data = test_data)
        df_preds = pd.DataFrame(data = model_predictions)

        date_df.reset_index(drop = True, inplace = True)
        df_test.reset_index(drop = True, inplace = True)
        df_preds.reset_index(drop = True, inplace = True)

        df_preds = df_preds.shift(periods = -3)

        frames = [date_df['time_period_end'], df_test[0], df_preds[0]]
        headers = ['Date', 'test', 'preds']
        df_graphdata = pd.concat(frames, axis = 1, keys = headers)

        # Create Training and Test
        split = 85
        train = df[0:int(len(df['price_close'])*split/100)].price_close
        test = df[int(len(df['price_close'])*split/100):].price_close

        # Build Model
        # model = ARIMA(train, order=(2,1,0))  
        model = ARIMA(np.asarray(train), order = (2, 1, 0))  
        fitted = model.fit(disp=0)  

        # Forecast
        fc, se, conf = fitted.forecast(len(test), alpha = 0.05)  # 95% conf

        # Make as pandas series
        fc_series = pd.Series(fc, index=test.index)

        lower_series = pd.Series(conf[:, 0], index=test.index)
        upper_series = pd.Series(conf[:, 1], index=test.index)

        # Plot
        plt.figure(figsize = (12,5), dpi = 100)
        plt.plot(train, label = 'training')
        plt.plot(test, label = 'actual')
        plt.plot(fc_series, label = 'forecast')
        plt.fill_between(lower_series.index, lower_series, upper_series, color = 'k', alpha = 0.15)
        plt.title(f'Forecast vs Actuals 1st Order: {asset}')
        plt.legend(loc='upper left', fontsize = 8)
        plt.savefig(f'{path}{asset}_oot_forecast_order1.png')
        plt.close()    

        #Build Model with Forecast
        # model = ARIMA(train, order=(2,1,0))  
        model = ARIMA(np.asarray(train), order = (3, 2, 0))  
        fitted = model.fit(disp=0)  

        # Forecast
        fc, se, conf = fitted.forecast(len(test), alpha = 0.05)  # 95% conf

        # Make as pandas series
        fc_series = pd.Series(fc, index=test.index)

        lower_series = pd.Series(conf[:, 0], index=test.index)
        upper_series = pd.Series(conf[:, 1], index=test.index)

        # Plot
        plt.figure(figsize = (12,5), dpi = 100)
        plt.plot(train, label = 'training')
        plt.plot(test, label = 'actual')
        plt.plot(fc_series, label = 'forecast')
        plt.fill_between(lower_series.index, lower_series, upper_series, 
                         color = 'k', alpha = 0.15)
        plt.title(f'Forecast vs Actuals 2nd Order: {asset}')
        plt.legend(loc = 'upper left', fontsize = 8)
        plt.savefig(f'{path}{asset}_oot_forecast_order2.png')
        plt.close()

# END data_model

#data_csv_load()
data_api_load()
data_model()









# import plotly.express as px
# import plotly.graph_objects as go

# fig = go.Figure([

#     go.Scatter(
#         name='Actual',
#         x=graphdata.Date,
#         y=graphdata['test'],
#         mode='lines',
#         marker=dict(color="#008080"),
#         line=dict(width=1),
#         showlegend=True
#     ),
#     go.Scatter(
#         name='Prediction',
#         x=graphdata.Date,
#         y=graphdata['preds'],
#         marker=dict(color="#FF8C00"),
#         line=dict(width=1),
#         mode='lines',
#         fillcolor='rgba(68, 68, 68, 0.3)',
#         showlegend=True
#     )
# ])

# fig.update_layout(
#     yaxis_title='',
#     title='',
#     hovermode="x"
# )

# fig.show()

