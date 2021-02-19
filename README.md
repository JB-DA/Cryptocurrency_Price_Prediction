![CryptoHeader](/resources/rm_header.png)
### CRYPTOCURRENCY PRICE PREDICTION with MACHINE LEARNING
> An adventure in predicting future prices of cryptocurrency

Bitcoin was invented in 2009 by a programmer known as Satoshi Nakamoto. The creation of Bitcoin follows precise rules derived from the gold market. So-called 'miners' competitively use computer resources to solve cryptographic problems and verify the validity of transactions. Success is rewarded by newly issued Bitcoin. The subsequent money creation evolves according to a fixed scheme pre-established by the inventor.
The value of Bitcoin solely depends on supply and demand. Currently, Bitcoins are actively traded against hard currencies on well-organized virtual exchange markets. These markets remain accessible during week-ends, which is valuable to investors, especially in hectic times.

The popularity of cryptocurrencies increased exponentially in 2017 due to several consecutive months’ growth of their market capitalization. The entire cryptocurrency market was worth more than $1,833,940,056,538 AUD in early February 2021 due to several factors that saw massive spikes in prices.

Machine learning can work through many problems, but its application in predicting crypto prices may be limited by several factors. As cryptocurrencies are decentralised, they have more factors that influence their price than traditional assets, which can make modelling & predictions very hard. Some factors affecting cryptocurrencies are technological progress, political pressures, security, consumer sentiment, sheer variety of currencies etc. Their high volatility can be very rewarding if they are traded at the correct time. Unfortunately, due to their lack of indexes, cryptocurrencies are relatively unpredictable compared to traditional financial models.

* [Synopsis](#--synopsis--)
* [Project Summary](#--project_summary--)
* [Resources](#--resources--)
* [Contributors](#--contributors--)
* [Task List](#--task_list--)

##### **- SYNOPSIS-**

This project aims to yield applicable models & predictions into the future pricing of selected cryptocurrencies. This will be achieved by using OHLCV (Open, high, low, close, volume) data of several cryptocurrencies, and passing the values into an ARIMA (Auto Regressive Integrated Moving Average) model for machine/deep learning. Target output will be aimed at the closing price for the following days/week, in order to try and judge the best time to purchase and sell. This can be gauged by comparing the delta values of daily trends in order to find the largest predicted returns.

Using ARIMA data from alternative asset classes (oil, gold, and S&P 500); we will make comparisons to the results returned from cryptocurrency predictions in order to benchmark the percentage returns of all classes.

##### **- PROJECT SUMMARY-**

Topic: Cryptocurrency Price Predictions using Machine Learning  
Target for Machine Learning: Closing delta prices  
Variables: OHLCV, internal model calculations (rolling average, rolling std dev)  
Model: ARIMA  
Sample Datasets:
![SampleData1](/resources/rm_sample1.png)
![SampleData2](/resources/rm_sample2.png)

##### **- RESOURCES -**

[CoinAPI.io](https://www.coinapi.io/)
[The CoinDesk 20](https://www.coindesk.com/coindesk20)

##### **- CONTRIBUTORS -**
* Divya [@github](https://github.com/divyagururajansumangala)
* JB [@github](https://github.com/JB-DA)

##### **- TASK LIST -**

 - [ ] Design Database
	 - [ ] Type
	 - [ ] ERD
	 - [ ] Schema
	 - [ ] Tables
	 - [ ] Columns
 
 - [ ] API calls for autonomous historical data collection
	 - [ ] Single item
	 - [ ] n items
	 - [ ] Error catching
	 - [ ] Storage of data
 
 - [ ] Data Cleaning 
 
 - [ ] Database Load
	- [ ] Append/Overwrite
 
 - [ ] Flask app for calling of data
	 - [ ] Routes
	 - [ ] Pages 
 
- [ ] Machine Learning Component
	- [ ] Data load
	- [ ] Data summary
	- [ ] Data visualise
	- [ ] Algorithm
	- [ ] Predictions

 - [ ] Webpage
	 - [ ] Layout
	 - [ ] Colour Scheme
	 - [ ] Menus
	 - [ ] Visualisations



