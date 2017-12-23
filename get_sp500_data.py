"""
gets 2 years worth of stock data for sp500 companies and stores them as csv files in 
stock_dfs directory.
"""

import bs4 as bs
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests

def save_sp500_tickers():
	resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
	soup = bs.BeautifulSoup(resp.text, "lxml")
	table = soup.find('table', {'class': 'wikitable sortable'})
	tickers = []
	
	for row in table.findAll('tr')[1:]:
		ticker = row.findAll('td')[0].text
		tickers.append(ticker)
		
	with open("sp500tickers.pickle", "wb") as f:
		pickle.dump(tickers, f)
	
	print(tickers)
	return tickers
	

def getData(reload_sp500=False):
	if reload_sp500:
		tickers = save_sp500_ticers
	
	else: 
		with open("sp500tickers.pickle", "rb") as f:
			tickers = pickle.load(f)
	
	if not os.path.exists('stock_dfs'):
		os.makedirs('stock_dfs')
	
	start = dt.datetime(2014, 1, 1)
	end = dt.datetime(2016, 1, 1)
	
	for ticker in tickers:
		if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
			print('Getting {} data'.format(ticker))
			try:
				df = web.DataReader(ticker, 'yahoo', start, end)
				df.to_csv('stock_dfs/{}.csv'.format(ticker))
				print('Got {} data'.format(ticker))
				
			except:
				print('**Unable to get {} data'.format(ticker))
			
		else:
			print('Already have {}'.format(ticker))
			
getData()

# save_sp500_tickers()
		