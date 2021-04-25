"""
Created on Wed Apr  24 17:15:16 2021

@author: Thomas Erber
"""

import numpy as np
import numpy.ma as ma
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import math




class Stock:
	"This is the stock class, which contains methods for time history of prices etc and info about the stock and some metrics derived from the stockprice"

	def __init__(self,stockticker):
		self.stockticker=stockticker
		#stock = [stockticker]
		self.stock_obj = yf.Ticker(stockticker)
		self.stock_history=self.stock_obj.history(period="MAX") #pandas dataframe
		self.segment=""

	def GetTicker(self):
		return self.stockticker


	def GetMetric(self,Metric,Date):

		if (Metric=="ALL_TIME_HIGH"):
			SH=self.stock_history
			return max(SH[SH.index< Date ]['Close'])
			##other metrics to be added

	def GetStockPrice(self,Date):
		SH=self.stock_history
		if len(SH[SH.index==Date].index)==0:
			return None
		else:
			return SH[SH.index==Date]['Close'][0]
		###



class Stock_in_PF(Stock):
	"This is a stock_in_portfolio class, inherited from Stock class....has additional info like nr of stocks and price paid"

	def __init__(self,stockticker,nrofStocks):
		self.nrofStocks=nrofStocks
		Stock.__init__(self,stockticker)
		

	def GetNr(self):
		return self.nrofStocks

	def AddNr(self,amount):
		self.nrofStocks+=amount
		return self

	def RemNr(self,amount):
		self.nrofStocks-=amount
		return self

	def returnValue(self,Date):
		price=self.GetStockPrice(Date)
		return price*self.nrofStocks


class Portfolio:
	"This is the Portfolio class, should contain a dictionary of Stocks_in_Portfolio"

	def __init__(self,PortfDict):
		self.PortfDict=PortfDict

	def add(self,St_in_PF):
		ticker=St_in_PF.GetTicker()
		if ticker in self.PortfDict.keys(): #check if stock is already in portfolio
			self.PortfDict[ticker]= self.PortfDict[ticker].AddNr(St_in_PF.GetNr()) #if yes add nr of stocks
		else:
			self.PortfDict[ticker]=St_in_PF # if no, create new entry in dictionary

	def remove(self,St_in_PF):
		ticker=St_in_PF.GetTicker()
		if ticker in self.PortfDict.keys(): #check if stock is already in portfolio
			self.PortfDict[ticker]= self.PortfDict[ticker].RemNr(St_in_PF.GetNr()) #if yes add nr of stocks
			if self.PortfDict[ticker].GetNr()==0: #if stocknr is zero remove it
				self.PortfDict.pop(ticker)

	def GetContent(self):
		return self.PortfDict

	def CalcValue(self,Date):
		total=0
		for key in self.PortfDict:
			total+=self.PortfDict[key].returnValue(Date)
		return total

###create instance of stock class, do some tests
MSFT=Stock("MSFT")
Date=dt.datetime(2021, 1, 5)
print(type(MSFT))
print(MSFT.GetStockPrice(Date))
assert MSFT.GetStockPrice(Date)==217.4
print(MSFT.GetMetric("ALL_TIME_HIGH",Date))
assert MSFT.GetMetric("ALL_TIME_HIGH",Date)==230.51


##do some tests on Stock and Portfolio
MSFT=Stock_in_PF("MSFT",5)
Date=dt.datetime(2021, 1, 5)

TSLA=Stock_in_PF("TSLA",6)
print(type(MSFT))
print(MSFT.GetStockPrice(Date))
assert MSFT.GetStockPrice(Date)==217.4
print(MSFT.GetMetric("ALL_TIME_HIGH",Date))
assert MSFT.GetMetric("ALL_TIME_HIGH",Date)==230.51
print(MSFT.GetNr())
assert MSFT.GetNr()==5

PF=Portfolio({})
print(type(PF))
PF.add(MSFT)
print(PF.GetContent())
PF.add(TSLA)
print(PF.GetContent())
print(PF.GetContent()['MSFT'].GetNr())
assert PF.GetContent()['MSFT'].GetNr()==5
assert PF.GetContent()['TSLA'].GetNr()==6

MSFT2=Stock_in_PF("MSFT",8)
PF.add(MSFT2)
print(PF.GetContent())
print(PF.GetContent()['MSFT'].GetNr())
assert PF.GetContent()['MSFT'].GetNr()==13

print(PF.CalcValue(Date))
assert (TSLA.GetStockPrice(Date)*6+MSFT2.GetStockPrice(Date)*13)==PF.CalcValue(Date)

MSFT=Stock_in_PF("MSFT",7)
PF.remove(MSFT)
print(PF.GetContent())
print(PF.GetContent()['MSFT'].GetNr())
assert PF.GetContent()['MSFT'].GetNr()==6

MSFT=Stock_in_PF("MSFT",6)
PF.remove(MSFT)
print(PF.GetContent())
assert ("MSFT" not in PF.GetContent().keys())