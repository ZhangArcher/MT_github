import datetime
import os.path
import time

import pandas as pd
import numpy as np
from enum import Enum
import logging as log
import datetime
"""
data handler class for the stock prices , to receive and select the data set

 Attributes
 ----------
    price_data : Dataframe
        DataFrame of stock prices
    years: list
        a list of years , which is related to our price_data
    trading_time: list
        a list of the times from each date
 Methods
 -------
    __load_data:
        
    get_price_data:
        
    get_year_list:
        
    get_price_data_by_time_intetval:

 """


class data_handler_price:
    price_data :pd.DataFrame= None

    years=None
    trading_time=None
    def  __init__(self, csv_file: str):
        self.__load_data(csv_file)

    def __load_data(self, csv_file: str):
        """
            load data set
            and keep price data in data_handler_price
          :arg
          -------
            csv_file: str
                the path of the data set

            time_type: time_type
                the time unit of the data set ,
                for example , time_type of daily price is "Day" ,
                time_type of monthly price is "Month"
          :return
          -------
            None
        """

        assert os.path.isfile(csv_file)

        # read data set
        try:
            self.price_data = pd.read_csv(csv_file)
        except Exception:
            print("can not read csv data , please check the target data set :"+csv_file)

        #refine price data
        self.price_data = self.price_data.iloc[:, [0, 4]]
        self.price_data = self.price_data.dropna()

        self.price_data = self.__refine_timestamp()

        self.price_data.Date = pd.to_datetime(self.price_data.Date)

        # get the years that the data is related to
        self.years = self.get_year_list(self.price_data)
        # get the time point that the data is related to
        self.trading_time=self.price_data.Date.copy()

    def get_price_data(self):
        """
        sort price_data by "timestamp" and return the sorted price_data
        :arg
        -------
             None
        :return
        -------
             df_price:  Dataframe
             a sorted price_data
        """
        assert len(self.years)>0
        assert self.price_data.empty==False
        assert self.years!=None

        df_price=self.price_data.copy()
        df_price.sort_values("timestamp")

        return df_price


    def __refine_timestamp(self):

        """
        sort price_data and add new attribute "timestamp" into price_data according to attribute "Date"
        for example:
            "Date":1990-01-01 00:00:00-05:00
            "timestamp": 1990-01-01 00:00:00
      :arg
      -------
            None
      :return
      -------
            result_price: Dataframe
                The new price data with "timestamp"
        """
        for time_d in self.price_data.loc[:,"Date"]:
            time_str = (time_d[:19])
            result =time.mktime(time.strptime(time_str, '%Y-%m-%d %X'))
            self.price_data.loc[self.price_data["Date"]==time_d,"timestamp"]=result
            self.price_data.loc[self.price_data["Date"] == time_d, "Date"]=time_str
        return self.price_data

    def get_price_data_by_time_intetval(self,start_time :datetime.datetime, end_time:datetime.datetime):
        """
            return prices data according to time interval

        :arg
        -------
            start_time: datetime.datetime
                the start time of time interval
            end_time: datetime.datetime
               the end time of time interval
        :return
        -------
            result_price: Dataframe
              The corresponding price data
        """
        start_timestamp=datetime.datetime.timestamp(start_time)
        end_timestamp = datetime.datetime.timestamp(end_time)
        price_d=self.price_data[self.price_data["timestamp"]>=start_timestamp]
        result_price=price_d[price_d["timestamp"]<end_timestamp]
        result_price.sort_values("timestamp")
        return result_price

    def get_year_list(self,price_data):
        """
           genereate a list of years from price_data

        :arg
        -------
            price_data: Dataframe
                price data
        :return
        -------
            year_list:  list
                all years from price data
        """
        year_list_temp=list(price_data.Date)
        year_list=list({year_list_temp[i].year for i in range(0, len(year_list_temp))})
        return year_list

    def get_trading_time(self):
        """
           return a list of all trading times of each data

        :arg
        -------
            price_data: Dataframe
                price data
        :return
        -------
            year_list:  list
                all years from price data
        """


        return self.trading_time


if __name__ == '__main__':
    data_file="data_set_price/long_1mo_with_back_ADJ/AAPL.csv"
    dpp=data_handler_price(data_file)
    #dpp.reorgnize_price_data()
    print("asdsd")