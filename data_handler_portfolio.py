import datetime
import os.path
import time

import pandas as pd
import numpy as np
from enum import Enum
import logging as log
import datetime

"""
 data handler class for the Portfolio , to receive and preprocess the data set

 Attributes
 ----------
    __portfolio_data : DataFrame
        DataFrame of historical portfolios
        
    __asset_position_dict :dict
        key : stock_id
        values :  Series  timestamp & position 
    
    __trading_time: pd.Series
        a Series of the trading time points from the portfolio data
        
    __assets_id_list: list
        a list which contains all stock ids.
 Methods
 -------
    __load_data:

 
 """


class data_handler_portfolio:
    __portfolio_data = None
    __asset_position_dict = None
    __trading_time=None
    __assets_id_list=None


    def __init__(self, csv_file: str):
        self.__load_data(csv_file)

    def __load_data(self, csv_file: str):
        """
        load data set
        input:
            csv_file: str
                the path of the data set

            time_type: time_type
                the time unit of the data set ,
                for example , time_type of daily price is "Day" , time_type of monthly price is "Month"
        output:
            None
        """
        assert os.path.isfile(csv_file)
        self.__portfolio_data = pd.read_csv(csv_file,index_col=0)

        # self.df.iloc[:, [0, 4]] : just select date and close
        self.__asset_position_dict={}
        self.__trading_time = pd.Series(pd.to_datetime(self.__portfolio_data.index.str[:18]))
        self.__portfolio_data.index = pd.to_datetime(self.__portfolio_data.index)
        #self.__trading_time = pd.to_datetime(self.__trading_time)
        for i in self.__portfolio_data.iloc[:]:
            fff=self.__portfolio_data.loc[:, i]
            self.__asset_position_dict[i]=self.__portfolio_data.loc[:,i]
        self.__assets_id_list=list(self.__portfolio_data.columns[0:-1])
        print("assds")


    def get_historical_position_by_id(self,id:str):
        assert self.__assets_id_list.count(id)==1
        historical_pos=self.__asset_position_dict[id]
        return historical_pos

    def get_historical_position_by_id_time(self,id:str,start:datetime.datetime,end:datetime.datetime):
        assert self.__assets_id_list.count(id) == 1
        historical_pos = self.__asset_position_dict[id]
        ddd=historical_pos.index[0]
        historical_pos=historical_pos[historical_pos.index>=start]
        historical_pos = historical_pos[historical_pos.index<end]
        return historical_pos

    def get_assets_id_list(self):
        return self.__assets_id_list

    def get_trading_time(self):
        return self.__trading_time


if __name__ == '__main__':
    file_path="portfolios_long_1month.csv"
    ppp=data_handler_portfolio(csv_file=file_path)
    id_name="ABT"
    ddd=ppp.get_historical_position_by_id(id_name)
    print("asdsd")