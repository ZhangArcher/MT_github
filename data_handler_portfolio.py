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
        values :  Series  ,which contains  timestamp & position 
    
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
        """
                To initialize  class
                :arg
                -------
                      csv_file: str
                          the path of the portfolios

                  :return
                -------
                      None
                """
        self.__load_data(csv_file)

    def __load_data(self, csv_file: str):
        """
        load data set
        :arg
        -------
            csv_file: str
                the path of the data set

        :return
        -------
            None
        """
        assert os.path.isfile(csv_file) , "can not find such file:"+csv_file
        self.__portfolio_data = pd.read_csv(csv_file,index_col=0)

        # self.df.iloc[:, [0, 4]] : just select date and close
        self.__asset_position_dict={}
        self.__trading_time = pd.Series(pd.to_datetime(self.__portfolio_data.index.str[:18]))
        self.__portfolio_data.index = pd.to_datetime(self.__portfolio_data.index)

        for i in self.__portfolio_data.iloc[:]:
            fff=self.__portfolio_data.loc[:, i]
            self.__asset_position_dict[i]=self.__portfolio_data.loc[:,i]

        self.__assets_id_list=list(self.__portfolio_data.columns[0:])



    def get_historical_position_by_id(self,id:str):
        """
           get historical positions of the stock by the stock id
           :arg
           -------
               id: str
                stock id

           :return
           -------
               None
        """
        assert type(self.__assets_id_list)==list
        assert self.__assets_id_list.count(id)==1
        historical_pos=self.__asset_position_dict[id]
        return historical_pos

    def get_historical_position_by_id_time(self,id:str,start:datetime.datetime,end:datetime.datetime):
        """
              get historical positions of the stock by the stock id and time interval [start,end]
              :arg
              -------
                id: str
                   stock id

                start:datetime.datetime

                end:datetime.datetime

              :return
              -------
                  None
           """

        assert self.__assets_id_list.count(id) == 1
        historical_pos = self.__asset_position_dict[id]
        ddd=historical_pos.index[0]
        historical_pos=historical_pos[historical_pos.index>=start]
        historical_pos = historical_pos[historical_pos.index<=end]

        assert  historical_pos.shape[0]>0,"can not find the positions data between the corresponding time interval"

        return historical_pos

    def get_assets_id_list(self):
        """
           get assets_id_list
           :arg
           -------
              None

           :return
           -------
               assets_id_list ;list
               a list of stock id
        """
        return self.__assets_id_list

    def get_trading_time(self):
        """
               get a list of the trading time for each position
               :arg
               -------
                 id: str
                    stock id

                 start:datetime.datetime

                 end:datetime.datetime

               :return
               -------
                   None
            """
        return self.__trading_time


if __name__ == '__main__':
    file_path="portfolios_long.csv"
    ppp=data_handler_portfolio(csv_file=file_path)
    id_name="ABT"
    ddd=ppp.get_historical_position_by_id(id_name)
    print("asdsd")