import os

import pandas
import pypfopt
import pypfopt.expected_returns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from data_handler_price import data_handler_price
import gp_wrapper_prices
import pandas as pd
import datetime
import logging
import util

"""
Portfolios class about different assets allocations.
Portfolios class can load historical prices  by data handler ,
 generate historical return ，covariance matrix and optimized portfolio.

Portfolios class can generate and store  a stocks_prices, a corresponding historical_return , mean_historical_return and a covariance matrix according to a special time interval.
It can compute Expected Returns and generate a optimized portfolios according Mean-Variance Optimization , Risk Models and some customized object function.

Attributes
----------
 __data_set_path: str
    the path of the directory of the corresponding data sets

__stock_path_dict: dict
    A dictionary  , key is stock ID and values is the path of the file of the corresponding data set

__stock_handler: dict
    A dictionary  , key is stock ID and values is the data_handler_price of  the corresponding stocks.

__stocks_prices: pd.Dataframe
    The dataframe about the  prices of the different stocks at different time points.

__portfolios: pd.Dataframe
    The portfolios at different time points

__historical_return: pd.Dataframe
    The dataframe  about the return of the different stocks at different  periods.

__mu: pd.Dataframe
    The mean_historical_return

__covariance: pd.Dataframe
    The covariance 

Methods:
generate_stock_dict

generate_dataframe_by_stock_dict

generate_stock_handler

get_stock_name_from_file

generate_dataframe_by_time_interval_string

generate_dataframe_by_time_interval

calculate_historical_return

calculate_historical_portfolios

get_profit_by_portfolio_and_asset_return

convert_time_into_datetiem
-------
"""


class Portfolios:


    __data_set_path = None
    __stock_dict = None
    __stock_handler = None
    __time_type = None
    __start_time=None
    __end_time=None
    __current_historical_stocks_prices = None
    __current_historical_return = None
    __target_semideviation=1
    __trading_time=None


    __current_portfolios = None
    __mu = None
    __covariance = None

    __trading_time=None


    def __init__(self, data_set_path: str):
        """
        To initialize  class
        :arg
        -------
              data_set_path: str
                  the path of the folder that contains stock prices

          :return
        -------
              None
        """
        self.__data_set_path = data_set_path
        self.__stock_dict = self.generate_stock_dict(data_set_path=data_set_path)
        self.__stock_handler = self.generate_stock_handler(stock_dict=self.__stock_dict)

    def generate_stock_dict(self, data_set_path: str):
        """
        generate  stock_dict , a dictionary  ( key is stock ID and values is the path of
        the file of the corresponding data set)

        :arg
        -------
            data_set_path:str
                The path of the directory of data sets

        :return
        -------
            stock_dict: dict
                dic , key is stock id and values is the path of data set.

        """
        file_list = os.listdir(data_set_path)
        assert len(file_list) > 0
        stock_dict = {}
        for ele in file_list:
            name = self.get_stock_id_from_file(ele)
            path = os.path.join(data_set_path, ele)
            stock_dict[name] = path
        return stock_dict



    def generate_stock_handler(self, stock_dict: dict):
        """
        generate stock_handler according stock_dict.
        :arg
        -------
            stock_dict: dict

        :arg
        -------
            stock_handler: data_handler_price.data_handler_price
                a handler ,which can load and process data sets.

        """
        stock_handler_dict = {}
        for ele in stock_dict.keys():
            handler = data_handler_price(stock_dict[ele])
            stock_handler_dict[ele] = handler
        assert len(stock_handler_dict) == len(stock_dict)
        return stock_handler_dict


    def get_stock_id_from_file(self, file_name: str):
        """
        get the stock id from the file name of data set

        :arg
        -------
            file_name:str
            The file name of the data set
        :return
        -------
            id_name:str
            stock id
        """
        id_name = (file_name.split("."))[0]
        return id_name


    def get_stock_price_by_handler(self, stock_id: str, handler: data_handler_price,
                                   start: datetime.datetime, end: datetime.datetime):
        """
        get the stock price by data_handler with a stock id and a time interval

        :arg
        -------
            id_name:str
                stock id
            handler: data_handler_price.data_handler_price
                data_handler
            start: datetime
            end:datetime

        :return
        -------
            result_price: Dataframe
                The dataframe of stock prices

        """
        result_price = handler.get_price_data_by_time_intetval(start_time=start, end_time=end)
        result_price = result_price.rename(columns={"Close": stock_id})
        return result_price



    def generate_portfolios(self,stock_returns:pd.DataFrame,
                                               start: datetime.datetime, end: datetime.datetime):
        """
        generate portfolio from the current stock_returns
        :arg
        -------
            start:datetime
            end:datetime
        :return
        -------
            weights:dataframe
                The portfolio
            pf:dataframe
                The performance
        """

        stock_returns=stock_returns[stock_returns.index<end]
        stock_returns=stock_returns[stock_returns.index>start]
        mu = self.calculate_expected_return(stock_returns,is_last_hiostorical_return=True,is_mean_hiostorical_return=False)
        S = pypfopt.risk_models.CovarianceShrinkage(stock_returns,returns_data=True).ledoit_wolf()


        ef=pypfopt.efficient_frontier.EfficientSemivariance(returns=stock_returns,expected_returns=mu)
        ef.add_objective(pypfopt.objective_functions.L2_reg, gamma=2)
        weights = ef.efficient_risk(self.__target_semideviation)
        pf=ef.portfolio_performance(verbose=True)

        return weights,pf



    def calculate_expected_return(self, stock_returns: pd.DataFrame,is_last_hiostorical_return=True,is_mean_hiostorical_return=False):
        assert (is_last_hiostorical_return & is_mean_hiostorical_return)==False
        assert ((is_last_hiostorical_return or is_mean_hiostorical_return)==True)
        if(is_last_hiostorical_return):
            mu=self.calculate_last_hiostorical_return(stock_returns)
        if(is_mean_hiostorical_return):
            mu=self.calculate_mean_hiostorical_return(stock_returns)
        return mu

    def calculate_last_hiostorical_return(self, stock_returns: pd.DataFrame):
       # mu = pypfopt.expected_returns.mean_historical_return(df)
        mu=stock_returns.iloc[-1]

        return mu

    def calculate_mean_hiostorical_return(self, stock_returns: pd.DataFrame):
        mu = pypfopt.expected_returns.mean_historical_return(prices=stock_returns,returns_data=True)
        return mu

    def calculate_covariance(self, df):
        S = pypfopt.risk_models.CovarianceShrinkage(df).ledoit_wolf()
        return S

    #    mu = pyportofilo.mu(histoircal_return)
    #   covariance = pyportofilo.cov(histoircal_return)
    #    return mu, covariance

    def calculate_portfolios(self, mu, covariance):
        ef = pypfopt.efficient_frontier.EfficientFrontier(mu, covariance)
        weights = ef.max_sharpe()

        return weights

    #   assert len(historical_return) > 0
    #    historical_portfolio = pyportfoilo.ptf(histoircal_return)
    #      return historical_portfolio



    def get_stock_dict(self):
        return self.__stock_dict

    def is_stock_dict_similar(self, stock_dict_other:dict):
        my_stock_dict = self.__stock_dict
        other_keys = stock_dict_other.keys
        my_keys = my_stock_dict.keys


    def generate_historical_price(self,start:datetime.datetime,end:datetime.datetime):
        """
        get the historical prices table  according time_interval
        This historical prices table is about all stocks
        :arg
        -------
              start: datetime.datetime

              end: datetime.datetime

        :return
        -------
              stocks_prices:dataframe

        """

        assert  start<end
        assert len(self.__stock_handler) > 0

        df = None
        df_init = False
        stock_handler=self.__stock_handler
        initial_trading_time=True
        data_set_size=0

        for id in stock_handler.keys():

            target_handler = stock_handler[id]

            assert self.validate_data_set_time_interval(target_handler=target_handler,start=start,end=end)
            tmp_stock_price = self.get_stock_price_by_handler(stock_id=id, handler=target_handler, start=start, end=end)
            tmp_stock_price.sort_values("Date")
            tmp_stock_price = tmp_stock_price.drop(columns=["timestamp"])

            if (initial_trading_time):
                self.__trading_time = target_handler.get_trading_time()
                initial_trading_time = False
                data_set_size=len(tmp_stock_price)

            assert len(tmp_stock_price)==data_set_size

            if (df_init == False):
                df = tmp_stock_price
                df_init = True
            else:
                df = pd.merge(df, tmp_stock_price, on="Date")

        df["Date"] = pd.to_datetime(df["Date"])
        stocks_prices = df.set_index("Date")
        self.__current_historical_stocks_prices=stocks_prices
        self.__start_time=start
        self.__end_time=end
        return stocks_prices


    def validate_data_set_time_interval(self, target_handler:data_handler_price,
                                        start:datetime.datetime, end:datetime.datetime):
        """
        check whether or not data_handler can offer the data between this time interval
        :arg
        -------
        target_handler:data_handler_price

        start:datetime.datetime

        end:datetime.datetime

        :return
        -------
              result: bool

        """

        handler_trading_time_list=target_handler.get_trading_time()
        handler_trading_time_list=handler_trading_time_list.sort_values()
        handler_start_time=handler_trading_time_list.iloc[0]
        handler_end_time=handler_trading_time_list.iloc[-1]
        result=(start>handler_start_time)&(end<handler_end_time)
        #result2=(len(self.__trading_time)==len(handler_trading_time_list))
        return result





    def generate_historical_returns(self):
        """
            To generate the historical_returns from self.__current_historical_stocks_prices
         :arg
         -------
            None
        :return
         -------
                 his_return:pandas.Dataframe
                  The historical returns of all stocks
         """
        assert type(self.__current_historical_stocks_prices)==pandas.DataFrame , \
            "you need to load historical stocks_prices(method: get_historical_price_by_time_interval()"
        assert len(self.__current_historical_stocks_prices)>2 , "you need at least 2 " \
                                                                "different stock and their prices "
        his_return=util.calculate_historical_returns(self.__current_historical_stocks_prices)
        self.__current_historical_return=his_return
        return his_return


    def compute_portfolios_profit(self):
        """
        compute portfolios profit  based the historical returns
        The portfolios is self.__current_portfolios
        :arg
        -------
            None
        :return
        -------
        profit:pandas.Dataframe
            profit
        """

        assert len(self.__current_historical_return)>0
        assert len(self.__current_portfolios)>0
        profit=util.compute_profit(portfolios=self.__current_portfolios,returns=self.__current_historical_return)
        return profit


    def generate_historical_portfolios(self,start:datetime.datetime,
                                                        end:datetime.datetime,forward_length:int):
        """
        generate portfolios  based the historical returns
        :arg
        -------
        start:datetime.datetime

        end:datetime.datetime

        forward_length:int
            forward_length is the data set size used for the portfolios generation.

        :return
        -------
        portfolio:pandas.Dataframe
            portfolio


        """
        assert type(self.__current_historical_return) == pandas.DataFrame ,\
            "you need to initial self.__current_historical_return (method: generate_historical_returns() ) "
        assert start>self.__start_time , "we can not find historical_return before "+str(self.__start_time)
        assert end<self.__end_time , "we can not find historical_return after "+str(self.__end_time)

        tmp_start=start
        tmp_end=util.find_matched_time_with_increment(trading_time=self.__trading_time,
                                             begin_time=start,time_increment=forward_length)
        list_result=[]
        while(tmp_end<end):
            # find the corresponding time of the current portfolio
            next_time=(self.__current_historical_return[self.__current_historical_return.index>tmp_end]).sort_index().index[0]
            # generate the portfolio and the performance
            weights,pf=self.generate_portfolios(stock_returns=self.__current_historical_return,start=tmp_start,end=tmp_end)

            tmp_start = util.find_matched_time_with_increment(trading_time=self.__trading_time,
                                                              begin_time=tmp_start,time_increment=1)
            tmp_end= util.find_matched_time_with_increment(trading_time=self.__trading_time,
                                                           begin_time=tmp_end,time_increment=1)
            df=None
            df=pd.DataFrame(weights,index=[next_time])
            list_result.append(df)

        portfolio=pd.concat(list_result)
        self.__current_portfolios=portfolio
        return portfolio








if __name__ == '__main__':
    path_dataset = "data_set_price/long"
    start_time = "2014-01-01 00:00:00"
    end_time = "2019-01-01 00:00:00"

    ppp = Portfolios(path_dataset)
    #ppp.generate_dataframe_by_time_interval_string(start=start_time, end=end_time)
    start=util.convert_time_into_datetime(start_time)
    end=util.convert_time_into_datetime(end_time)


    start_time1 = "2015-01-01 00:00:00"
    end_time1 = "2018-01-01 00:00:00"
    start1 = util.convert_time_into_datetime(start_time1)
    end1 = util.convert_time_into_datetime(end_time1)
    #
    ppp.generate_historical_price(start=start,end=end)
    historical_return=ppp.generate_historical_returns()
    portfolio= ppp.generate_historical_portfolios(start=start1,end=end1,forward_length=10)
    profit = ppp.compute_portfolios_profit()

    print("asdsdsd")