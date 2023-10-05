
import numpy as np
import pandas
import pandas as pd
import pypfopt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import data_handler_price
import logging
import datetime



def convert_time_into_datetime(time: str):
    """
      transform time (string form) into a datetime
      input:
          ime: str
              start time


      output:
          time: datetime.datetime


      """
    if (type(time) != datetime.datetime):
        time = (datetime.datetime.strptime(time, '%Y-%m-%d %X'))

    return time


def find_matched_time_with_increment(trading_time:pandas.DataFrame,
                                     begin_time:datetime.datetime,time_increment:int):
    """
       find the corresponding time points after time_increment from trading_time
    :arg
    -------
           begin_time:datetime.datetime
           time_increment:int
   :return
    -------
          matched_times:pandas.Dataframe
    """
    # get time list from data_handler
    trading_time=trading_time
    trading_time=trading_time.sort_values()
    #list all prospective time points from target_time
    next_possible_days=np.where(trading_time>=begin_time)
    new_end_time_index=next_possible_days[0][time_increment]
    new_end_time=trading_time.iloc[new_end_time_index]

    indexs=next_possible_days[0][time_increment]
    matched_time=pd.to_datetime(trading_time.iloc[indexs])


    return matched_time


def calculate_historical_returns(historical_prices: pd.DataFrame):
    """
    generate historical returns from the historical prices
    :arg
    -------
        historical_prices:pd.DataFrame
            The historical stock prices
    :return
    -------
        return_from_price:pd.dataframe
            The historical stock return
    """
    return_from_price = pypfopt.expected_returns.returns_from_prices(historical_prices)
    return return_from_price


def compute_profit(portfolios:pd.DataFrame,returns:pd.DataFrame):
    """
    compute portfolios profit based the corresponding returns
    :arg
    -------
        historical_prices:pd.DataFrame
                The historical stock prices
    :return
    -------
        return_from_price:pd.dataframe
                The historical stock return
    """
    returns = match_return_portfolio(historical_return=returns, portfolios=portfolios)
    returns.sort_index(ascending=True)
    portfolios.sort_index(ascending=True)
    profit=portfolios.mul(returns,axis=0)
    profit["Profit"]=profit.sum(axis=1,numeric_only=True)

    return profit


def match_return_portfolio(historical_return:pd.DataFrame,portfolios:pd.DataFrame):

    index_list_portf=portfolios.index
    historical_return=historical_return.filter(items=index_list_portf,axis=0)
    assert len(historical_return) == len(portfolios)

    return historical_return