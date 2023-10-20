
import numpy as np
import pandas
import pandas as pd
import pypfopt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import math
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
       In other words , it is to find (begin_time+time_increment)

                   for example : trading_time is daily price , begin_time=01.01.2020 , time_increment=3 ,
                            it will return 04.01.2020

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

def compute_profit_from_short_to_long(portfolios:pd.DataFrame,returns_short_term:pd.DataFrame,timestamp_long_term:datetime.datetime):
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

    timestamp_long_term=timestamp_long_term-datetime.timedelta(days=1)
    returns= match_return_portfolio_by_time_interval(returns_short_term,start_time=portfolios.index[0],end_time=timestamp_long_term)
   # returns=returns_short_term.loc(return_index_list)
    returns=returns.add(1)
    returns.sort_index(ascending=True)

    result=returns.cumprod(axis=0)
    final_return=result.iloc[-1]

    portfolios.sort_index(ascending=True)
    portfolios.index=[timestamp_long_term]
    profit=portfolios.mul(final_return)
    profit=profit-portfolios
    profit["Profit"]=profit.sum(axis=1,numeric_only=True)
    #profit["Profit"]= profit["Profit"]-1
    return profit


def computer_profit_by_time_interval(portfolios:pd.DataFrame,returns:pd.DataFrame,
                                     start_time:datetime.datetime,end_time:datetime.datetime):
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
    profit = portfolios.mul(returns, axis=0)
    profit["Profit"] = profit.sum(axis=1, numeric_only=True)

    return profit


def match_return_portfolio_by_time_interval(historical_return:pd.DataFrame,
                                     start_time:datetime.datetime,end_time:datetime.datetime):

    historical_return=historical_return.loc[start_time:end_time]


    return historical_return

def match_return_portfolio(historical_return:pd.DataFrame,portfolios:pd.DataFrame):

    index_list_portf=portfolios.index
    historical_return=historical_return.filter(items=index_list_portf,axis=0)
    assert len(historical_return) == len(portfolios)

    return historical_return

def find_all_matched_times_by_length(trading_time:pandas.DataFrame,begin_time:datetime.datetime,time_length:int):
    """
       find all corresponding future time points within a time_lendth  from trading times
       find begin_time+1 , begin_time+2.....begin_time+(time_length)

            for example : begin_time=02.2020 , time_length=3  ,trading_time=[01.2020,02.2020,03.2020,04.2020,05.2020,06.2020]
                            it will return [03.2020,04.2020,05.2020]
      (matched_times include begin_time)
    :arg
    -------
           begin_time:datetime.datetime
           time_increment:int
   :return
    -------
          matched_times:pandas.Dataframe
    """
    # get time list from data_handler
    assert time_length > 0, "time_length must be larger than 0"
    trading_time=trading_time.sort_values()
    #list all prospective time points from target_time
    next_possible_days=np.where(trading_time>begin_time)
    new_end_time_index=next_possible_days[0][time_length]
    new_end_time=trading_time.iloc[new_end_time_index]

    indexs=next_possible_days[0][0:time_length]
    matched_times=pd.to_datetime(trading_time.iloc[indexs])


    return matched_times


def rebalance_portfolio(portfolio:pd.DataFrame):
    """
       normalize the portfolios
       to ensure that the sum of all assets position  is 1 (100%) each time

     :arg
     -------
       portfolio：datetime.datetime
            The portfolio

     :return
     -------
        portfolio: datetime.datetime
        The normalized portfolio



    """
   # portfolio=portfolio.apply(rebalance_function,axis=1)

    before_porfolio=portfolio
    for ele in portfolio.index:
        ddd=portfolio.loc[ele]
        sum_row=np.sum(ddd)
        portfolio.loc[ele]=(portfolio.loc[ele]).div(sum_row)
    #print("assds")
    return portfolio


def compute_cumulative_profit(df_profit:pandas.DataFrame):
    profit_list=df_profit["Profit"].tolist()
    cum_pred_profit=math.prod(list(map(lambda x:x+1,profit_list)))
    return cum_pred_profit