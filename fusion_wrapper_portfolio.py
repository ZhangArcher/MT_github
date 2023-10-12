import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas
import pandas as pd

import data_handler_portfolio
import fusion
import gp_wrapper_portfolio

import datetime
import logging

import util


class fusion_wrapper_portfolio:
    """


     """


    __wrapper_long_term = None
    __wrapper_short_term = None

    __loss_score_list=None
    __correction_term=None

    __long_term_reutrn=None
    __short_term_reutrn=None

    __trading_time_long_term = None
    __trading_time_short_term = None

    __fusion=fusion.fusion()

    def __init__(self,return_long_term: str,portfolio_long_term:str,
                 return_short_term: str,portfolio_short_term:str):
        """
        To initialize  class
        to set Gaussian Process Regressor
        :arg
        -------
              return_long_term: str
                  the path of the long-term historical return data set
              portfolio_long_term: str
                  the path of the long-term historical portfolio data set
              return_short_term: str
                  the path of the short-term historical return data set
              portfolio_short_term: str
                  the path of the short-term historical portfolio data set
        :return
        -------
              None
        """


        self.__wrapper_long_term = gp_wrapper_portfolio.gp_wrapper_portfolio(data_file_portfolio=portfolio_long_term,data_file_return=return_long_term)
        self.__wrapper_short_term = gp_wrapper_portfolio.gp_wrapper_portfolio(data_file_portfolio=portfolio_short_term,data_file_return=return_short_term)

        self.__trading_time_short_term=self.__wrapper_short_term.get_trading_time()
        self.__trading_time_long_term = self.__wrapper_long_term.get_trading_time()

        self.__short_term_reutrn=self.__wrapper_short_term.get_total_historical_return()
        self.__long_term_reutrn = self.__wrapper_long_term.get_total_historical_return()

        self.__loss_score_list=[]

        self.__correction_term=0




    def fusion_next_portfolio(self, start: datetime.datetime, end: datetime.datetime,excess_time:int):
        """
                   it is going to predict the  long-term portfolio at time point (long_term_end+1) using long-term GP, short-term GP and  fusion algorithm.

                   important:
                    (long_term_end+1) is based on long-term unit time.
                   (short_term_end+1) is based on short-term unit time.
                   hence , (short_term_end+1) != (long_term_end+1)

                   long_term_end=end
                   long_term_start=start
                   short_term_end=end
                   short_term_start=start
                   long-term GP is generated from the data set between [long_term_start,long_term_end].
                   short-term GP is generated from the data set between [short_term_start,short_term_end+excess_time].

                :arg
                -------
                       start:datetime.datetime

                       end:datetime.datetime

                       excess_time:int
               :return
                -------
                    actual_portfolio_l:pd.Dataframe
                        The actual long-term portfolios
                    fusion_portfolio:pd.Dataframe
                        The fused long-term portfolios
                    actual_profit_l:pd.Dataframe
                        The actual long-term profits
                    fusion_profit:pd.Dataframe
                        The fused long-term profits
                    fusion_error:pd.Dataframe
                        The fusion error between the fusion_profit and actual_profit_l
                    pred_long_error:pd.Dataframe
                        The predicted error between the pred_profit_l and actual_profit_l
                """

        start_l = start
        start_s = start
        end_l = end
        #end_short_term = self.__wrapper_short_term.add_time(target_time=end, add_length=self.__excess_time)
        end_s = self.__wrapper_short_term.find_matched_time_with_increment(begin_time=end, time_increment=excess_time)



        actual_portfolio_l, actual_portfolio_s, actual_profit_l, actual_profit_s, pred_portfolio_l, pred_portfolio_s  = \
            self.gp_predict(start_long_term=start_l,end_long_term=end_l,start_short_term=start_s,end_short_term=end_s)



        fusion_portfolio,fusion_profit,fusion_error,pred_long_error = self.__fusion.fusion_portfolio_long_term(
            pred_s=pred_portfolio_s,pred_l=pred_portfolio_l,actual_l=actual_portfolio_l,return_l=self.__long_term_reutrn)



      #  loss_score = self.get_loss_function_result(target_profit_long_term=actual_portfolio_l, fusion_profit=fusion_profit)

        #return actual_portfolio_l,fusion_portfolio,actual_profit_l,fusion_profit
        return actual_portfolio_l, fusion_portfolio, actual_profit_l,fusion_profit,fusion_error,pred_long_error








    def fusion_next_portfolio_cumulative(self,start:datetime.datetime,predict_begin:datetime.datetime,end:datetime.datetime, excess_time=0):

        end_l=self.__wrapper_long_term.find_matched_time_with_increment(begin_time=predict_begin,time_increment=1)

        score_list=[]
        score_list2=[]
        time_index=[]
        while(end_l<end):
            print("end :"+str(end))
            print("end_long_term :" + str(end_l))



           # target_portfolio_long_term,target_portfolio_short_term,target_profit_long_term,target_profit_short_term,pred_portfolio_long_term,pred_portfolio_short_term,cur_returns_long_term=self.fusion_next_portfolio( start_long_term=start,end_long_term=end_l,start_short_term=start,end_short_term=end_s)
            actual_portfolio_l, fusion_portfolio, actual_profit_l, fusion_profit, fusion_error, \
            pred_long_error=self.fusion_next_portfolio(start=end_l, end=end_l,excess_time=excess_time)
            end_l = self.__wrapper_long_term.find_matched_time_with_increment(begin_time=end_l, time_increment=1)
            score_list.append(fusion_error)
            #score_list.append(pd.DataFrame(data={"fusion_loss_score":fusion_loss_score,"pred_long_term_loss_score":pred_long_term_loss_score},index=[end_long_term]))
            score_list2.append(pred_long_error)
            time_index.append(end_l)

        df=pd.DataFrame(data={"fusion_loss_score": score_list, "pred_long_term_loss_score": score_list2},index=[time_index])

        return df

    def mapping_short_to_long(self,Y_mean_short_term:np.ndarray):
        #Y_mean_short_to_long=Y_mean_short_term+self.__correction_term
        Y_mean_short_to_long = Y_mean_short_term + 0
        return Y_mean_short_to_long

    def get_loss_function_result(self,fusion_profit:pd.DataFrame,target_profit_long_term:pd.DataFrame,long_term_profit:pd.DataFrame):
        fusion_loss_score=np.mean(fusion_profit.loc[:,"Profit"]-target_profit_long_term.loc[:,"Profit"])
        pred_long_term_loss_score = np.mean(long_term_profit.loc[:, "Profit"] - target_profit_long_term.loc[:, "Profit"])
        return fusion_loss_score,pred_long_term_loss_score

    def gp_predict(self,start_long_term:datetime.datetime,end_long_term:datetime.datetime,start_short_term:datetime.datetime,end_short_term:datetime.datetime):

        actual_portfolio_l,actual_profit_l,pred_portfolio_l,pred_profit_l,loss_score_l=self.__wrapper_long_term.predict(start_time=start_long_term, end_time= end_long_term,pred_length=1)
        actual_portfolio_s, actual_profit_s,pred_portfolio_s,pred_profit_s,loss_score_s=self.__wrapper_short_term.predict(start_time=start_short_term, end_time=end_short_term, pred_length=1)

        return actual_portfolio_l,actual_portfolio_s,actual_profit_l,actual_profit_s,pred_portfolio_l,pred_portfolio_s






if __name__ == '__main__':

   portfolio_s = "demo_data/historical_portfolios_short.csv"
   return_s = "demo_data/historical_return_short.csv"

   portfolio_l= "demo_data/historical_portfolios_long.csv"
   return_l = "demo_data/historical_return_long.csv"

   #data_file_month="data/long_1month/long_1month_term_MSFT.csv"
   start_time="2015-02-01 00:00:00"
   predict_begin = "2016-02-01 00:00:00"
   end_time="2017-02-01 00:00:00"
   start_time = util.convert_time_into_datetime(time=start_time)
   end_time = util.convert_time_into_datetime(time=end_time)
   predict_begin = util.convert_time_into_datetime(time=predict_begin)
   df_score_list=[]
   fff=fusion_wrapper_portfolio(return_long_term=return_l,return_short_term=return_s,
            portfolio_long_term=portfolio_l,portfolio_short_term=portfolio_s)

   #actual_portfolio_l, fusion_portfolio, actual_profit_l, fusion_profit, fusion_error, pred_long_error=fff.fusion_next_portfolio(start=start_time, end=end_time, excess_time=14)

   df = fff.fusion_next_portfolio_cumulative(start=start_time,predict_begin=predict_begin, end=end_time,excess_time=0)


   # file_name_excel = "test_result/Experiment6_portfolio_fusion_result.xlsx"
   # file_name_csv = "test_result/Experiment6_portfolio_fusion_result.csv"
   # df_score.to_excel(file_name_excel, sheet_name="Sheet1")
   # df_score.to_csv(file_name_csv,date_format='%Y-%m-%d %X')
   print("asasdsd")