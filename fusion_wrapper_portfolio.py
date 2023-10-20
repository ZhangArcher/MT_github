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
    __stock_ids=None
    __fusion_dict=None

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


        self.__wrapper_long_term = gp_wrapper_portfolio.gp_wrapper_portfolio(data_file_portfolio=portfolio_long_term, data_file_return=return_long_term)
        self.__wrapper_short_term = gp_wrapper_portfolio.gp_wrapper_portfolio(data_file_portfolio=portfolio_short_term, data_file_return=return_short_term)

        self.__trading_time_short_term=self.__wrapper_short_term.get_trading_time()
        self.__trading_time_long_term = self.__wrapper_long_term.get_trading_time()

        self.__short_term_reutrn=self.__wrapper_short_term.get_total_historical_return()
        self.__long_term_reutrn = self.__wrapper_long_term.get_total_historical_return()

        self.__stock_ids=self.__wrapper_long_term.get_stock_ids()
        self.__fusion_dict={}

        for id in self.__stock_ids:
            self.__fusion_dict[id]=fusion.fusion(fusion_type="LinearRegression",data_type="portfolio")

        self.__loss_score_list=[]

        self.__correction_term=0




    def fusion_next_portfolio_single_short_term_data(self, start: datetime.datetime, end: datetime.datetime,forward_window:int):
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
                   short-term GP is generated from the data set between [short_term_start,short_term_end+forward_window].

                :arg
                -------
                       start:datetime.datetime

                       end:datetime.datetime

                       forward_window:int
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
        #end_short_term = self.__wrapper_short_term.add_time(target_time=end, add_length=self.__forward_window)
        end_s = self.__wrapper_short_term.find_matched_time_with_increment(begin_time=end, time_increment=forward_window)



        actual_portfolio_l,actual_profit_l,pred_portfolio_l,pred_profit_l,loss_score_l  = \
            self.gp_predict_long_term(start_long_term=start_l,end_long_term=end_l)

        actual_portfolio_s, actual_profit_s, pred_portfolio_s, pred_profit_s, loss_score_s = \
            self.gp_predict_short_term(start_short_term=start_s,end_short_term=end_s)
        fusion_portfolio_list=[]
        for id in self.__stock_ids:

            fusion_portfolio, error_fusion, error_long, error_short=self.__fusion_dict[id].fusion_single_short(
            pred_s=pred_portfolio_s[id],pred_l=pred_portfolio_l[id], actual_l=actual_portfolio_l[id],
            actual_s=actual_portfolio_s[id])

            df_fusion=pandas.DataFrame(fusion_portfolio,columns=[id])

            fusion_portfolio_list.append(df_fusion)
        #  loss_score = self.get_loss_function_result(target_profit_long_term=actual_portfolio_l, fusion_profit=fusion_profit)
        df_fusion_porfolio=pd.concat(fusion_portfolio_list,axis=1)
        df_fusion_porfolio_rebalance=util.rebalance_portfolio(df_fusion_porfolio)
        df_fusion_porfolio_rebalance.reindex(actual_profit_s.index)


        next_long_term_timepoint= self.__wrapper_long_term.find_matched_time_with_increment(begin_time=end_l, time_increment=1)

        fusion_profit=util.compute_profit_from_short_to_long(portfolios=df_fusion_porfolio,returns_short_term=self.__short_term_reutrn,timestamp_long_term=next_long_term_timepoint)
        pred_profit_s=util.compute_profit_from_short_to_long(portfolios=pred_portfolio_s,returns_short_term=self.__short_term_reutrn,timestamp_long_term=next_long_term_timepoint)
        actual_profit_s=util.compute_profit_from_short_to_long(portfolios=actual_portfolio_s,returns_short_term=self.__short_term_reutrn,timestamp_long_term=next_long_term_timepoint)

        print("asdssd")

        # RESET TIME INDEX based on actual_profit_l
        base_timeindex=actual_profit_l.index
        fusion_profit=self.reset_timeindex(df=fusion_profit,timeindex=base_timeindex)
        pred_profit_l = self.reset_timeindex(df=pred_profit_l, timeindex=base_timeindex)
        actual_profit_s = self.reset_timeindex(df=actual_profit_s, timeindex=base_timeindex)
        pred_profit_s = self.reset_timeindex(df=pred_profit_s, timeindex=base_timeindex)




        error_fusion,error_pred_profit_l,error_actual_profit_s,error_pred_profit_s=self.compute_loss(fusion_profit=fusion_profit,
        pred_profit_s=pred_profit_s,actual_profit_s=actual_profit_s,actual_profit_l=actual_profit_l,pred_profit_l=pred_profit_l)





        return fusion_profit,actual_profit_l,pred_profit_l,actual_profit_s,pred_profit_s,error_fusion,error_pred_profit_l,error_actual_profit_s,error_pred_profit_s

    def fusion_next_portfolio_mult_short_term(self, start: datetime.datetime, end: datetime.datetime,forward_window:int):
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
                   short-term GP is generated from the data set between [short_term_start,short_term_end+forward_window].

                :arg
                -------
                       start:datetime.datetime

                       end:datetime.datetime

                       forward_window:int
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
        #end_short_term = self.__wrapper_short_term.add_time(target_time=end, add_length=self.__forward_window)
        end_s = self.__wrapper_short_term.find_matched_time_with_increment(begin_time=end, time_increment=forward_window)



        # actual_portfolio_l, actual_portfolio_s, actual_profit_l, actual_profit_s, pred_portfolio_l, pred_portfolio_s  = \
        #     self.gp_predict(start_long_term=start_l,end_long_term=end_l,start_short_term=start_s,end_short_term=end_s)

        actual_portfolio_l,actual_profit_l,pred_portfolio_l,pred_profit_l,loss_score_l=self.get_long_term_data(start_long_term=start_l,end_long_term=end_l)

        actual_portfolio_s_list,actual_profit_s_list,pred_portfolio_s_list=self.get_mult_short_term_data(start_short_term=start_l,end_short_term=end_l,forward_window=forward_window)
        for id in self.__stock_ids:
            pred_s_list, actual_s_list, pred_l, actual_l=self.reorgnize_data(actual_portfolio_l=actual_portfolio_l,pred_portfolio_l=pred_portfolio_l,
                                            actual_portfolio_s_list=actual_portfolio_s_list,pred_portfolio_s_list=pred_portfolio_s_list,id=id)


            self.__fusion_dict[id].fusion_price_linear(pred_s_list=pred_s_list,pred_l=pred_l,
                actual_l=actual_l,actual_s_list=actual_s_list,return_l=self.__long_term_reutrn,return_s=self.__short_term_reutrn)





        # fusion_portfolio,fusion_profit,fusion_error,pred_long_error = self.fusion_portfolio_mult_short_one_long(
        #     pred_s_list=pred_portfolio_s_list,pred_l=pred_portfolio_l,actual_l=actual_portfolio_l,actual_s_list=actual_profit_s_list,return_l=self.__long_term_reutrn,return_s=self.__short_term_reutrn)
        #


      #  loss_score = self.get_loss_function_result(target_profit_long_term=actual_portfolio_l, fusion_profit=fusion_profit)

        #return actual_portfolio_l,fusion_portfolio,actual_profit_l,fusion_profit
        return actual_portfolio_l, fusion_portfolio, actual_profit_l,fusion_profit,fusion_error,pred_long_error




    def reorgnize_data(self,actual_portfolio_l,pred_portfolio_l,actual_portfolio_s_list,pred_portfolio_s_list,id):
        pred_s_list = []

        for ele in pred_portfolio_s_list:
            pred_s_list.append(ele[id].values)

        actual_s_list=[]

        for ele in actual_portfolio_s_list:
            actual_s_list.append(ele[id].values)


        pred_l = [pred_portfolio_l[id].values]
        actual_l = [actual_portfolio_l[id].values]

        return pred_s_list,actual_s_list,pred_l,actual_l




    def fusion_next_portfolio_cumulative(self,start:datetime.datetime,predict_begin:datetime.datetime,end:datetime.datetime, forward_window=0):

        end_l=self.__wrapper_long_term.find_matched_time_with_increment(begin_time=predict_begin,time_increment=0)

        error_fusion_list=[]
        error_pred_profit_l_list=[]
        error_actual_profit_s_list=[]
        error_pred_profit_s_list=[]
        time_index=[]
        while(end_l<end):
            print("end :"+str(end))
            print("end_long_term :" + str(end_l))



           # target_portfolio_long_term,target_portfolio_short_term,target_profit_long_term,target_profit_short_term,pred_portfolio_long_term,pred_portfolio_short_term,cur_returns_long_term=self.fusion_next_portfolio( start_long_term=start,end_long_term=end_l,start_short_term=start,end_short_term=end_s)
            fusion_profit, actual_profit_l, pred_profit_l, actual_profit_s, pred_profit_s, error_fusion, error_pred_profit_l, error_actual_profit_s, error_pred_profit_s\
                =self.fusion_next_portfolio_single_short_term_data(start=start, end=end_l,forward_window=forward_window)
          #  end_l = self.__wrapper_long_term.find_matched_time_with_increment(begin_time=end_l, time_increment=1)
            error_fusion_list.append(error_fusion.values[0])
            error_pred_profit_l_list.append(error_pred_profit_l.values[0])
            error_actual_profit_s_list.append(error_actual_profit_s.values[0])
            error_pred_profit_s_list.append(error_pred_profit_s.values[0])
            time_index.append(end_l)
            end_l = self.__wrapper_long_term.find_matched_time_with_increment(begin_time=end_l, time_increment=1)

        df=pd.DataFrame(data={"error_fusion": error_fusion_list, "error_pred_profit_l": error_pred_profit_l_list,
            "error_actual_profit_s":error_actual_profit_s_list,"error_pred_profit_s":error_pred_profit_s_list},index=[time_index])
        print("as")
        return df

    def mapping_short_to_long(self,Y_mean_short_term:np.ndarray):
        #Y_mean_short_to_long=Y_mean_short_term+self.__correction_term
        Y_mean_short_to_long = Y_mean_short_term + 0
        return Y_mean_short_to_long

    def get_loss_function_result(self,fusion_profit:pd.DataFrame,target_profit_long_term:pd.DataFrame,long_term_profit:pd.DataFrame):
        fusion_loss_score=np.mean(fusion_profit.loc[:,"Profit"]-target_profit_long_term.loc[:,"Profit"])
        pred_long_term_loss_score = np.mean(long_term_profit.loc[:, "Profit"] - target_profit_long_term.loc[:, "Profit"])
        return fusion_loss_score,pred_long_term_loss_score


    def reset_timeindex(self,df:pandas.DataFrame,timeindex):
        df.index=timeindex
        return df


    def compute_loss(self,fusion_profit,actual_profit_l,pred_profit_l,actual_profit_s,pred_profit_s):
        base=actual_profit_l.loc[:,"Profit"]

        error_fusion=self.loss_function(base=base,target=fusion_profit.loc[:,"Profit"])
        error_pred_profit_l = self.loss_function(base=base, target=pred_profit_l.loc[:, "Profit"])
        error_actual_profit_s = self.loss_function(base=base, target=actual_profit_s.loc[:, "Profit"])
        error_pred_profit_s = self.loss_function(base=base, target=pred_profit_s.loc[:, "Profit"])

        return error_fusion,error_pred_profit_l,error_actual_profit_s,error_pred_profit_s


    def loss_function(self,base,target):
        return target-base



    def get_long_term_data(self, start_long_term: datetime.datetime, end_long_term: datetime.datetime):
        actual_portfolio_l, actual_profit_l, pred_portfolio_l, pred_profit_l, loss_score_l=self.gp_predict_long_term(start_long_term=start_long_term,end_long_term=end_long_term)
        return actual_portfolio_l,actual_profit_l,pred_portfolio_l,pred_profit_l,loss_score_l

    def get_mult_short_term_data(self, start_short_term: datetime.datetime, end_short_term: datetime.datetime,forward_window:int):

        actual_portfolio_list=[]
        actual_profit_list=[]
        pred_portfolio_list=[]
        pred_profit_list=[]
        loss_score_list=[]
        for i in range(0,forward_window):
            end_short_term = self.__wrapper_short_term.find_matched_time_with_increment(begin_time=end_short_term,
                                                                                       time_increment=1)

            #actual_portfolio_s, actual_profit_s, pred_portfolio_s, pred_profit_s, loss_score_s
            actual_portfolio_s, actual_profit_s, pred_portfolio_s, pred_profit_s, loss_score_s=self.gp_predict_short_term(start_short_term=start_short_term, end_short_term=end_short_term)

            actual_portfolio_list.append(actual_portfolio_s)
            actual_profit_list.append(actual_profit_s)
            pred_portfolio_list.append(pred_portfolio_s)
            pred_profit_list.append(pred_profit_s)
            loss_score_list.append(loss_score_s)



        return actual_portfolio_list,actual_profit_list,pred_portfolio_list




    def gp_predict_long_term(self,start_long_term:datetime.datetime,end_long_term:datetime.datetime):

        actual_portfolio_l,actual_profit_l,pred_portfolio_l,pred_profit_l,loss_score_l=self.__wrapper_long_term.predict_V1(start_time=start_long_term, end_time= end_long_term,pred_length=1)

        return actual_portfolio_l,actual_profit_l,pred_portfolio_l,pred_profit_l,loss_score_l

    def gp_predict_short_term(self,start_short_term: datetime.datetime, end_short_term: datetime.datetime):
        actual_portfolio_s, actual_profit_s, pred_portfolio_s, pred_profit_s, loss_score_s = self.__wrapper_short_term.predict_V1(start_time=start_short_term, end_time=end_short_term, pred_length=1)

        return actual_portfolio_s, actual_profit_s, pred_portfolio_s, pred_profit_s, loss_score_s


if __name__ == '__main__':

   portfolio_s = "demo_data/historical_portfolios_short.csv"
   return_s = "demo_data/historical_return_short.csv"

   portfolio_l= "demo_data/historical_portfolios_long.csv"
   return_l = "demo_data/historical_return_long.csv"

   #data_file_month="data/long_1month/long_1month_term_MSFT.csv"
   start_time="2014-09-01 00:00:00"
   predict_begin = "2016-02-01 00:00:00"
   end_time="2017-05-01 00:00:00"
   start_time = util.convert_time_into_datetime(time=start_time)
   end_time = util.convert_time_into_datetime(time=end_time)
   predict_begin = util.convert_time_into_datetime(time=predict_begin)
   df_score_list=[]

   error_fusion_list=[]
   error_pred_profit_l_list=[]
   error_actual_profit_s_list=[]
   error_pred_profit_s_list=[]
   # forwarding_window=range(0,20,1)
   # for i in forwarding_window:
   #     fff=fusion_wrapper_portfolio(return_long_term=return_l,return_short_term=return_s,
   #              portfolio_long_term=portfolio_l,portfolio_short_term=portfolio_s)
   #
   #     error_fusion, error_pred_profit_l, error_actual_profit_s, error_pred_profit_s=fff.fusion_next_portfolio_single_short_term_data(start=start_time, end=end_time, forward_window=i)
   #     print("forward_window:",i)
   #     print("error_fusion:",error_fusion)
   #     print("error_pred_profit_l:", error_pred_profit_l)
   #     print("error_actual_profit_s:", error_actual_profit_s)
   #     print("error_pred_profit_s:", error_pred_profit_s)
   #     error_fusion_list.append(error_fusion.iloc[0])
   #     error_pred_profit_l_list.append(error_pred_profit_l.iloc[0])
   #     error_actual_profit_s_list.append(error_actual_profit_s.iloc[0])
   #     error_pred_profit_s_list.append(error_pred_profit_s.iloc[0])
   #
   # fig, ax = plt.subplots()
   #
   # ax.plot(forwarding_window, error_fusion_list, label="error_fusion")
   # ax.plot(forwarding_window, error_pred_profit_l_list, label="error_pred_profit_l ")
   # ax.plot(forwarding_window, error_actual_profit_s_list, label="error_actual_profit_s ")
   # ax.plot(forwarding_window, error_pred_profit_s_list, label="error_pred_profit_s ")
   # ax.set_xlabel('fitting_windows')
   # ax.set_ylabel('score')
   #      # plt.yscale("log")
   # ax.legend()
   # plt.savefig('fusion_.png')
   # plt.show()

   # actual_portfolio_l, fusion_portfolio, actual_profit_l, fusion_profit, fusion_error, pred_long_error=fff.fusion_next_portfolio_mult_short_term(start=start_time, end=end_time, forward_window=5)
   forward_windows=range(0,20,1)
   for i in forward_windows:
       fff=fusion_wrapper_portfolio(return_long_term=return_l,return_short_term=return_s,
                     portfolio_long_term=portfolio_l,portfolio_short_term=portfolio_s)
       df = fff.fusion_next_portfolio_cumulative(start=start_time,predict_begin=predict_begin, end=end_time,forward_window=0)
       file_name="test_result_record/" +str(i)+"_error_result.xlsx"
       df.to_excel(file_name, sheet_name="Sheet1")
   # file_name_excel = "test_result/Experiment6_portfolio_fusion_result.xlsx"
   # file_name_csv = "test_result/Experiment6_portfolio_fusion_result.csv"
   # df_score.to_excel(file_name_excel, sheet_name="Sheet1")
   # df_score.to_csv(file_name_csv,date_format='%Y-%m-%d %X')


