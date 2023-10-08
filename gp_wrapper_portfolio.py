
import numpy as np
import pandas
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import GP
import data_handler_portfolio
import logging
import datetime
import util


class gp_wrapper_portfolio:
    """
    data wrapper class for different portfolio

     Attributes
     ----------
        __portfolios_data : portfolio_handler



     Methods

     -------
     """

    __portfolios_handler = None

    __historical_portofolios=None

    __historical_return=None

    __target_portfolios=None

    __pred_portfolios=None


    __stock_list=None
    __GPs_dict=None

    __prices_data = None
    __alpha = None
    __iterations = None
    __kernels = None
    #__gp = None
    __eval_score_list = None
    __trading_time = None

    def __init__(self, data_file_portfolio: str,data_file_return:str):
        """
          To initialize class and to set Gaussian Process Regressor
          input:
            data_file_portfolio: str
                the path of the portfolio data

            data_file_return:str
                the path of the asset return data
          output:
              None
          """

        self.__portfolios_handler = data_handler_portfolio.data_handler_portfolio(data_file_portfolio)
        self.__trading_time=self.__portfolios_handler.get_trading_time()
        self.__stock_list=self.__portfolios_handler.get_assets_id_list()

        self.__historical_return=pd.read_csv(data_file_return,index_col=0)

        self.__historical_return.index = pd.to_datetime(self.__historical_return.index)
        self.__historical_return=self.__historical_return.loc[:, self.__stock_list]

        self.__eval_score_list = []
        self.__GPs_dict={}

        for ele in self.__stock_list:
            self.__GPs_dict[ele]= GP.GaussianProcess

    def get_historical_returns_by_portfolios(self,current_historical_return:pd.DataFrame,portfolios:pd.DataFrame):

        """
         select a suitable time interval according  portfolios and return the corresponding historical return from current_historical_return
         :arg
         -------
           current_historical_return：pd.DataFrame
                The total historical return
           portfolios：pd.DataFrame
                the portfolios

         :return
         -------
            target_return:pd.DataFrame
                the corresponding historical return

        """


        index_list=portfolios.index
        start=index_list[0]
        end=index_list[-1]
        target_return=current_historical_return[current_historical_return.index>=start]
        target_return = target_return[target_return.index <= end]
        assert target_return.shape==portfolios.shape
        return target_return

    def predict(self, start_time: datetime.datetime, end_time: datetime.datetime, pred_length: int,with_loss=True):
        """
            predict the portfolios between [end_time+1,end_time+pred_length) using GPs
            GPs are generated from the data sets between [start_time,end_time]

         :arg
         -------
           start_time：datetime.datetime
                start time
           end_time：datetime.datetime
                end time
           pred_length: int
               The size of the time length what we want to predict (ignored time unit)
            with_loss:bool
               whether or not to compute loss
         :return
         -------
            actual_portfolio:pd.Dataframe

            actual_profit:pd.Dataframe

            pred_portfolio:pd.Dataframe

            pred_profit:pd.Dataframe

            self.__historical_return:pd.Dataframe

            loss_score:pd.Dataframe


        """

        assert start_time <= end_time,"error : start_time > end_time"
        start_time = self.find_matched_time_with_increment(begin_time=start_time, time_increment=0)
        end_time = self.find_matched_time_with_increment(begin_time=end_time, time_increment=0)

        pred_start_time = self.find_matched_time_with_increment(begin_time=end_time,time_increment=1)
        pred_end_time=self.find_matched_time_with_increment( begin_time=end_time,time_increment=pred_length)


        pred_time_index=self.__trading_time
        pred_time_index=pred_time_index[pred_time_index<=pred_end_time]
        pred_time_index=pred_time_index[pred_time_index>=pred_start_time]

        pred_portfolio_list=[]
        actual_portfolio_list=[]

        # predict the position for each stock
        for id in self.__stock_list:

            X, Y, X_pred, Y_pred_actual = self.get_historical_positions(id=id,start_time=start_time,
                                            end_time= end_time, pred_start_time=pred_start_time,pred_end_time=pred_end_time)
            assert  len(pred_time_index)==len(X_pred)
            self.__GPs_dict[id]=GP.GaussianProcess()
            self.__GPs_dict[id].fit(X, Y)

            assert len(X_pred) > 0
            Y_pred_mean, Y_cov = (self.__GPs_dict[id]).predict(X_pred)
            assert len(pred_time_index) == len(Y_pred_mean)
            pred_portfolio_list.append(pd.DataFrame({id:Y_pred_mean},index=pred_time_index))
            actual_portfolio_list.append(pd.DataFrame({id: Y_pred_actual}, index=pred_time_index))


        pred_portfolio=pd.concat(pred_portfolio_list,axis=1)
        #to normalize the portfolio
        pred_portfolio=util.rebalance_portfolio(portfolio=pred_portfolio)
        actual_portfolio=pd.concat(actual_portfolio_list,axis=1)

        pred_profit=util.compute_profit(portfolios=pred_portfolio,returns=self.__historical_return)
        actual_profit = util.compute_profit(portfolios=actual_portfolio, returns=self.__historical_return)

        if(with_loss):
            loss_score=self.get_loss_function_result(pred_profit=pred_profit,target_profit=actual_profit)
            print("score is " + str(loss_score))
        else:
            loss_score=np.zeros(actual_profit.shape)
            print("score is " + str(loss_score))


        return  actual_portfolio,actual_profit,pred_portfolio,pred_profit,loss_score



    def find_matched_time_with_increment(self,begin_time:datetime.datetime,time_increment:int):
        """
           find the corresponding time points after time_increment from trading_time
        :arg
        -------
               trading_time: DataFrame
               begin_time:datetime.datetime
               time_increment:int
       :return
        -------
              matched_times:pandas.Dataframe
        """

        return util.find_matched_time_with_increment(trading_time=self.__trading_time,
                                     begin_time=begin_time,time_increment=time_increment)

    def find_all_matched_times_by_length(self,trading_time: pandas.DataFrame, begin_time: datetime.datetime,
                                             time_increment: int):
        """
           find all corresponding time points within a time_increment  from trading times
        :arg
        -------
               begin_time:datetime.datetime
               time_increment:int
       :return
        -------
              matched_times:pandas.Dataframe
        """
        # get time list from data_handler

        return util.find_all_matched_times_by_length(trading_time=trading_time,begin_time=begin_time,time_length=time_increment)


    def get_loss_function_result(self,pred_profit:pd.DataFrame,target_profit:pd.DataFrame):
        """
           compute the loss for the predicted result
        :arg
        -------
               begin_time:datetime.datetime
               time_increment:int
       :return
        -------
              loss_score:pandas.Dataframe
        """
        #loss_score = np.mean(pred_profit.loc[:, "Profit"] - target_profit.loc[:, "Profit"])
        loss_score = (pred_profit.loc[:, "Profit"] - target_profit.loc[:, "Profit"])
        return loss_score


    def get_historical_positions(self, id:str,start_time: datetime.datetime, end_time: datetime.datetime,
                     pred_start_time: datetime.datetime, pred_end_time: datetime.datetime):
        """
           get the historical_position between time interval [ start_time,end_time] and
           the historical_position between time interval [pred_start_time,pred_end_time]
         :arg
         -------
               start_time:datetime.datetime

               end_time:datetime.datetime

               pred_start_time:datetime.datetime

               pred_end_time:datetime.datetime
        :return
         -------
              X:np.array  2D
                [[year,data_id],.......]

              Y:np.array 1D
                [price,....]

              X_pred:np.array 2D
                [[year,data_id],.......]

              Y_pred_actual:np.array  1D
                [price,....]
            """
        assert start_time <= end_time
        X = []
        Y = []
        his_position_data = self.__portfolios_handler.get_historical_position_by_id_time(id=id,start=start_time,end=end_time)
        his_position_data.sort_values()
        size_of_data = his_position_data.shape[0]
        data_id = 1
        Y=list(his_position_data)
        for i in range(0, size_of_data):
            X.append([1, int(data_id)])
            data_id=data_id+1

        X = np.array(X)
        Y = np.array(Y)

        assert pred_start_time <= pred_end_time
        X_pred = []
        # data_id=0

        target_position_data = self.__portfolios_handler.get_historical_position_by_id_time(id=id, start=pred_start_time,end=pred_end_time)
        target_position_data.sort_values()

        Y_pred_actual=list(target_position_data)
        size_of_pred_data = target_position_data.shape[0]

        for i in range(0, size_of_pred_data):
            X_pred.append([1, int(data_id)])
            data_id = data_id + 1

        X_pred = np.array(X_pred)
        Y_pred_actual = np.array(Y_pred_actual)

        return X, Y, X_pred, Y_pred_actual


    def get_trading_time(self):
        assert len(self.__trading_time)>0
        return self.__trading_time


    def predict_multi(self,start_time: datetime.datetime, end_time: datetime.datetime,
                           fitting_windows:int):
        """

               predict the target portfolio with a fixed time windows multiple
               For example:
               iteration 1 :
                 predict the portfolio at timepoint (start_time+fitting_length) using GP
                GP is generated from the data set between [start_time,start_time+fitting_length-1]

                iteration 2:
                  predict the portfolio at timepoint (start_time+fitting_length+1) using a GP.
                This GP is generated/fitted by the data set between  [start_time+1,start_time+fitting_length]

                iteration ....

                iteration n:
            predict the portfolio at timepoint (start_time+fitting_length+n-1) using a GP.
                This GP is generated/fitted by the data set between  [start_time+n-1,start_time+fitting_length+n-2]

                :arg
                -------
                   start_time：str
                        start time
                   end_time：str
                        end time
                   fitting_length: int
                       The size of the time length for the training (ignored time unit)

               :return
                -------
                    df_mean: pandas.Dataframe
                    the corresponding mean  of the predicted GP

                    df_var: pandas.Dataframe
                    the corresponding cov  of the predicted GP

                    df_score: pandas.Dataframe
                    the score of loss function

                """

        assert start_time < end_time
        assert fitting_windows > 0

        # get the cooresponding time points
        temp_start_time = self.find_matched_time_with_increment(begin_time=start_time, time_increment=0)
        temp_end_time = self.find_matched_time_with_increment(begin_time=temp_start_time,
                                                              time_increment=fitting_windows - 1)
        loss_score_list=[]
        time_index=[]
        mean_list=[]

        while(temp_end_time<=end_time):

            actual_portfolio, actual_profit, pred_portfolio, pred_profit, loss_score = self.predict(start_time=temp_start_time,
                                                                      end_time=temp_end_time,pred_length=1)


            time_index.append(actual_portfolio.index[0])
            loss_score_list.append(loss_score)
            mean_list.append(actual_profit)



            temp_start_time=self.find_matched_time_with_increment(begin_time=temp_start_time,time_increment=1)
            temp_end_time=self.find_matched_time_with_increment(begin_time=temp_end_time,time_increment=1)

        # collect the result
        #df_score = pd.DataFrame(data=loss_score_list, index=time_index)

       # df_mean = pd.DataFrame(data=mean_list, index=time_index)


        return mean_list, loss_score_list

    def get_total_historical_return(self):
        return self.__historical_return

if __name__ == '__main__':
    #data_file_portfolio="portfolios_short_1day.csv"
    #data_file_return="historical_return_short_1day.csv"
    data_file_portfolio="portfolios_long.csv"
    data_file_return="historical_return_long.csv"


    start_time = "2016-01-22 00:00:00"
    end_time = "2018-05-22 00:00:00"

    start=util.convert_time_into_datetime(time=start_time)
    end = util.convert_time_into_datetime(time=end_time)

    dpp = gp_wrapper_portfolio(data_file_portfolio=data_file_portfolio, data_file_return=data_file_return)
    #target_portfolio,target_profit,pred_portfolio,pred_profit,loss_score=dpp.predict(start_time=start,end_time=end, pred_length=1,with_loss=False)
    df_mean, df_score = dpp.predict_multi(start_time=start, end_time=end,fitting_windows=16)
    #dpp.predict_by_time_interval(start_time=start, end_time=end,pred_length=3,add_correction_term=True)



