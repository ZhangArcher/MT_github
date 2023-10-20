import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import data_handler_price
import gp_wrapper_prices
import fusion
import datetime
import logging
import util
import pandas as pd

class fusion_wrapper_price:
    """
    The data fusion class for 2 different term GP wrapper
    This class loads two different term data sets, long-term data set and short-term data set, using two data wrappers, long-term wrapper and short-term wrapper.
    Short-term data set is more fine than long-term data set.
    For example , long-term data set is about monthly prices and short-term data set is about daily prices.
    After receiving 28/29/30/31 daily prices , we can just get  one monthly price.
    Let us assumed there is only 30 days each month.
    Hence, we can know/predict the monthly price after receiving at most 30 daily prices.

    In this class , we try to find a mapping function which can map daily prices into monthly price.
    Mapping function makes it possible that we can know the monthly price after receiving at most 30-m daily prices.   (0<m<30)
    30-m is our _forward_window.

    (_forward_window is a scala , and we ignore the unit)

     Attributes
     ----------
        __company_name : string
            The stock name

        __company_wrapper_long_term : prices_wrapper_gp.Wrapper

        __company_wrapper_short_term : prices_wrapper_gp_my_version.Wrapper

        __time_type_long_term:  prices_handler.time_type
             The unit of long-term time , includes :  "Day" ,"Hour","Week","Month","Year"

        __time_type_short_term :data_handler_my_version.time_type
             The unit of short-term time , includes :  "Day" ,"Hour","Week","Month","Year"

        __forward_window:  int
        The size of the corresponding time units each year .
        For example, max_time_unit_length is 12 , while time_type is "Month"
        max_time_unit_length is 12 , while time_type is "Month"
        max_time_unit_length is 252 , while time_type is "Day" , (Because there are only 252 trading days each year)

        __loss_score_list: list
            A list of loss scores

        __correction_term: float
            The correction term



     Methods
        fusion_from_short_to_long
     -------
     """

    __company_name = None
    __company_wrapper_long_term = None
    __company_wrapper_short_term = None

    __loss_score_list=None
    __correction_term=None

    __fusion=None
    __trading_time_long_term = None
    __trading_time_short_term = None

    __GP_type=None


    def __init__(self, csv_path_long_term: str,csv_path_short_term: str,GP_type="GPR"):
        """
        To initialize  class
        to set Gaussian Process Regressor
        :arg
        -------
              csv_path_long_term: str
                  the path of the long-term data set
              csv_path_short_term: str
                  the path of the short-term data set
        :return
        -------
              None
        """
        self.__company_name = csv_path_long_term.split(".")[0]

        self.__GP_type=GP_type

        self.__company_wrapper_long_term = gp_wrapper_prices.gp_wrapper_price(csv_path_long_term,GP_type= self.__GP_type)
        self.__company_wrapper_short_term = gp_wrapper_prices.gp_wrapper_price(csv_path_short_term,GP_type= self.__GP_type)

        self.__trading_time_short_term=self.__company_wrapper_short_term.get_trading_time()
        self.__trading_time_long_term = self.__company_wrapper_long_term.get_trading_time()

        assert self.__trading_time_short_term.shape > self.__trading_time_long_term.shape

        self.__loss_score_list=[]
        self.__fusion=fusion.fusion()
        self.__correction_term=0





    def gp_predict_long_term(self,start_long_term:datetime.datetime,end_long_term:datetime.datetime):
        """
        To predict the long-term price with long-term data ,
        And it returns the results and the errors.
        
        :arg
        -------
            start_long_term:datetime.datetime
                
            end_long_term:datetime.datetime

            
       :return
        -------
             Y_pred_mean_l: np.array 1D
                The predicted  long-term price
             Y_pred_actual_l:
                The actual long-term price
             loss_score_l: np.array 1D
                The loss  from the difference between  the predicted  long-term price  and the actual long-term price

        """

        X_pred_times_l,Y_pred_mean_l,Y_pred_cov_l,Y_trad_l,Y_pred_actual_l,loss_score_pred_l,loss_score_trad_l = self.__company_wrapper_long_term.predict(
            start_time=start_long_term, end_time=end_long_term, pred_length=1)

        return Y_pred_mean_l,Y_pred_actual_l,loss_score_pred_l,loss_score_trad_l

    def gp_predict_short_term(self,  start_short_term: datetime.datetime, end_short_term: datetime.datetime):
        """
        To predict short-term price with short-term data  .
        And it returns the results and the errors.

        :arg
        -------
            start_short_term:datetime.datetime

            end_short_term:datetime.datetime

       :return
        -------
             Y_pred_mean_s: np.array 1D
                 The predicted  short-term price
             Y_pred_actual_s:np.array 1D
                The actual short-term price
             loss_score_s:np.array 1D
             The loss  from the difference between  the predicted  short-term price  and the actual short-term price

        """

        X_pred_times_s, Y_pred_mean_s, Y_pred_cov_s,Y_trad_s, Y_pred_actual_s, loss_score_pred_s,loss_score_trad_s = self.__company_wrapper_short_term.predict(
            start_time=start_short_term, end_time=end_short_term, pred_length=1)

        return  Y_pred_mean_s, Y_pred_actual_s,loss_score_pred_s,loss_score_trad_s


    def get_short_term_data(self, start_short_term: datetime.datetime, end_short_term: datetime.datetime,forward_window:int):
        Y_pred_mean_s_list=[]
        Y_pred_actual_s_list=[]
        loss_score_pred_s_list=[]
        loss_score_trad_s_list=[]
        for i in range(0,forward_window):
            end_short_term = self.__company_wrapper_short_term.find_matched_time_with_increment(begin_time=end_short_term,
                                                                                       time_increment=1)
            Y_pred_mean_s, Y_pred_actual_s,loss_score_pred_s,loss_score_trad_s=self.gp_predict_short_term(start_short_term=start_short_term, end_short_term=end_short_term)
            Y_pred_mean_s_list.append(Y_pred_mean_s[0][0])
            Y_pred_actual_s_list.append(Y_pred_actual_s[0][0])
            loss_score_pred_s_list.append(loss_score_pred_s)
            loss_score_trad_s_list.append(loss_score_trad_s)

        Y_pred_mean_s_list=np.array(Y_pred_mean_s_list)
        Y_pred_actual_s_list=np.array(Y_pred_actual_s_list)
        return Y_pred_mean_s_list,Y_pred_actual_s_list,loss_score_pred_s_list,loss_score_trad_s_list



    def get_short_term_data_single_short_term(self, start_short_term: datetime.datetime, end_short_term: datetime.datetime,forward_window:int):

        end_short_term = self.__company_wrapper_short_term.find_matched_time_with_increment(begin_time=end_short_term,
                                                                                       time_increment=forward_window)
        Y_pred_mean_s, Y_pred_actual_s,loss_score_pred_s,loss_score_trad_s=self.gp_predict_short_term(start_short_term=start_short_term, end_short_term=end_short_term)

        return Y_pred_mean_s[0], Y_pred_actual_s[0], loss_score_pred_s, loss_score_trad_s


    def get_long_term_data(self, start_long_term: datetime.datetime, end_long_term: datetime.datetime):
        Y_pred_mean_l, Y_pred_actual_l, loss_score_pred_l, loss_score_trad_l=self.gp_predict_long_term(start_long_term=start_long_term, end_long_term=end_long_term)
        return Y_pred_mean_l[0], Y_pred_actual_l[0], loss_score_pred_l, loss_score_trad_l




    def fusion_next_price(self, start: datetime.datetime, end: datetime.datetime, forward_window=0):
        """
           it is going to predict the  long-term price at time point (long_term_end+1) using long-term GP, short-term GP and  fusion algorithm.

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
       :return
        -------
                Y_mean_fusion:float
                    the fusion long-term price
                Y_pred_mean_l:float
                    the predicted long-term price
                Y_pred_actual_l: float
                    the actual long-term price
                error_fusion:float
                    the difference between the fusion  price and the actual price
                error_long_term:float
                    the difference between the predicted long-term price and the actual price
     
        """
        start_l = start
        start_s = start
        end_l = end
        end_s = end




        Y_pred_mean_l, Y_pred_actual_l, loss_score_pred_l, loss_score_trad_l = \
            self.get_long_term_data(start_long_term=start_l, end_long_term=end_l)

        Y_pred_mean_s_list,Y_pred_actual_s_list,loss_score_pred_s_list,loss_score_trad_s_list= \
            self.get_short_term_data(start_short_term=start_l, end_short_term=end_l,forward_window=forward_window)

        # using fusion to fuse long-term and short-term

        Y_mean_fusion,error_fusion,error_l,error_s=self.__fusion.fusion_price_long_term(pred_l=Y_pred_mean_l,
                                                        pred_s_list=Y_pred_mean_s_list,actual_l=Y_pred_actual_l)


        X_time = self.__company_wrapper_long_term.find_matched_time_with_increment(begin_time=end_l, time_increment=1)


        return X_time,Y_mean_fusion,Y_pred_mean_l,Y_pred_actual_l,error_fusion,error_l,error_s



    def fusion_next_price_single_short_term_data(self, start: datetime.datetime, end: datetime.datetime, forward_window=0):
        """
            it is going to predict the  long-term price at time point (long_term_end+1) using long-term GP, short-term GP and  fusion algorithm.

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
        :return
         -------
                 Y_mean_fusion:float
                     the fusion long-term price
                 Y_pred_mean_l:float
                     the predicted long-term price
                 Y_pred_actual_l: float
                     the actual long-term price
                 error_fusion:float
                     the difference between the fusion  price and the actual price
                 error_long_term:float
                     the difference between the predicted long-term price and the actual price

         """
        start_l = start
        start_s = start
        end_l = end
        end_s = end

        Y_pred_mean_l, Y_pred_actual_l, loss_score_pred_l, loss_score_trad_l = \
            self.get_long_term_data(start_long_term=start_l, end_long_term=end_l)

        Y_pred_mean_s_list, Y_pred_actual_s_list, loss_score_pred_s_list, loss_score_trad_s_list = \
            self.get_short_term_data_single_short_term(start_short_term=start_l, end_short_term=end_l, forward_window=forward_window)

        # using fusion to fuse long-term and short-term

        Y_mean_fusion, error_fusion, error_l, error_s = self.__fusion.fusion_price_long_term(pred_l=Y_pred_mean_l,
                                                                                             pred_s_list=Y_pred_mean_s_list,
                                                                                             actual_l=Y_pred_actual_l)

        X_time = self.__company_wrapper_long_term.find_matched_time_with_increment(begin_time=end_l, time_increment=1)

        return X_time, Y_mean_fusion, Y_pred_mean_l, Y_pred_actual_l, error_fusion, error_l, error_s

    def fusion_single_trad(self, start: datetime.datetime, end: datetime.datetime, forward_window=0):
        """
            it is going to predict the  long-term price at time point (long_term_end+1) using long-term GP, short-term GP and  fusion algorithm.

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
        :return
         -------
                 Y_mean_fusion:float
                     the fusion long-term price
                 Y_pred_mean_l:float
                     the predicted long-term price
                 Y_pred_actual_l: float
                     the actual long-term price
                 error_fusion:float
                     the difference between the fusion  price and the actual price
                 error_long_term:float
                     the difference between the predicted long-term price and the actual price

         """
        start_l = start
        start_s = start
        end_l = end
        end_s = end

        Y_pred_mean_l, Y_pred_actual_l, loss_score_pred_l, loss_score_trad_l = \
            self.get_long_term_data(start_long_term=start_l, end_long_term=end_l)

        Y_pred_mean_s_list, Y_pred_actual_s_list, loss_score_pred_s_list, loss_score_trad_s_list = \
            self.get_short_term_data_single_short_term(start_short_term=start_l, end_short_term=end_l, forward_window=forward_window)

        # using fusion to fuse long-term and short-term

        Y_mean_fusion, error_fusion, error_l, error_s = self.__fusion.fusion_price_long_term_trad(pred_l=Y_pred_mean_l,
                                                                                             pred_s_list=Y_pred_mean_s_list,
                                                                                             actual_l=Y_pred_actual_l)

        X_time = self.__company_wrapper_long_term.find_matched_time_with_increment(begin_time=end_l, time_increment=1)

        return X_time, Y_mean_fusion, Y_pred_mean_l, Y_pred_actual_l, error_fusion, error_l, error_s



    def fusion_next_price_cumulative(self,start:datetime.datetime,predict_begin:datetime.datetime,end:datetime.datetime, forward_window=0):
        """
           it is going to predict the  long-term price at time point cumulatively using long-term GP, short-term GP and  fusion algorithm.

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
               predict_begin:datetime.datetime
               end:datetime.datetime
       :return
        -------
                Y_mean_fusion:float
                    the fusion long-term price
                Y_pred_mean_l:float
                    the predicted long-term price
                Y_pred_actual_l: float
                    the actual long-term price
                error_fusion:float
                    the difference between the fusion  price and the actual price
                error_long_term:float
                    the difference between the predicted long-term price and the actual price

        """

        # initial the start time
        start_time=self.__company_wrapper_long_term.find_matched_time_with_increment(begin_time=start,time_increment=1)
        end_tmp=self.__company_wrapper_long_term.find_matched_time_with_increment(begin_time=predict_begin,time_increment=1)

        #  list for error evaluation
        error_fusion_list=[]
        error_long_term_list=[]
        error_short_term_list=[]
        time_index=[]

        # predict next price cumulatively
        while(end_tmp<end):
            X_time, Y_mean_fusion, Y_pred_mean_l, Y_pred_actual_l, error_fusion, error_l,error_s = self.fusion_next_price(
                start=start_time, end=end_tmp, forward_window=forward_window)

            error_short_term_list.append(error_s)
            error_fusion_list.append(error_fusion)
            error_long_term_list.append(error_l)
            time_index.append(X_time)
            end_tmp = self.__company_wrapper_long_term.find_matched_time_with_increment(begin_time=end_tmp,
                                                                                        time_increment=1)



        # evaluate the errors
        df_error_fusion_list=pd.DataFrame(data=error_fusion_list,index=time_index)
        df_error_long_term_list = pd.DataFrame(data=error_long_term_list, index=time_index)
        df_error_short_term_list = pd.DataFrame(data=error_short_term_list, index=time_index)
        df_error_fusion_list.columns = [forward_window]
        df_error_long_term_list.columns = [forward_window]


        return df_error_fusion_list,df_error_long_term_list,df_error_short_term_list,time_index

    def fusion_next_price_multi(self,start:datetime.datetime,end:datetime.datetime, forward_window=0,fitting_windows=3):
        """
           it is going to predict the  long-term price at time point cumulatively using long-term GP, short-term GP and  fusion algorithm.

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
               predict_begin:datetime.datetime
               end:datetime.datetime
       :return
        -------
                Y_mean_fusion:float
                    the fusion long-term price
                Y_pred_mean_l:float
                    the predicted long-term price
                Y_pred_actual_l: float
                    the actual long-term price
                error_fusion:float
                    the difference between the fusion  price and the actual price
                error_long_term:float
                    the difference between the predicted long-term price and the actual price

        """

        # initial the start time
        start_time=self.__company_wrapper_long_term.find_matched_time_with_increment(begin_time=start,time_increment=1)
        end_tmp=self.__company_wrapper_long_term.find_matched_time_with_increment(begin_time=start_time,time_increment=fitting_windows-1)

        #  list for error evaluation
        error_fusion_list=[]
        error_long_term_list=[]
        error_short_term_list=[]
        time_index=[]

        # predict next price cumulatively
        while(end_tmp<end):
            X_time, Y_mean_fusion, Y_pred_mean_l, Y_pred_actual_l, error_fusion, error_l,error_s = self.fusion_next_price(
                start=start_time, end=end_tmp, forward_window=forward_window)


            error_fusion_list.append(error_fusion)
            error_long_term_list.append(error_l)
            error_short_term_list.append(error_s)
            time_index.append(X_time)
            end_tmp = self.__company_wrapper_long_term.find_matched_time_with_increment(begin_time=end_tmp,
                                                                                        time_increment=1)
            start_time = self.__company_wrapper_long_term.find_matched_time_with_increment(begin_time=start_time,
                                                                                           time_increment=1)


        # evaluate the errors
        df_error_fusion_list=pd.DataFrame(data=error_fusion_list,index=time_index)
        df_error_long_term_list = pd.DataFrame(data=error_long_term_list, index=time_index)
        df_error_short_term_list = pd.DataFrame(data=error_short_term_list, index=time_index)
        df_error_fusion_list.columns = [forward_window]
        df_error_long_term_list.columns = [forward_window]
        df_error_short_term_list.columns = [forward_window]


        return df_error_fusion_list,df_error_long_term_list,df_error_short_term_list,time_index

    def multi_fusion_next_price_single_short_term_data(self, start: datetime.datetime, end: datetime.datetime, forward_window=0,
                                fitting_windows=3):
        """
           it is going to predict the  long-term price at time point cumulatively using long-term GP, short-term GP and  fusion algorithm.

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
               predict_begin:datetime.datetime
               end:datetime.datetime
       :return
        -------
                Y_mean_fusion:float
                    the fusion long-term price
                Y_pred_mean_l:float
                    the predicted long-term price
                Y_pred_actual_l: float
                    the actual long-term price
                error_fusion:float
                    the difference between the fusion  price and the actual price
                error_long_term:float
                    the difference between the predicted long-term price and the actual price

        """

        # initial the start time
        start_time = self.__company_wrapper_long_term.find_matched_time_with_increment(begin_time=start,
                                                                                       time_increment=1)
        end_tmp = self.__company_wrapper_long_term.find_matched_time_with_increment(begin_time=start_time,
                                                                                    time_increment=fitting_windows - 1)

        #  list for error evaluation
        error_fusion_list = []
        error_long_term_list = []
        error_short_term_list = []
        time_index = []

        # predict next price cumulatively
        while (end_tmp < end):
            X_time, Y_mean_fusion, Y_pred_mean_l, Y_pred_actual_l, error_fusion, error_l, error_s = self.fusion_next_price_single_short_term_data(
                start=start_time, end=end_tmp, forward_window=forward_window)

            error_fusion_list.append(error_fusion)
            error_long_term_list.append(error_l)
            error_short_term_list.append(error_s)
            time_index.append(X_time)
            end_tmp = self.__company_wrapper_long_term.find_matched_time_with_increment(begin_time=end_tmp,
                                                                                        time_increment=1)
            start_time = self.__company_wrapper_long_term.find_matched_time_with_increment(begin_time=start_time,
                                                                                           time_increment=1)

        # evaluate the errors
        df_error_fusion_list = pd.DataFrame(data=error_fusion_list, index=time_index)
        df_error_long_term_list = pd.DataFrame(data=error_long_term_list, index=time_index)
        df_error_short_term_list = pd.DataFrame(data=error_short_term_list, index=time_index)
        df_error_fusion_list.columns = [forward_window]
        df_error_long_term_list.columns = [forward_window]
        df_error_short_term_list.columns = [forward_window]

        return df_error_fusion_list, df_error_long_term_list, df_error_short_term_list, time_index

    def multi_fusion_single_trad(self, start: datetime.datetime, end: datetime.datetime, forward_window=0,
                                fitting_windows=3):
        """
           it is going to predict the  long-term price at time point cumulatively using long-term GP, short-term GP and  fusion algorithm.

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
               predict_begin:datetime.datetime
               end:datetime.datetime
       :return
        -------
                Y_mean_fusion:float
                    the fusion long-term price
                Y_pred_mean_l:float
                    the predicted long-term price
                Y_pred_actual_l: float
                    the actual long-term price
                error_fusion:float
                    the difference between the fusion  price and the actual price
                error_long_term:float
                    the difference between the predicted long-term price and the actual price

        """

        # initial the start time
        start_time = self.__company_wrapper_long_term.find_matched_time_with_increment(begin_time=start,
                                                                                       time_increment=1)
        end_tmp = self.__company_wrapper_long_term.find_matched_time_with_increment(begin_time=start_time,
                                                                                    time_increment=fitting_windows - 1)

        #  list for error evaluation
        error_fusion_list = []
        error_long_term_list = []
        error_short_term_list = []
        time_index = []

        # predict next price cumulatively
        while (end_tmp < end):
            X_time, Y_mean_fusion, Y_pred_mean_l, Y_pred_actual_l, error_fusion, error_l, error_s = self.fusion_next_price_single_short_term_data(
                start=start_time, end=end_tmp, forward_window=forward_window)

            error_fusion_list.append(error_fusion)
            error_long_term_list.append(error_l)
            error_short_term_list.append(error_s)
            time_index.append(X_time)
            end_tmp = self.__company_wrapper_long_term.find_matched_time_with_increment(begin_time=end_tmp,
                                                                                        time_increment=1)
            start_time = self.__company_wrapper_long_term.find_matched_time_with_increment(begin_time=start_time,
                                                                                           time_increment=1)

        # evaluate the errors
        df_error_fusion_list = pd.DataFrame(data=error_fusion_list, index=time_index)
        df_error_long_term_list = pd.DataFrame(data=error_long_term_list, index=time_index)
        df_error_short_term_list = pd.DataFrame(data=error_short_term_list, index=time_index)
        df_error_fusion_list.columns = [forward_window]
        df_error_long_term_list.columns = [forward_window]
        df_error_short_term_list.columns = [forward_window]

        return df_error_fusion_list, df_error_long_term_list, df_error_short_term_list, time_index



if __name__ == '__main__':

    long_term_data_file = "data_set_price/long/MSFT.csv"
    short_term_data_file="data_set_price/short/MSFT.csv"
    #data_file_month="data/long_1month/long_1month_term_MSFT.csv"
    # start_time="2008-01-01 00:00:00"
    # end_time="2010-01-01 00:00:00"
    #
    # start_time, end_time = util.convert_time_into_datetime(start_time=start_time, end_time=end_time)
    # #fff.fusion_start_end(start=start_time,end=end_time)
    # df_list=[]
    # df2_list=[]
    #
    # fff = fusion_price(csv_path_long_term=long_term_data_file, csv_path_short_term=short_term_data_file, forward_window=0)
    # errors_fusion, errors_long = fff.fusion_start_end(start=start_time,end=end_time)

    start_time="2007-01-01 00:00:00"
    predict_begin="2008-02-01 00:00:00"
    #end_time="2014-02-01 00:00:00"\
    end_time="2010-02-01 00:00:00"

    start_time = util.convert_time_into_datetime(time=start_time)
    predict_begin = util.convert_time_into_datetime(time=predict_begin)
    end_time = util.convert_time_into_datetime(time=end_time)

    #fff = fusion_wrapper_price(csv_path_long_term=long_term_data_file, csv_path_short_term=short_term_data_file,GP_type="GPR")
    #X_time,Y_mean_fusion,Y_pred_mean_l,Y_pred_actual_l,error_fusion,error_l,error_s = fff.fusion_next_price_single_short_term_data(start=start_time,end=end_time,forward_window=5)
   #
    #df_error_fusion_list,df_error_long_term_list,time_index= fff.fusion_next_price_cumulative(start=start_time,end=end_time,predict_begin=predict_begin,forward_window=5)
    print("fusion_next_price_cumulative")
   #  print("df_error_fusion_list: ",np.mean(error_fusion))
   #  print("df_error_long_term_list: ", np.mean(error_l))
    forward_window_list=[1,2]
    #
    for forward_window in forward_window_list:
        fff2 = fusion_wrapper_price(csv_path_long_term=long_term_data_file, csv_path_short_term=short_term_data_file,GP_type="GPR")
        df_error_fusion_list, df_error_long_term_list, df_error_short_term_list, time_index = fff2.multi_fusion_next_price_single_short_term_data(start=start_time,
                                                                                                     end=end_time,
                                                                                                     fitting_windows=6,
                                                                                                     forward_window=forward_window)
        print("forward_window:",forward_window)


        print("df_error_fusion_list: ",(df_error_fusion_list))
        print("df_error_long_term_list: ",(df_error_long_term_list))
        print("df_error_short_term_list: ", (df_error_short_term_list))

        fig, ax = plt.subplots()

        ax.plot(df_error_fusion_list.index, df_error_fusion_list.values,label="error of fusion ")
        ax.plot(df_error_long_term_list.index, df_error_long_term_list.values, label="error of long_term ")
        ax.plot(df_error_short_term_list.index, df_error_short_term_list.values, label="error of short_term ")
        plt.yscale("log")
        ax.legend()

        plt.show()