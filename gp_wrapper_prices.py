

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import data_handler_price
import logging
import datetime
import util
import gp_wrapper



class gp_wrapper_price:
    """
    Gaussian Process wrapper class for stock prices
    wrapper_gp_price offer a traditional Gaussian Process Model that is generated from stock prices
    It can predict the stock prices based on Gaussian Process Regressor.

     Attributes
     ----------
        __company_data : data_handler_price.data_handler_price

        __prices_data : Dataframe

        __gp :GP.GaussianProcess


     Methods
        predict:

        add_pred_time_into_time_interval:

        predict_by_time_interval_cumulative:

        get_kernels

        get_loss_function_result

        add_eval_score

        initial_GP

        get_trading_time
     -------
     """

    __company_data = None
   # __prices_data = None
    __gp=None
    __eval_score_list=None
    __trading_time=None
    __GP_type=None
    def __init__(self, csv_file: str,GP_type="GPR"):
        """
        To initialize  class
        to set Gaussian Process Regressor
        :arg
        -------
              csv_file: str
                  the path of the data set
        :return
        -------
              None
        """

        self.__GP_type=GP_type
        self.__company_data = data_handler_price.data_handler_price(csv_file)
        self.__gp=gp_wrapper.GaussianProcessWrapper(GP_type=self.__GP_type)
        self.__eval_score_list=[]



    def predict(self,start_time: datetime.datetime,
                end_time: datetime.datetime ,pred_length: int):
        """
        predict the target price(s) between [end_time+1,end_time+pred_length] using a GP.
        This GP is generated/fitted from the data set between [start_time,end_time]

        :arg
        -------
            start_time：datetime.datetime
                start time of the train data
            end_time：datetime.datetime
                end time of the train data
            pred_length: int
               The size of the time length what we want to predict (ignored time unit)
            eval_error: bool
                whether or not evaluate the result using a loss function
            add_correction_term:bool
                whether or not to use the correction_term
       :return
        -------
            X_pred_times:np.array 1D
                The predict  time points 
            Y_pred_mean: np.array 1D
                the corresponding mean  of the predicted GP 
            Y_pred_cov: np.array 1D
                the corresponding cov  of the predicted GP 
            Y_pred_actual: np.array 1D
                the actual result
            loss_score: list
                the score of loss function

        """

        assert start_time<end_time
        assert pred_length>=1
        start_time=self.find_matched_time_with_increment(begin_time=start_time,time_increment=0)
        end_time = self.find_matched_time_with_increment(begin_time=end_time, time_increment=0)


        # get the train data and the predicted target time.
        pred_times = self.find_all_matched_times_by_length(begin_time=end_time, time_length=pred_length)
        pred_start_time = self.find_matched_time_with_increment(begin_time=end_time, time_increment=1)
        pred_end_time=self.find_matched_time_with_increment(begin_time=end_time, time_increment=pred_length)


        # The data set between [start_time,end_time]
        X_train_array, Y_train_array, X_pred_array, Y_pred_actual  = self.get_data_set(
            start_time=start_time, end_time=end_time,pred_start_time=pred_start_time,pred_end_time=pred_end_time)
        # train GP
        self.__gp.fit(X_train_array, Y_train_array)
        assert len(X_pred_array)>0
        # predict the result
        Y_pred_mean, Y_pred_cov = self.__gp.predict(X_pred_array)
       # Y_pred_mean=np.array(Y_pred_mean)
      #  X_pred_times=np.array(pred_times)
        X_pred_times=pred_times.tolist()
        Y_trad=[]
        if(pred_length==1):
            Y_trad.append((Y_train_array[-1]).tolist())
        else:
            Y_trad.append(Y_train_array[-1].tolist())
            Y_trad.extend(Y_pred_actual[0:-1].tolist())



        loss_score_pred=self.get_loss_function_result(Y_pred_mean,Y_pred_actual)
        loss_score_trad = self.get_loss_function_result(Y_trad, Y_pred_actual)

        return X_pred_times,Y_pred_mean,Y_pred_cov,Y_trad,Y_pred_actual,loss_score_pred,loss_score_trad



    def get_data_set_size_by_time_interval(self,start_time:datetime.datetime,
                                           end_time:datetime.datetime):
        """
         get the size of the corresponding data sets
         :arg
         -------
             start_time:datetime.datetime

             end_time:datetime.datetime

        :return
         -------
           size_data_set: int
        """
        train_price_data = self.__company_data.get_price_data_by_time_intetval(start_time, end_time)
        train_price_data.sort_values("timestamp")
        size_of_data = train_price_data.shape[0]
        return size_of_data

    def get_data_set(self,start_time:datetime.datetime,end_time:datetime.datetime,
                     pred_start_time:datetime.datetime, pred_end_time:datetime.datetime):
        """

           get the train data set between [start_time,end_time]
           and the predict actual data set [pred_start_time,pred_end_time]


        :arg
        -------
               start_time:datetime.datetime

               end_time:datetime.datetime

               pred_start_time:datetime.datetime

               pred_end_time:datetime.datetime
       :return
        -------
              X_array:np.array  2D
                [[year,data_id],.......]

              Y_array:np.array 1D
                [price,....]

              X_pred_array:np.array 2D
                [[year,data_id],.......]

              Y_pred_target_array:np.array  1D
                [price,....]
        """
        X_train = []
        Y_train = []
        #   get the train data set between [start_time,end_time]
        train_price_data=self.__company_data.get_price_data_by_time_intetval(start_time,end_time)
        train_price_data.sort_values("timestamp")
        # reconstruct data set for the training and the prediction
        size_of_data=train_price_data.shape[0]
        data_id=0


        for i in range(0,size_of_data):
            data_id = data_id + 1
            close_p=train_price_data.iloc[i]["Close"]
         #   X_train.append([1,int(data_id)])
            X_train.append([int(data_id)])
            Y_train.append([close_p])

        X_train_array = np.array(X_train)
        Y_train_array = np.array(Y_train)

        #   get the predicted data set between  [pred_start_time,pred_end_time]
        pred_price_data = self.__company_data.get_price_data_by_time_intetval(pred_start_time, pred_end_time)
        pred_price_data.sort_values("timestamp")
        assert pred_start_time <= pred_end_time
        size_of_pred_data = pred_price_data.shape[0]

        while(size_of_pred_data==0):
            pred_end_time=self.find_matched_time_with_increment(begin_time=pred_end_time,time_increment=1)
            pred_price_data = self.__company_data.get_price_data_by_time_intetval(pred_start_time, pred_end_time)
            pred_price_data.sort_values("timestamp")
            assert pred_start_time < pred_end_time
            size_of_pred_data = pred_price_data.shape[0]

        X_pred = []
        Y_pred = []

        for i in range(0, size_of_pred_data):
            close_pred = pred_price_data.iloc[i]["Close"]
            X_pred.append([int(data_id)])
            data_id = data_id + 1
            Y_pred.append([close_pred])

        X_pred_array = np.array(X_pred)
        Y_pred_array = np.array(Y_pred)


        return X_train_array,Y_train_array,X_pred_array,Y_pred_array


    def get_trading_time(self):
        return self.__company_data.get_trading_time()

    def find_all_matched_times_by_length(self,begin_time:datetime.datetime,time_length:int):

        return util.find_all_matched_times_by_length(trading_time=self.get_trading_time(),begin_time=begin_time,time_length=time_length)


    def find_matched_time_with_increment(self,begin_time:datetime.datetime,time_increment:int):

        return util.find_matched_time_with_increment(trading_time=self.get_trading_time(),
                                                     begin_time=begin_time,time_increment=time_increment)



    def predict_multi(self,start_time: datetime.datetime, end_time: datetime.datetime,
                           fitting_windows:int, add_correction_term=False):
        """

       predict the target price with a fixed time windows multiple
       For example:
       iteration 1 :
         predict the target price(s) at time point (start_time+fitting_length) using a GP.
        This GP is generated/fitted from the data set between [start_time,start_time+fitting_length-1]

        iteration 2:
          predict the target price(s) at time point (start_time+fitting_length+1) using a GP.
        This GP is generated/fitted by the data set between  [start_time+1,start_time+fitting_length]

        iteration ....

        iteration n:
          predict the target price(s) at time point (start_time+fitting_length+n-1) using a GP.
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
        assert fitting_windows>0

        # get the cooresponding time points
        temp_start_time=self.find_matched_time_with_increment(begin_time=start_time,time_increment=0)
        temp_end_time=self.find_matched_time_with_increment(begin_time=temp_start_time,time_increment=fitting_windows-1)


        #initial
        X=np.array([])
        Y_mean=np.array([])
        Y_cov=np.array([])
        Y = np.array([])
        loss_score_pred_list=[]
        loss_score_trad_list = []
        time_index=[]
        mean_list=[]
        var_list=[]

        #predict next price cumulatively
        while(temp_end_time<=end_time):

            x_pred, Y_mean, Y_cov,Y_trad,Y_actual, loss_score_pred, loss_score_trad= self.predict(start_time=temp_start_time,
                                                                      end_time=temp_end_time,pred_length=1)
            if(add_correction_term):
                # add a correction_term
                Y_mean=self.add_correction_term(Y_mean)

            time_index.append(x_pred[0])
            loss_score_pred_list.append(loss_score_pred)
            loss_score_trad_list.append(loss_score_trad)
            mean_list.append(Y_mean[0])
            var_list.append(Y_cov[0])
            self.update_eval_score(loss_score_pred)

            temp_start_time=self.find_matched_time_with_increment(begin_time=temp_start_time,time_increment=1)
            temp_end_time=self.find_matched_time_with_increment(begin_time=temp_end_time,time_increment=1)

        #collect the result
        loss_score_pred=pd.DataFrame(data=loss_score_pred_list,index=time_index)
        loss_score_pred.columns=[fitting_windows]
        loss_score_trad=pd.DataFrame(data=loss_score_trad_list,index=time_index)
        loss_score_trad.columns=[fitting_windows]

        df_mean=pd.DataFrame(data=mean_list,index=time_index)
        df_mean.columns = [fitting_windows]
        df_var=pd.DataFrame(data=var_list,index=time_index)
        df_mean.columns = [fitting_windows]


        return df_mean,df_var,loss_score_pred,loss_score_trad

    def get_kernels(self):
        """
           return kernel
        :arg
        -------
            None

        :return
        -------
            kernel
        """

        return self.__gp.get_kernels()

    def get_loss_function_result(self,pred: np.array,actual:np.array):
        """
             get the result of loss function according pred_result and result
         :arg
         -------
            result: np.array 1D
                The target what we want to predict
            pred_result:  np.array 1D
                Our prediction
         :return
         -------
            score:float
                The score form loss function
        """

        score=np.mean(abs(pred-actual))


        return score

    def update_eval_score(self,score):
        """
         add loss score  into __eval_score_list
         :arg
         -------
            score: int/float
                score/result from loss function

         :return
         -------
            None
        """
        self.__eval_score_list.append(score)

    def add_correction_term(self,y_mean:np.ndarray):
        """
             compute a correction_term according to __eval_score_list  and add the correction_term into y_mean
         :arg
         -------
              y_mean: ndarray 1D
         :return
         -------
               y_mean_adjusted : ndarray 1D
               y_mean added by the correction_term
        """

        if(len(self.__eval_score_list)==0):
            score_mean=0
        else:
            score_mean=np.mean(self.__eval_score_list)
        y_mean_adjusted=y_mean+(score_mean)
        return y_mean_adjusted

    def get_trading_time(self):
        return self.__company_data.get_trading_time()

    def initial_GP(self):
        self.__gp.initial_GP()

if __name__ == '__main__':
    data_file_month="data_set_price/long/AAPL.csv"

    start_time = "2008-01-01 00:00:00"
    end_time = "2010-01-01 00:00:00"
    start=util.convert_time_into_datetime(time=start_time)
    end = util.convert_time_into_datetime(time=end_time)
    dpp=gp_wrapper_price(csv_file=data_file_month,GP_type="VGP")
    # score_list=[]
    # mean_list=[]
    # var_list=[]
    #df_mean,df_var,df_score=dpp.predict_cumulative(start_time=start, end_time=end,fitting_length=10,add_correction_term=True)

    #
    X_pred_times,Y_pred_mean,Y_pred_cov,Y_trad,Y_pred_actual,loss_score_pred,loss_score_trad=dpp.predict(start_time=start, end_time=end,pred_length=4)
    #df_mean,df_var,loss_score_pred,loss_score_trad = dpp.predict_multi(start_time=start, end_time=end, fitting_windows=5)
    # print("Y_pred_mean:",Y_pred_mean)
    # print("Y_pred_actual:",Y_pred_actual)
    # print("score:",loss_score)
    #
    # print("df_mean:",df_mean)
    # print("df_var:",df_var)
    # print("loss_score_pred:",np.mean(loss_score_pred))
    # print("loss_score_trad:",np.mean(loss_score_trad))
    # print("asdsdsd")