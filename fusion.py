import numpy as np
import pandas

import util


class fusion:
    """
    fusion class , which can receive the short-term data to predict the long term data better
    Now , suppose fusion function :   fusion_long_term = f(short_term) + correction_term

     Attributes
     ----------

     Methods
     -------

     """
    __correction_term=None
    def __init__(self):
        self.__correction_term=None

    def fusion_price_long_term(self,pred_l,pred_s,actual_l):
        """
           it is going to predict the next long-term price using fusion algorithm

        :arg
        -------
             pred_l: np.array 1D
                The predicted  long-term price
             actual_l:
                The actual long-term price
             pred_s: np.array 1D
                 The predicted  short-term price

       :return
        -------
        fusion_result:float
             the fusion long-term price
        error_fusion:float
             the difference between the fusion  price and the actual price
        error_long:float
             the difference between the predicted long-term price and the actual price

        """
        if(self.__correction_term is None):
            if(len(pred_s.shape)>1):
                self.__correction_term=np.zeros(pred_s.shape[1])
            else:
                self.__correction_term = np.zeros(pred_s.shape[0])
            #self.__correction_term = ((pred_s))*0

        fusion_result = pred_s + self.__correction_term
        #fusion_result = pred_s
        error_fusion=np.mean(abs(actual_l-fusion_result))
        error_long=np.mean(abs(actual_l-pred_l))
        self.update_error_price(error_fusion,error_long)

        return fusion_result,error_fusion,error_long

    def fusion_portfolio_long_term(self,pred_l,pred_s,actual_l,return_l):
        """
           it is going to predict the next long-term portfolio using fusion algorithm

        :arg
        -------
             pred_l: np.array 1D
                The predicted  long-term price
             actual_l:
                The actual long-term price
             pred_s: np.array 1D
                 The predicted  short-term price

       :return
        -------
        fusion_result:float
             the fusion long-term price
        error_fusion:float
             the difference between the fusion  price and the actual price
        error_long:float
             the difference between the predicted long-term price and the actual price

        """



        if(self.__correction_term is None):
            self.__correction_term=np.zeros(pred_s.shape[1])
            #self.__correction_term = ((pred_s))*0

        #fusion_portfolio = pred_s + self.__correction_term
        fusion_portfolio = pred_s
        # reset time index for fusion_portfolio
        fusion_portfolio = self.reset_time_index(base_portfolio=pred_l, target_portfolio=fusion_portfolio)


        fusion_profit = util.compute_profit(portfolios=fusion_portfolio, returns=return_l)

        actual_long_term_profit = util.compute_profit(portfolios=actual_l, returns=return_l)
        pred_long_term_profit = util.compute_profit(portfolios=pred_l, returns=return_l)


        fusion_error=self.loss_profit(base_profit=actual_long_term_profit,target_profit=fusion_profit)
        pred_long_error=self.loss_profit(base_profit=actual_long_term_profit,target_profit=pred_long_term_profit)

        self.update_error_portfolio(error_fusion=fusion_error,error_long=pred_long_error)

        return fusion_portfolio,fusion_profit,fusion_error,pred_long_error

    def update_error_price(self,error_fusion:pandas.DataFrame,error_long):

        self.__correction_term= (self.__correction_term+error_fusion)/2


    def update_error_portfolio(self,error_fusion:pandas.DataFrame,error_long):

        error_fusion=error_fusion.drop(columns=["Profit"])
        ef=np.array(error_fusion.values.tolist()[0])
        self.__correction_term= self.__correction_term+ef

    def loss_profit(self,base_profit,target_profit):
      #  target_profit

        error_profit=target_profit-base_profit

        return error_profit


    def reset_time_index(self,base_portfolio:pandas.DataFrame,target_portfolio:pandas.DataFrame):
        target_portfolio.index=base_portfolio.index
        return target_portfolio