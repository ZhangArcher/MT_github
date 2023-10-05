import numpy as np


class fusion:
    """
    fusion class , which can receive the short-term data to predict the long term data better
    Now , suppose fusion function :   fusion_long_term = f(short_term) + correction_term

     Attributes
     ----------

     Methods
     -------

     """
    __correction_term=0
    def __init__(self):
        self.__correction_term=0

    def predict_long_term(self,pred_l,pred_s,actual_l):
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
        fusion_result = pred_s + self.__correction_term
        error_fusion=np.mean(abs(actual_l-fusion_result))/np.mean(actual_l)
        error_long=np.mean(abs(actual_l-pred_l))/np.mean(actual_l)
        self.update_error(error_fusion,error_long)

        return fusion_result,error_fusion,error_long


    def update_error(self,error_fusion,error_long):
        self.__correction_term= self.__correction_term+error_fusion/2