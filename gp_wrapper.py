import numpy
import numpy as np
import pandas
import tensorflow
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from tensorflow import float32, float64

import GPR
import VGP


class GaussianProcessWrapper:

    __gp=None
    __type=None
    def __init__(self,GP_type="GPR"):
        self.__type=GP_type

        if(self.__type.__eq__("GPR")):
            self.__gp=GPR.GaussianProcess()
            print("GP is : GPR"
                  "")
        elif (self.__type.__eq__("VGP")):
            self.__gp=VGP.GaussianProcess()
            print("GP is : VGP")

    def fit(self,X_array,Y_array):

        new_xs=tensorflow.cast(X_array,dtype=float64)
        new_ys=tensorflow.cast(Y_array,dtype=float64)
        self.__gp.fit(X_array=np.array(new_xs),Y_array=np.array(new_ys))


    def predict(self,X_pred_array):

        new_xs=tensorflow.cast(X_pred_array,dtype=float64)
        return self.__gp.predict(np.array(new_xs))

    def get_kernels(self):
        return self.__gp.get_kernels()

    def initial_GP(self):
        self.__gp.initial_GP()
