import numpy
import pandas
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF



class GaussianProcess:

    __alpha = None
    __iterations = None
    __kernels = None
    __gp = None
    __eval_score_list=None
    __trading_time=None
    __max_value=None
    __min_value=None

    def __init__(self):
        #kernel =  3*RBF(length_scale=1)\
        kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        noise_std = 0.9
        #self.__time_type=self.__company_data.get_time_type()
        #self.__alpha = 1e-10
        self.__iterations = 15
        self.__kernels = kernel
      #  self.__gp = GaussianProcessRegressor(kernel=self.__kernels, alpha=self.__alpha,n_restarts_optimizer=self.__iterations,normalize_y=True,random_state=0)
        self.__gp = GaussianProcessRegressor(kernel=self.__kernels,alpha=noise_std**2,n_restarts_optimizer=self.__iterations, normalize_y=True, random_state=0)

    def fit(self,X_array,Y_array):
       # Y=self.normalize_data(Y_array)
        self.__gp=self.__gp.fit(X_array, Y_array)
       # self.__kernels.append(self.__gp.kernel_)
        self.__gp.score(X_array, Y_array)

    def predict(self,X_pred_array):
        Y_mean, Y_cov = self.__gp.predict(X_pred_array, return_cov=True)

        #Y=self.rev_normalize_data(Y_mean)
        return Y_mean,Y_cov

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
        return self.__kernels

    def initial_GP(self):
        self.__gp = GaussianProcessRegressor(kernel=self.__kernels[0], alpha=self.__alpha,
                                             n_restarts_optimizer=self.__iterations, normalize_y=False)

    def normalize_data(self,data):
        self.__max_value = numpy.max(data)
        self.__min_value = numpy.min(data)
        normalize_ddata=(data-self.__min_value)/(self.__max_value-self.__min_value)
        return normalize_ddata

    def rev_normalize_data(self,data):
        rev_normalize_data=((self.__max_value-self.__min_value)*data)+self.__min_value
        return rev_normalize_data