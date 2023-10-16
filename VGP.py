import numpy
import pandas
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import gpflow


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

      #  self.__gp = GaussianProcessRegressor(kernel=self.__kernels, alpha=self.__alpha,n_restarts_optimizer=self.__iterations,normalize_y=True,random_state=0)
        self.__gp =None

    def fit(self,X_array,Y_array):
       # Y=self.normalize_data(Y_array)

        self.__gp=gpflow.models.VGP( (X_array, Y_array),  kernel=gpflow.kernels.SquaredExponential(),
    likelihood=gpflow.likelihoods.StudentT(),)
       # self.__kernels.append(self.__gp.kernel_)
        opt = gpflow.optimizers.Scipy()

        opt.minimize(self.__gp.training_loss, self.__gp.trainable_variables)

    def predict(self,X_pred_array):
        Y_mean, Y_cov = self.__gp.predict_y(X_pred_array, full_cov=False)
        #Y=self.rev_normalize_data(Y_mean)
        Y_mean = Y_mean.numpy()
        Y_cov = Y_cov.numpy()
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
        pass

    def normalize_data(self,data):
        self.__max_value = numpy.max(data)
        self.__min_value = numpy.min(data)
        normalize_ddata=(data-self.__min_value)/(self.__max_value-self.__min_value)
        return normalize_ddata

    def rev_normalize_data(self,data):
        rev_normalize_data=((self.__max_value-self.__min_value)*data)+self.__min_value
        return rev_normalize_data