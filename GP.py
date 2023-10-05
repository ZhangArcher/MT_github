from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF



class GaussianProcess:

    __alpha = None
    __iterations = None
    __kernels = None
    __gp = None
    __eval_score_list=None
    __trading_time=None

    def __init__(self):
        kernel = 63 * RBF(length_scale=1)
        #self.__time_type=self.__company_data.get_time_type()
        self.__alpha = 1e-10
        self.__iterations = 10
        self.__kernels = [kernel]
        self.__gp = GaussianProcessRegressor(kernel=self.__kernels[0], alpha=self.__alpha,n_restarts_optimizer=self.__iterations,normalize_y=False)

    def fit(self,X_array,Y_array):
        self.__gp=self.__gp.fit(X_array, Y_array)
        self.__kernels.append(self.__gp.kernel_)

    def predict(self,X_pred_array):
        Y_mean, Y_cov = self.__gp.predict(X_pred_array, return_cov=True)
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
