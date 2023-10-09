from mfgp.acquisition_functions.abstract_acquisition import AbstractAcquisitionFun
import numpy as np
from scipy.optimize import approx_fprime
#chengye has read

""" wrapper class for the maximum uncertainty acquisition function which is used as the objective
    function in the optimization problem inside model adaptation step                          

 Attributes                                                                                    
 ----------                                                                                    
 model_predict                                                                                 

 Methods                                                                                       
 -------                                                                                       
 acquisition_curve                                                                             
 derivative_aprroximation                                                                      

 """



class MaxUncertaintyAcquisition(AbstractAcquisitionFun):
    """wrapper class for the maximum uncertainty acquisition function which is used as the objective
    function in the optimization problem inside model adaptation step.
    """

    def __init__(self, model_predict: callable):
        """
        Parameters
        ----------
        model_predict : callable
            Predict function of GP
        """
        super().__init__(model_predict=model_predict)

    def acquisition_curve(self, x: np.ndarray):
        """Returns the value the variance at required location

        Parameters
        ----------
        x : np.ndarray
            Target location for evaluation of acquisition function

        Returns
        -------
        np.ndarray
            Value of acquisition function at the target locations
        """
        _, uncertainty = self.model_predict(x[None])    
        return uncertainty[:, None].ravel()

    def derivative_aprroximation(self, x: float):
        """computes the finite difference approximation of the derivative at a specific point x within the input space.
           Used in optimization methods with needs the derivatives.
        Parameters
        ----------
        x : float
            Location at which you want to calculate derivative of acquisition curve
        """
        eps = np.sqrt(np.finfo(float).eps)
        # grad = np.zeros_like(x)
        # for i in range(len(x)):
        # grad[i] = (self.acquisition_curve(x[i]+eps) - self.acquisition_curve(x[i]-eps) ) / (2*eps)
        # print("----- grad",grad," -  grad.shape", grad.shape,"-----")
        # return grad
        return approx_fprime(x, self.acquisition_curve, eps)
        """
        scipy.optimize.approx_fprime(xk, f, epsilon=1.4901161193847656e-08, *args)
        
        Returns:
            jac :ndarray
                The partial derivatives of f to xk.
        
        
        Finite difference approximation of the derivatives of a scalar or vector-valued function.
        
        """