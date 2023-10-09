import numpy as np
from mfgp.adaptation_maximizers.abstract_maximizer import AbstractMaximizer
from scipy.optimize import minimize
from scipy.optimize import Bounds
#chengye has read
"""Wrapper class for the uncertainty maximization in the adaptation 
process usin Nelder Mead method.
https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method

we maximize f(x) by minimizing -f(x)


PS :
ScipyOpt maybe contain some bug.
try to use ScipyOpt1.




"""




class ScipyOpt(AbstractMaximizer):
    """Wrapper class for the uncertainty maximization in the adaptation 
    process usin Nelder Mead method.
    """

    def __init__(self, parallelization=False, n_restarts=6):
        super().__init__()
        self.n_restarts = n_restarts 
        self.parallelization = parallelization

    def one_opt(self, function, lower_bound, upper_bound, method='L-BFGS-B', maxiter=100):
        x0 = np.random.uniform(lower_bound, upper_bound)
        res = minimize(function, x0, bounds=Bounds(lower_bound, upper_bound), method=method, options={'maxfev':maxiter})
        '''
        scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, 
        hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
        
        Minimization of scalar function of one or more variables.
        
        Returns
            resOptimizeResult
                The optimization result represented as a OptimizeResult object. 
                Important attributes are: x the solution array, 
                success a Boolean flag indicating if the optimizer exited successfully 
                and message which describes the cause of the termination. See OptimizeResult for a description of other attributes.
       
       
       OptimizeResult:
            Represents the optimization result.
            OptimizeResult may have additional attributes not listed here depending on the specific solver being used. Since this class is essentially a subclass of dict with attribute accessors, one can see which attributes are available using the OptimizeResult.keys method.

            Attributes:
                x :ndarray
                    The solution of the optimization.
                fun, : ndarray
                    Values of objective function, 
                
            so res.x is the input , res.fun is the output
        '''



        return res.x, res.fun

    def maximize(self, function, lower_bound: np.ndarray, upper_bound: np.ndarray, method='L-BFGS-B'):
        '''
        In this implementation, we perform gradient based optimisation choosing a random initial point.
        We perform this process, n_restart times. Then we choose the point with the highest value
        TODO
        1. Multiple starting point. Write a separate function for that
        2. Remove points that are out of bounds.
        2. Choose point with the minimum value.
        3. Multiprocessing
        '''
        neg_function = lambda x: -1 * function(x)
        X, f = [], []
        for i in range(self.n_restarts):
            x, fun = self.one_opt(neg_function, lower_bound, upper_bound, method=method)
            print(i, x, fun)
            X.append(x)
            f.append(fun) 
        minval_index = np.argmin(f)
        selected_point, val = X[minval_index], float(f[minval_index])
        print("Selected point is", selected_point, "with acquisition function", val)
        return selected_point, val  #I think we should return -1.* val. TODO: Think about it and check it.
    #I think so


class ScipyOpt1(AbstractMaximizer):
    """Wrapper class for the uncertainty maximization in the adaptation 
    process usin Nelder Mead method.
    """

    def __init__(self, num_initial_evaluations=1000):
        self.num_initial_evaluations = num_initial_evaluations
        super().__init__()

    def find_initial_point(self, function: callable, lower_bound:np.ndarray, upper_bound: np.ndarray):
        dim =  len(lower_bound)
        sampled_points =  np.random.uniform(lower_bound, upper_bound, (self.num_initial_evaluations, dim))
        f = function(sampled_points).ravel()
        minval_index = np.argmin(f)
        return sampled_points[minval_index]

    def maximize(self, function, lower_bound: np.ndarray, upper_bound: np.ndarray, method='L-BFGS-B', maxiter=100):
        '''
        In this implementation, we perform evaluate the function at many randim points. 
        Then we choose the optimum point, and start the gradient based optimisation from the aforementioned point.
        '''
        # neg_function = lambda x: -1. * function(x[:, None])
        def neg_function(x):
            return -1. * function(np.atleast_2d(x)).ravel()
        initial_point = self.find_initial_point(neg_function, lower_bound, upper_bound)
        # print("Initial point", initial_point)
        res = minimize(neg_function, initial_point, bounds=Bounds(lower_bound, upper_bound), method=method, options={'maxfev':maxiter})
        print("Selected point is", res.x, "with acquisition function", res.fun)
        return res.x, -1.*res.fun