import numpy as np
from mfgp.application.uq.gpc_abstract import AbstractGPC
import chaospy as cp

# Chengye has read
# Remarks chaospy works strangely when it comes to calculation of the statistical moments
# Even with very low degree polynomials and lowe quadrature order, it calculate mean very accurately


"""
 wrapper class for the polynomial interpolant using gaussian quadrature .

polynomial interpolant
In numerical analysis, polynomial interpolation is the interpolation of a
given data set by the polynomial of lowest possible degree that passes through the points of the dataset


 Attributes
 ----------
function : Calculate the polynomial chaos expansion of the MFGP after adaptation
 distribution :
 polynomial_order :
 quadrature_order :
polynomial_expansion:
f_approx: polynomial approximation function


 """


class ChaospyWrapper(AbstractGPC):

    def __init__(self, function: callable, distribution: cp.distributions, polynomial_order=8, quadrature_order=8):
        self.distribution, self.polynomial_order, self.quadrature_order = distribution, polynomial_order, quadrature_order
        self.quad_points, self.quad_weights = cp.generate_quadrature(self.quadrature_order, self.distribution, rule="gaussian")


        self.polynomial_expansion = cp.generate_expansion(self.polynomial_order, self.distribution)


        self.f_approx = None
        super().__init__(function)

    def calculate_coefficients(self):
        evaluations = self.function(self.quad_points.T)
        self.f_approx, self.coefficients = cp.fit_quadrature(self.polynomial_expansion, self.quad_points,
                                                             self.quad_weights, evaluations, retall=True)
        # chaospy.fit_quadrature(orth, nodes, weights, solves, retall=False, norms=None)
        #Fit polynomial chaos expansion using spectral projection.
        # Create a polynomial approximation model from orthogonal expansion, quadrature nodes and weights.

    def get_mean(self):
        # return self.coefficients[0]
        # chaospy.E(poly, dist=None, **kws) : The expected value of a distribution or polynomial.
        return cp.E(self.f_approx, self.distribution)

    def get_var(self):
        # return np.sum(self.coefficients**2) - self.get_mean()**2
        # chaospy.Var(poly, dist=None, **kws) : The variance of a distribution or polynomial.

        return cp.Var(self.f_approx, self.distribution)

    def update_order(self, new_order):
        self.polynomial_order, self.quadrature_order = new_order, new_order
        self.quad_points, self.quad_weights = cp.generate_quadrature(self.quadrature_order, self.distribution, rule="gaussian")
        self.polynomial_expansion = cp.generate_expansion(self.polynomial_order, self.distribution)
