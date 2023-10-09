import numpy as np
import mfgp.application.uq.chaospy_wrapper as cpw
import chaospy as cp

# Chengye has read
'''
question:

In analytical_mean:
    why use np.prod?  (Is it just for more effienct?)
    why not directly use  :
    for a_i in a:
        ((1 - np.cos(a_i)) / a_i) 

In analytical_var:
what is term 1 ,2,3 ? (analytical_var)



'''




def analytical_mean(a):
    if not isinstance(a, list):
        a = [a]

    '''
    numpy.prod(a, axis=None, dtype=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)
    Return the product of array elements over a given axis.
    
    numpy.cos(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'cos'>
    Cosine element-wise.
    
    
    
    '''

    return np.prod([((1 - np.cos(a_i)) / a_i) for a_i in a])


def analytical_var(a):
    if not isinstance(a, list):
        a = [a]
    m = analytical_mean(a)
    term1 = np.prod([0.5 - (np.sin(2 * a_i) / (4 * a_i)) for a_i in a])
    term2 = m**2
    term3 = 2 * m * np.prod([(np.cos(a_i) - 1) / a_i for a_i in a]) * ((-1)**(len(a)-1))
    return term1 + term2 + term3


def test_2d(a):
    distribution = cp.J(cp.Uniform(0, 1), cp.Uniform(0, 1))
    '''
    cp.j :  Joint random variable
    '''
    cp_wrapper = cpw.ChaospyWrapper(function_2d, distribution, polynomial_order=4, quadrature_order=4)

    actual_mean, actual_variance = analytical_mean(a), analytical_var(a)

    print("Analytical Mean", actual_mean)

    print("Analytical Variance", actual_variance)

    cp_wrapper.calculate_coefficients()

    mean, variance = cp_wrapper.get_mean_var()

    print("Chaospy mean", mean)

    print("Chaospy variance", variance)

    relative_error_mean, relative_error_variance = np.abs((mean - actual_mean) / actual_mean), \
                                                   np.abs((variance - actual_variance) / actual_variance)

    print("Error in mean", relative_error_mean)

    print("Error in variance", relative_error_variance)

    # assert relative_error_mean < 1e-3, "Chaospy error in mean calculation for 2d test problem"
    # assert relative_error_variance < 1e-3, "Chaospy error in variance calculation for 2d test problem"
    return cp_wrapper


def function_2d(param):
    x = np.atleast_2d(param)

    '''
    numpy.atleast_2d(*arys)
    View inputs as arrays with at least two dimensions.
    '''
    return np.sin(x[:, 0] * 2.2 * np.pi) * np.sin(x[:, 1] * np.pi)


if __name__ == '__main__':
    a = [2.2 * np.pi, np.pi]
    cp_wrapper = test_2d(a)
