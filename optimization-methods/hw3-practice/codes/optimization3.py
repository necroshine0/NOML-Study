from collections import defaultdict
import numpy as np
from numpy.linalg import norm
import scipy
import scipy.sparse
import scipy.optimize

# from scipy.sparse.linalg import minres

import time
from utils import get_line_search_tool
import oracles


def findAlpha(oracle, x_k, u_k, dx, du, options, theta=0.99):

    diff_1 = dx - du
    diff_2 = dx + du

    inds_1 = np.argwhere(diff_1 > 0)[:, 0]
    inds_2 = np.argwhere(diff_2 < 0)[:, 0]

    alphas = []
    alphas = np.append(alphas, (-x_k + u_k)[inds_1] / diff_1[inds_1])
    alphas = np.append(alphas, -(x_k + u_k)[inds_2] / diff_2[inds_2])

    try:
        alpha_0 = min(1., theta * np.min(alphas))
    except:
        alpha_0 = 1.

    line_search_tool = get_line_search_tool(options)
    alpha = line_search_tool.line_search(oracle, np.hstack((x_k, u_k)),
                                         np.hstack((dx, du)), previous_alpha=alpha_0)
    return alpha


def newtonLasso(oracle, v_0, grad_norm_0, theta=0.99, c1=1e-4, tolerance=1e-5, max_iter=100,
                line_search_options=None, fix=0.01, print_errors=False):
    # fix var is needed in case if LinAlgError

    if line_search_options is None:
        line_search_options = {'method': 'Armijo', 'c1': c1}

    def valid_check(list_):
        for val in list_:
            if val is not None:
                if type(val) == np.ndarray:
                    if np.inf in val or True in np.isnan(val):
                        return True
                else:
                    if np.isinf(val) or np.isnan(val):
                        return True
            else:
                return True
        return False

    norm_0 = tolerance * grad_norm_0
    if valid_check([norm_0]):
        print('grad_0 error') if print_errors else 0
        return np.array([]), 'computational_error'

    v_k = np.copy(v_0)

    for k in range(max_iter):
        nabla_f = oracle.grad(v_k)
        norm_val = nabla_f.dot(nabla_f)
        if valid_check([nabla_f, norm_val]):
            print('grad error') if print_errors else 0
            return v_k, 'computational_error'

        if norm_val <= norm_0:
            return v_k, 'success'

        hess_f = oracle.hess(v_k)

        try:
            U, lower = scipy.linalg.cho_factor(hess_f, check_finite=True)
        except np.linalg.LinAlgError:
            try:
                U, lower = scipy.linalg.cho_factor(hess_f + fix * np.eye(hess_f.shape[0]), check_finite=True)
            except:
                if print_errors:
                    print('LinAlgError')
                    print(np.all(np.linalg.eigvals(hess_f) > 0))
                return v_k, 'computational_error'

        d_k = scipy.linalg.cho_solve((U, lower), -nabla_f)
        if valid_check([d_k]):
            print('dk error') if print_errors else 0
            return v_k, 'computational_error'

        x_k, u_k = np.split(v_k, 2)
        dx, du = np.split(d_k, 2)

        alpha = findAlpha(oracle, x_k, u_k, dx, du, theta=theta, options=line_search_options)
        v_k = v_k + alpha * d_k
        if valid_check([v_k]):
            print('vk error') if print_errors else 0
            return v_k, 'computational_error'


    if np.linalg.norm(oracle.grad(v_k)) ** 2 > norm_0:
        return v_k, 'iterations_exceeded'
    return v_k, 'success'



def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5,
                         tolerance_inner=1e-8, max_iter=100,
                         max_iter_inner=20, t_0=1, gamma=10,
                         c1=1e-4, lasso_duality_gap=None,
                         trace=False, display=False, print_errors=False):
    
    regcoef = reg_coef
    if lasso_duality_gap is None:
        lasso_duality_gap = oracles.lasso_duality_gap
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is increased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for the outer loop stopping criterion:
        Stop the outer loop (which iterates over `k`) when
            `duality_gap(x_k) <= tolerance`
    tolerance_inner : float
        Epsilon value for the inner loop stopping criterion.
        Stop the inner loop (which iterates over `l`) when
            `|| \nabla phi_t(x_k^l) ||_2^2 <= tolerance_inner * \| \nabla \phi_t(x_k) \|_2^2 `
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    lasso_duality_gap : callable object or None.
        If calable the signature is lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)
        Returns duality gap value for esimating the progress of method.
    trace : bool
        If True, the progress information is appended into history dictionary
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every **outer** iteration of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    tau = t_0
    ATA = A.T @ A
    n, d = A.shape
    history = defaultdict(list) if trace else None

    def originalFunc(v):
        if len(v) == d:
            x = v.reshape((d,))
        else:
            x = v[:d].reshape((d,))
        return 0.5 * np.linalg.norm(A @ x) ** 2 + regcoef * np.linalg.norm(x, ord=1)


    def validCheck(list_):
        for val in list_:
            if val is not None:
                if type(val) == np.ndarray:
                    if np.inf in val or True in np.isnan(val):
                        return True
                else:
                    if np.isinf(val) or np.isnan(val):
                        return True
            else:
                return True
        return False


    def debugInfo(k, dgap, fval):
        if display:
            print('ITERATION:', k)
            print('Duality gap: %.5f' % (dgap))
            print('Function value: %.5f' % (fval))
            print()


    def append(fval, time, dgap, x):
        if history is not None:
            history['func'].append(fval)
            history['time'].append(time)
            history['duality_gap'].append(dgap)
            if x.shape[0] <= 2:
                history['x'].append(x)

    x_k = np.copy(x_0)
    u_k = np.copy(u_0)

    start_time = time.time()
    for k in range(max_iter + 1):
        Ax_b = A @ x_k - b
        gap = lasso_duality_gap(x_k, Ax_b, A.T @ Ax_b, b, regcoef)
        fval = originalFunc(np.hstack((x_k, u_k)))
        if validCheck([gap, fval]):
            print('gap error') if print_errors else 0
            return (x_k, u_k), 'computational_error', history

        append(fval, time.time() - start_time, gap, x_k)
        debugInfo(k, gap, fval)

        if gap < tolerance:
            return (x_k, u_k), 'success', history

        if k == max_iter:
            break

        oracle = oracles.LassoOracle(A, b, tau, regcoef, ATA)
        grad = oracle.grad( np.hstack((x_k, u_k)) )
        grad_norm_0 = grad.dot(grad)
        v, message = newtonLasso(oracle, v_0=np.hstack((x_k, u_k)), grad_norm_0=grad_norm_0,
                               tolerance=tolerance_inner, max_iter=max_iter_inner, print_errors=print_errors)

        x_k, u_k = np.split(v, 2)
        if message == 'iterations_exceeded':
            print('newton end') if print_errors else 0
        elif message == 'computational_error':
            print('newton error') if print_errors else 0
            return (x_k, u_k), 'computational_error', history
        
        tau *= gamma

    return (x_k, u_k), 'iterations_exceeded', history
