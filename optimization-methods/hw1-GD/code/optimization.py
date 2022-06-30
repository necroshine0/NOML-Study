from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
import numpy as np
from utils import get_line_search_tool
from datetime import datetime
import scipy


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.
    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.
    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)

    def debug_info(fval, gnorm):
        print('Func value: %.5f' % (fval))
        print('Grad norm: %.5f' % (gnorm))
        print()

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

    norm_0 = tolerance * np.power(np.linalg.norm(oracle.grad(x_0)), 2)
    x_k = np.copy(x_0)

    times, funcs, norms, xs = [], [], [], []
    xs.append(x_0)
    alpha_k = None
    flag = 1

    if trace:
        times.append(0.0)
        start_time = datetime.now()
        for k in range(max_iter):
            nabla_f = oracle.grad(x_k)
            func_val = oracle.func(x_k)
            norm_val = np.linalg.norm(nabla_f)

            if display:
                debug_info(func_val, norm_val)

            if np.power(norm_val, 2) <= norm_0:
                flag = -1
                break

            d_k = -nabla_f
            alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=alpha_k)
            x_k = x_k + alpha_k * d_k

            if valid_check([nabla_f, func_val, norm_val, d_k, alpha_k, x_k]):
                return x_k, 'computational_error', history

            xs.append(x_k)
            times.append((datetime.now() - start_time).seconds)
            norms.append(norm_val)
            funcs.append(func_val)

        if x_0.shape[0] <= 2:
            history['x'] = xs

        nabla_f = oracle.grad(x_k)
        norms.append(np.linalg.norm(nabla_f))
        history['grad_norm'] = norms
        history['time'] = times
        funcs.append(oracle.func(x_k))
        history['func'] = funcs

    else:
        for k in range(max_iter):
            nabla_f = oracle.grad(x_k)
            func_val = oracle.func(x_k)
            norm_val = np.linalg.norm(nabla_f)

            if display:
                debug_info(func_val, norm_val)

            if np.power(norm_val, 2) <= norm_0:
                flag = -1
                break

            d_k = -nabla_f
            alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=alpha_k)
            x_k = x_k + alpha_k * d_k

            if valid_check([nabla_f, func_val, norm_val, d_k, alpha_k, x_k]):
                return x_k, 'computational_error', history

    if flag > 0 and np.power(np.linalg.norm(oracle.grad(x_k)), 2) > norm_0:
        return x_k, 'iterations_exceeded', history
    return x_k, 'success', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.
    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)

    def debug_info(fval, gnorm):
        print('Func value: %.5f' % (fval))
        print('Grad norm: %.5f' % (gnorm))
        print()

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

    # compute value for norm comparison
    norm_0 = tolerance * np.power(np.linalg.norm(oracle.grad(x_0)), 2)
    x_k = np.copy(x_0)

    times, funcs, norms, xs = [], [], [], []
    xs.append(x_0)
    alpha_k = None
    flag = 1

    if trace:
        times.append(0.0)
        start_time = datetime.now()
        for k in range(max_iter):
            nabla_f = oracle.grad(x_k)
            hess_f = oracle.hess(x_k)
            func_val = oracle.func(x_k)
            norm_val = np.linalg.norm(nabla_f)

            if display:
                debug_info(func_val, norm_val)

            if np.power(norm_val, 2) <= norm_0:
                flag = -1
                break

            try:
                U, lower = scipy.linalg.cho_factor(hess_f)
            except np.linalg.LinAlgError:
                return x_k, 'computational_error', history
            d_k = -scipy.linalg.cho_solve((U, lower), nabla_f.reshape(nabla_f.shape[0]))
            x_k = x_k + d_k
            
            if valid_check([nabla_f, func_val, norm_val, d_k, x_k]):
                return x_k, 'computational_error', history

            xs.append(x_k)
            times.append((datetime.now() - start_time).seconds)
            norms.append(norm_val)
            funcs.append(func_val)

        if x_0.shape[0] <= 2:
            history['x'] = xs

        nabla_f = oracle.grad(x_k)
        norms.append(np.linalg.norm(nabla_f))
        history['grad_norm'] = norms
        history['time'] = times
        funcs.append(oracle.func(x_k))
        history['func'] = funcs

    else:
        for k in range(max_iter):
            nabla_f = oracle.grad(x_k)
            hess_f = oracle.hess(x_k)
            func_val = oracle.func(x_k)
            norm_val = np.linalg.norm(nabla_f)

            if display:
                debug_info(func_val, norm_val)

            if np.power(norm_val, 2) <= norm_0:
                flag = -1
                break

            try:
                U, lower = scipy.linalg.cho_factor(hess_f)
            except np.linalg.LinAlgError:
                return x_k, 'computational_error', history
            d_k = -scipy.linalg.cho_solve((U, lower), nabla_f.reshape(nabla_f.shape[0]))
            x_k = x_k + d_k

            if valid_check([nabla_f, func_val, norm_val, d_k, x_k]):
                return x_k, 'computational_error', history

    if flag > 0 and np.power(np.linalg.norm(oracle.grad(x_k)), 2) > norm_0:
        return x_k, 'iterations_exceeded', history
    return x_k, 'success', history
