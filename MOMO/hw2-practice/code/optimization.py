from time import time
import warnings
from time import time
import warnings
from collections import deque, defaultdict
import numpy as np
from numpy.linalg import norm
import scipy
import scipy.sparse
import scipy.optimize

from time import time
from utils import get_line_search_tool

from collections import deque
from copy import copy

# from dataclasses import dataclass

# @dataclass
# class gk:
#     prev_: np.ndarray = None
#     next_: np.ndarray = None

def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    b_norm = tolerance * norm(b)

    g_prev = None
    g_next = None

    # Some helpful functions
    def condition(norm_g_k):
        return norm_g_k <= b_norm

    def get_g_k(g_prev, alpha_k, Ad_k):
        return g_prev + alpha_k * Ad_k

    def get_d_next(g_prev, g_next, d_prev):
        return -g_next + np.dot(g_next, g_next) / np.dot(g_prev, g_prev) * d_prev

    def get_x_next(x_prev, g_prev, d_k, Ad_k):
        alpha_k = np.dot(g_prev, g_prev)  / np.dot(Ad_k, d_k)
        return x_prev + alpha_k * d_k, alpha_k

    def debug_info(k, norm):
        if display:
            print('ITERATION:', k)
            print('Residual norm: %.5f' % (norm))

    history = defaultdict(list) if trace else None

    def append(time, norm, x):
        if history is not None:
            history['time'].append(time)
            history['residual_norm'].append(norm)
            if x.shape[0] <= 2:
                history['x'].append(x)

    x_k = np.copy(x_0)
    if max_iter is None:
        max_iter = x_0.shape[0]

    start_time = time()

    g_prev = matvec(x_0) - b
    norm_g_k = norm(g_prev)
    # print('norm:', norm_g_k)
    d_k = np.copy(-g_prev)
    for k in range(max_iter + 1):
        if k == 0:
            append(
                time=time() - start_time,
                norm=norm_g_k,
                x=np.copy(x_k)
            )

            # print('x_k:', x_k)
            debug_info(k, norm_g_k)
            continue

        Ad_k = matvec(d_k)
        x_k, alpha_k = get_x_next(x_k, g_prev, d_k, Ad_k)
        # print('x_k:', x_k)
        g_next = get_g_k(g_prev, alpha_k, Ad_k)
        norm_g_k = norm(g_next)
        # print('norm:', norm_g_k)

        if condition(norm_g_k):
            append(
                time=time() - start_time,
                norm=norm_g_k,
                x=np.copy(x_k)
            )
            # print('OK')
            return x_k, 'success', history

        d_k = get_d_next(g_prev, g_next, d_k)
        g_prev = np.copy(g_next)

        append(
            time=time() - start_time,
            norm=norm_g_k,
            x=np.copy(x_k)
        )

        debug_info(k, norm_g_k)

    return x_k, 'iterations_exceeded', history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):

    def x_next(x_k, alpha_k, d_k):
        return x_k + alpha_k * d_k

    def debug_info(k, norm, fval):
        if display:
            print('ITERATION:', k)
            print('Residual norm: %.5f' % (norm))
            print('Function value: %.5f' % (fval))
            print()

    history = defaultdict(list) if trace else None

    def append(fval=None, time=None, norm=None, x=None):
        if history is not None:
            if fval is not None:
                history['func'].append(fval)
            if time is not None:
                history['time'].append(time)
            if norm is not None:
                history['grad_norm'].append(norm)
            if x is not None and x.shape[0] <= 2:
                history['x'].append(x)

    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    # HISTORY -- deque of tuples
    def BFGS_Recursive(v, HISTORY, gm_0):
        if len(HISTORY) == 0:
            return gm_0 * v

        s, y = HISTORY.pop()
        v_new = v - np.dot(s, v) / np.dot(s, y) * y
        z_new = BFGS_Recursive(v_new, HISTORY, gm_0)
        return z_new + (np.dot(s, v) - np.dot(y, z_new)) / np.dot(s, y) * s


    HISTORY = deque(maxlen=memory_size)

    norm_0 = norm(oracle.grad(x_0))
    start_time = time()
    append(
        fval=oracle.func(x_k),
        time=time() - start_time,
        norm=norm_0,
        x=x_k
    )

    norm_0 **= 2
    norm_0 *= tolerance

    gm_0 = 1.
    for k in range(max_iter):
        d_k = BFGS_Recursive(-oracle.grad(x_k), copy(HISTORY), gm_0)
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)

        s_k = -x_k
        y_k = -oracle.grad(x_k)
        x_k = x_next(x_k, alpha_k, d_k)

        s_k = s_k + x_k
        y_k = y_k + oracle.grad(x_k)
        HISTORY.append((s_k, y_k))
        # s, y = HISTORY[-1]
        gm_0 = np.dot(s_k, y_k) / norm(y_k) ** 2

        grad = oracle.grad(x_k)
        gnorm = norm(grad)
        func_val = oracle.func(x_k)
        append(
            fval=oracle.func(x_k),
            time=time() - start_time,
            norm=gnorm,
            x=x_k
        )

        debug_info(k, gnorm, func_val)

        if gnorm ** 2 <= norm_0:
            return x_k, 'success', history

    return x_k, 'iterations_exceeded', history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500,
                        line_search_options=None, linear_solver_options=None,
                        display=False, trace=False):

    def x_next(x_k, alpha_k, d_k):
        return x_k + alpha_k * d_k

    def debug_info(k, norm, fval):
        if display:
            print('ITERATION:', k)
            print('Residual norm: %.5f' % (norm))
            print('Function value: %.5f' % (fval))
            print()

    history = defaultdict(list) if trace else None

    def append(fval=None, time=None, norm=None, x=None):
        if history is not None:
            if fval is not None:
                history['func'].append(fval)
            if time is not None:
                history['time'].append(time)
            if norm is not None:
                history['grad_norm'].append(norm)
            if x is not None and x.shape[0] <= 2:
                history['x'].append(x)

    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    norm_0 = norm(oracle.grad(x_0))

    start_time = time()
    append(
        fval=oracle.func(x_k),
        time=time() - start_time,
        x=x_k
    )

    norm_0 **= 2
    norm_0 *= tolerance

    for k in range(max_iter):
        def matvec(v):
            return oracle.hess_vec(x_k, v)

        grad = oracle.grad(x_k)
        gnorm = norm(grad)

        append(
            norm=gnorm,
        )
        # print('norms:', gnorm ** 2, norm_0)
        if gnorm ** 2 <= norm_0:
            return x_k, 'success', history

        eta_k = min(0.5, np.sqrt(norm(grad)))
        d_k, _, _ = conjugate_gradients(matvec, b=-grad, x_0=-grad, tolerance=eta_k)
        # print('d_k first:', d_k)

        while np.dot(grad, d_k) >= 0:
            eta_k /= 10
            d_k, _, _ = conjugate_gradients(matvec, b=-grad, x_0=d_k, tolerance=eta_k)
            # print("  dot prod:", np.dot(grad, d_k))

        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)
        x_k = x_next(x_k, alpha_k, d_k)

        func_val = oracle.func(x_k)
        append(
            fval=func_val,
            time=time() - start_time,
            x=x_k
        )

        # print("d_k last:", d_k)

        debug_info(k, gnorm, func_val)

    return x_k, 'iterations_exceeded', history


def gradient_descent(oracle, x_0, tolerance=1e-4, max_iter=10000,
                     line_search_options=None, trace=False, display=False):

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
        start_time = time()
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
            times.append(time() - start_time)
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