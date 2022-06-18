import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x) @ v


class LassoOracle(BaseSmoothOracle):

    def __init__(self, A, b, tau, regcoef, AtA):
        self.A = A
        self.n = A.shape[0]
        self.d = A.shape[1]
        self.b = b.reshape((self.n,))
        self.tau = tau
        self.regcoef = regcoef
        self.AtA = AtA


    def func(self, v):
        x, u = np.split(v, 2)
        x = x.reshape((self.d,))
        u = u.reshape((self.d,))

        fval = 0.5 * np.linalg.norm(self.A @ x - self.b) ** 2 + self.regcoef * np.sum(u)
        f_tau_val = self.tau * fval - np.sum(np.log(u + x)) - np.sum(np.log(u - x))
        return f_tau_val / self.tau


    def grad(self, v):
        x, u = np.split(v, 2)
        x = x.reshape((self.d,))
        u = u.reshape((self.d,))
        tmp_1 = 1 / (x + u)
        tmp_2 = 1 / (u - x)
        dxf = self.tau * (self.AtA @ x - self.A.T @ self.b) + tmp_2 - tmp_1
        duf = self.tau * np.full(self.d, self.regcoef) - tmp_2 - tmp_1
        return np.hstack((dxf, duf)) / self.tau


    def hess(self, v):
        x, u = np.split(v, 2)
        x = x.reshape((self.d,))
        u = u.reshape((self.d,))

        v_1 = 1 / (u - x) ** 2
        v_2 = 1 / (x + u) ** 2
        M_1 = np.diag(v_1 + v_2)
        M_2 = np.diag(v_2 - v_1)

        H = np.vstack((
            np.hstack((self.tau * self.AtA + M_1, M_2)),
            np.hstack((M_2, M_1))
        ))

        return H / self.tau


def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for 
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    tmp = regcoef / np.linalg.norm(ATAx_b, ord=np.inf)
    mu_x = min(1, tmp) * Ax_b

    return 0.5 * np.linalg.norm(Ax_b) ** 2 + \
           regcoef * np.linalg.norm(x, ord=1) + \
           0.5 * np.linalg.norm(mu_x) ** 2 + np.dot(b, mu_x)
