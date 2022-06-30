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


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/l sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.
    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()
    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size d.
        matvec_ATy : function of y
            Computes matrix-vector product A^Ty, where y is a vector of size l.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """

    def __init__(self, matvec_Ax, matvec_ATy, matmat_ATsA, b, regcoef):
        # A - l x d
        self.matvec_Ax = matvec_Ax
        self.matvec_ATy = matvec_ATy
        self.matmat_ATsA = matmat_ATsA
        self.b = b.reshape((b.shape[0], 1))
        self.regcoef = regcoef
        self.l = b.shape[0]

    def func(self, x):
        x = x.reshape((x.shape[0], 1))
        ones_v = np.ones((self.l, 1))
        ans = np.logaddexp(ones_v * 0, np.multiply(-self.matvec_Ax(x), self.b)).T @ ones_v / self.l + \
               self.regcoef / 2 * np.power(np.linalg.norm(x), 2)
        return ans[0][0]

    def grad(self, x):
        x = x.reshape((x.shape[0], 1))
        vec = self.matvec_ATy(
            np.multiply(scipy.special.expit(np.multiply(self.matvec_Ax(x), self.b)), self.b) - self.b
        )
        return (vec / self.l + self.regcoef * x).reshape(x.shape[0])

    def hess(self, x):
        x = x.reshape((x.shape[0], 1))

        sigm_vec = scipy.special.expit(np.multiply(self.matvec_Ax(x), self.b))
        vec = sigm_vec - np.power(sigm_vec, 2)
        del sigm_vec
        return self.matmat_ATsA(np.multiply(np.power(self.b, 2), vec)) / self.l + self.regcoef * np.eye(x.shape[0])

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        x = x.reshape((x.shape[0], 1))

        sigm_vec = scipy.special.expit(np.multiply(self.matvec_Ax(x), self.b))
        vec = sigm_vec - np.power(sigm_vec, 2)
        del sigm_vec
        s = np.multiply(np.power(self.b, 2), vec)
        z = self.matvec_Ax(v)  # At v

        s = s.reshape((s.shape[0]))
        z = z.reshape((z.shape[0]))

        z = s * z # S (At v) = I s * (At v)
        z = self.matvec_ATy(z)  # A (S (At v))
        return z / self.l + self.regcoef * v

def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    def matvec_Ax(x):
        return A @ x

    def matvec_ATx(x):
        return A.T @ x

    def matmat_ATsA(s):
        return A.T @ np.diag(s.reshape(s.shape[0])) @ A

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def hess_vec_finite_diff(func, x, v, eps=1e-5):
    """
    Returns approximation of the matrix product 'Hessian times vector'
    using finite differences.
    """

    x = x.reshape(x.shape[0])
    I = np.eye(x.shape[0])
    return np.array(
        [
            (func(x + eps * v + eps * I[i]) - \
             func(x + eps * v) - \
             func(x + eps * I[i]) + \
             func(x)) / np.power(eps, 2) \
            for i in range(x.shape[0])
        ])