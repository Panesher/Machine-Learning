import numpy as np


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
        return self.hess(x).dot(v)


class BarrierOracle(BaseSmoothOracle):
    def __init__(self, A, b, lamda, t, eps=1e-10):
        self.A = A
        self.b = b
        self.t = t
        self.lamda = lamda
        self.eps = eps
        try:
            self.n = b.shape[0]
        except:
            try:
                self.n = len(b.shape)
            except:
                raise ValueError(
                    f'Barrier oracle cant innitialize due incorrect type of b: {type(b)}')

    def devide_vec(self, x):
        return x[:self.n], x[self.n:]

    def set_t(self, t):
        self.t = t

    def clear_func(self, x_):
        x, u = self.devide_vec(x_)
        return 0.5 * (np.linalg.norm(self.A @ x - self.b) ** 2) + self.lamda * np.sum(u)

    def func(self, x_):
        x, u = self.devide_vec(x_)
        return self.t * (self.clear_func(x_)) - np.sum(np.log(u - x) + np.log(x + u))

    def grad(self, x_):
        x, u = self.devide_vec(x_)
        umx = 1 / (u - x)
        upx = 1 / (u + x)
        dx = self.t * np.transpose(self.A) @ (self.A @ x - self.b) + umx - upx
        du = self.t * self.lamda * np.ones(self.n) - umx - upx
        return np.append(dx, du)

    def hess(self, x_):
        """
        Computes the Hessian matrix at point x.
        """
        x, u = self.devide_vec(x_)
        umx = 1 / ((u - x) ** 2)
        upx = 1 / ((u + x) ** 2)
        hess_u = np.diag(umx + upx)
        hess_x = self.t * np.transpose(self.A) @ self.A + hess_u
        hess_xu = np.transpose(np.diag(upx - umx))
        return np.concatenate((np.concatenate((hess_x, hess_xu), axis=1),
                               np.concatenate((hess_xu, hess_u), axis=1)), axis=0)

    def get_alpha_max(self, x_, d_):
        x, u = self.devide_vec(x_)
        dx, du = self.devide_vec(d_)
        alpha_max = 2.
        I_m = np.where(-dx - du > 0)
        if I_m[0].shape[0] > 0:
            alpha_max = np.min(
                [alpha_max, np.min((x[I_m] + u[I_m]) / (-dx[I_m] - du[I_m]))])
        I_p = np.where(dx - du > 0)
        if I_p[0].shape[0] > 0:
            alpha_max = np.min(
                [alpha_max, np.min((-x[I_p] + u[I_p]) / (dx[I_p] - du[I_p]))])
        alpha_max -= self.eps
        return alpha_max


def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for 
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    f = 0.5 * Ax_b @ Ax_b + regcoef * np.sum(x)
    mu = np.min([1, regcoef / np.max(np.abs(ATAx_b))]) * Ax_b
    return f + 0.5 * mu @ mu + b @ mu
