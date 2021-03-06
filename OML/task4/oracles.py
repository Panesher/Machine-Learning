from cmath import e
import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for smooth function.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func is not implemented.')

    def grad(self, x):
        """
        Computes the gradient vector at point x.
        """
        raise NotImplementedError('Grad is not implemented.')

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


class BaseProxOracle(object):
    """
    Base class for proximal h(x)-part in a composite function f(x) + h(x).
    """

    def func(self, x):
        """
        Computes the value of h(x).
        """
        raise NotImplementedError('Func is not implemented.')

    def prox(self, x, alpha):
        """
        Implementation of proximal mapping.
        prox_{alpha}(x) := argmin_y { 1/(2*alpha) * ||y - x||_2^2 + h(y) }.
        """
        raise NotImplementedError('Prox is not implemented.')


class BaseCompositeOracle(object):
    """
    Base class for the composite function.
    phi(x) := f(x) + h(x), where f is a smooth part, h is a simple part.
    """

    def __init__(self, f, h):
        self._f = f
        self._h = h

    def func(self, x):
        """
        Computes the f(x) + h(x).
        """
        return self._f.func(x) + self._h.func(x)

    def grad(self, x):
        """
        Computes the gradient of f(x).
        """
        return self._f.grad(x)

    def prox(self, x, alpha):
        """
        Computes the proximal mapping.
        """
        return self._h.prox(x, alpha)

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        return None


class BaseNonsmoothConvexOracle(object):
    """
    Base class for implementation of oracle for nonsmooth convex function.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func is not implemented.')

    def subgrad(self, x):
        """
        Computes arbitrary subgradient vector at point x.
        """
        raise NotImplementedError('Subgrad is not implemented.')

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        return None


class LeastSquaresOracle(BaseSmoothOracle):
    """
    Oracle for least-squares regression.
        f(x) = 0.5 ||Ax - b||_2^2
    """

    def __init__(self, matvec_Ax, matvec_ATx, b):
        self._matvec_Ax = matvec_Ax
        self._matvec_ATx = matvec_ATx
        self._b = b

    def func(self, x):
        return 0.5 * np.linalg.norm(self._matvec_Ax(x) - self._b) ** 2

    def grad(self, x):
        return self._matvec_ATx(self._matvec_Ax(x) - self._b)


def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for 
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    f = 0.5 * Ax_b @ Ax_b + regcoef * np.sum(np.abs(x))
    mx = np.max(np.abs(ATAx_b))
    if regcoef == 0:
        mu = np.zeros(len(x))
    elif mx > 0:
        mu = np.min([1, regcoef / mx]) * Ax_b
    else:
        mu = Ax_b
    return f + 0.5 * mu @ mu + b @ mu


class L1RegOracle(BaseProxOracle):
    """
    Oracle for L1-regularizer.
        h(x) = regcoef * ||x||_1.
    """

    def __init__(self, regcoef=1):
        self._regcoef = regcoef

    def func(self, x):
        return self._regcoef * np.sum(np.abs(x))

    def prox(self, x, alpha):
        alpha *= self._regcoef
        prox = np.zeros(x.shape)
        mask = x < -alpha
        prox[mask] = x[mask] + alpha
        mask = x > alpha
        prox[mask] = x[mask] - alpha
        return prox


class LassoProxOracle(BaseCompositeOracle):
    """
    Oracle for 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
        f(x) = 0.5 * ||Ax - b||_2^2 is a smooth part,
        h(x) = regcoef * ||x||_1 is a simple part.
    """

    def __init__(self, f, h):
        if not isinstance(f, LeastSquaresOracle):
            raise ValueError('f must be instance of LeastSquaresOracle')
        if not isinstance(h, L1RegOracle):
            raise ValueError('h must be instance of L1RegOracle')
        super().__init__(f, h)

    def duality_gap(self, x):
        Ax = self._f._matvec_Ax(x)
        Ax_b = Ax - self._f._b
        ATAx_b = self._f._matvec_ATx(Ax - self._f._b)
        return lasso_duality_gap(x, Ax_b, ATAx_b, self._f._b, self._h._regcoef)


class LassoNonsmoothOracle(BaseNonsmoothConvexOracle):
    """
    Oracle for nonsmooth convex function
        0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """

    def __init__(self, matvec_Ax, matvec_ATx, b, regcoef):
        self._least_squares_oracle = LeastSquaresOracle(
            matvec_Ax, matvec_ATx, b)
        self._regcoef = regcoef

    def func(self, x):
        return self._least_squares_oracle.func(x) + self._regcoef * np.sum(np.abs(x))

    def subgrad(self, x):
        reg_grad = np.zeros(x.shape)
        reg_grad[x > 0] = self._regcoef
        reg_grad[x < 0] = -self._regcoef
        return self._least_squares_oracle.grad(x) + reg_grad

    def duality_gap(self, x):
        Ax = self._least_squares_oracle._matvec_Ax(x)
        Ax_b = Ax - self._least_squares_oracle._b
        ATAx_b = self._least_squares_oracle._matvec_ATx(
            Ax - self._least_squares_oracle._b)
        return lasso_duality_gap(x, Ax_b, ATAx_b, self._least_squares_oracle._b, self._regcoef)


def create_lasso_prox_oracle(A, b, regcoef):
    def matvec_Ax(x):
        return A.dot(x)

    def matvec_ATx(x):
        return A.T.dot(x)

    return LassoProxOracle(LeastSquaresOracle(matvec_Ax, matvec_ATx, b),
                           L1RegOracle(regcoef))


def create_lasso_nonsmooth_oracle(A, b, regcoef):
    def matvec_Ax(x):
        return A.dot(x)

    def matvec_ATx(x):
        return A.T.dot(x)

    return LassoNonsmoothOracle(matvec_Ax, matvec_ATx, b, regcoef)


class BarrierOracle(BaseSmoothOracle):
    def __init__(self, A, b, lamda, t, closest_point=0.99):
        self.A = A
        self.b = b
        self.t = t
        self.lamda = lamda
        self.closest_point = closest_point
        try:
            self.n = A.shape[1]
        except:
            try:
                self.n = len(b)
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
        hess_xu = np.diag(upx - umx)
        return np.concatenate((np.concatenate((hess_x, hess_xu), axis=1),
                               np.concatenate((hess_xu, hess_u), axis=1)), axis=0)

    def get_alpha_max(self, x_, d_):
        x, u = self.devide_vec(x_)
        dx, du = self.devide_vec(d_)
        return self.closest_point * np.min([
            np.min((x + u) / (-dx - du), where=(-dx - du > 0), initial=2.),
            np.min((-x + u) / (dx - du), where=(dx - du > 0), initial=2.)
        ])
