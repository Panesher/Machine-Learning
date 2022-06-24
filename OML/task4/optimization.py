from collections import defaultdict
import numpy as np
from time import time
from oracles import BarrierOracle
from utils import get_line_search_tool
from scipy.linalg import cho_solve, cho_factor

def _log_if_needed(should_display: bool, *args, **kwargs):
    if should_display:
        print(*args, **kwargs)


def _fill_history_if_needed(history, func, oracle, x, start_time, line_iter=None):
    if history is None:
        return

    history['func'].append(func)
    if hasattr(oracle, 'duality_gap'):
        history['duality_gap'].append(oracle.duality_gap(x))
    history['time'].append(time() - start_time)
    if x.shape[0] <= 2:
        history['x'].append(x)
    if line_iter is not None and (line_iter or len(history['line_iter'])):
        history['line_iter'].append(line_iter)


def _fill_history_if_needed_duality(history, func, duality_gap, x, start_time):
    if history is None:
        return

    history['func'].append(func)
    if duality_gap is not None:
        history['duality_gap'].append(duality_gap)
    history['time'].append(time() - start_time)
    if x.shape[0] <= 2:
        history['x'].append(x)


def _check_duality(oracle, x, tolerance):
    if not hasattr(oracle, 'duality_gap'):
        return False
    if oracle.duality_gap(x) <= tolerance:
        return True
    return False


def _do_check_result(oracle, x, tolerance, start_time, history=None, display=False, line_iter=None):
    _fill_history_if_needed(
        history, oracle.func(x), oracle, x, start_time, line_iter)
    if hasattr(oracle, 'duality_gap') and not _check_duality(oracle, x, tolerance):
        _log_if_needed(display, 'Method couldnt satisfy tolerance after_max_iter',
                       'iterations. Cause: gradient greater then tolerance after loop')
        return x, 'iterations_exceeded', history

    return x, 'success', history


def subgradient_method(oracle, x_0, tolerance=1e-2, max_iter=1000, alpha_0=1.,
                       display=False, trace=False):
    """
    Subgradient descent method for nonsmooth convex optimization.

    Parameters
    ----------
    oracle : BaseNonsmoothConvexOracle-descendant object
        Oracle with .func() and .subgrad() methods implemented for computing
        function value and its one (arbitrary) subgradient respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    alpha_0 : float
        Initial value for the sequence of step-sizes.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values phi(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    if tolerance <= 0.:
        tolerance = 1e-32

    history = defaultdict(list) if trace else None
    x = np.copy(x_0)
    start_time = time()

    def log(*args, **kwargs):
        _log_if_needed(display, *args, **kwargs)

    log('Starting with x =', x)
    x_min = x
    func_min = oracle.func(x)

    for i in range(max_iter):
        func = oracle.func(x)
        grad = -oracle.subgrad(x)
        grad_norm = np.linalg.norm(grad)
        if func < func_min:
            x_min = x
            func_min = func

        if _check_duality(oracle, x, tolerance):
            log('Gradient descent done, x =', x, 'f(x) =', func)
            _fill_history_if_needed(
                history, func, oracle, x, start_time)
            return x_min, 'success', history

        _fill_history_if_needed(history, func, oracle, x, start_time)
        x = x + (alpha_0 / (np.sqrt(i + 1) * grad_norm)) * grad

    return _do_check_result(oracle, x_min, tolerance, start_time, history, display)


def proximal_gradient_method(oracle, x_0, L_0=1.0, tolerance=1e-5,
                             max_iter=1000, trace=False, is_storing_iter=False, display=False):
    """
    Gradient method for composite optimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented
        for computing function value, its gradient and proximal mapping
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of objective function values phi(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    if tolerance <= 0.:
        tolerance = 1e-32

    history = defaultdict(list) if trace else None
    x = np.copy(x_0)
    start_time = time()

    def log(*args, **kwargs):
        _log_if_needed(display, *args, **kwargs)

    l = L_0
    log('Starting with x =', x)
    line_iter = 0 if is_storing_iter else None
    for _ in range(max_iter):
        func = oracle.func(x)
        _fill_history_if_needed(history, func, oracle,
                                x, start_time, line_iter)
        line_iter = 0 if is_storing_iter else None
        if _check_duality(oracle, x, tolerance):
            log('Gradient descent done, x =', x, 'f(x) =', func)
            return x, 'success', history

        while True:
            y = oracle.prox(x - oracle.grad(x) / l, 1 / l)
            if oracle._f.func(y) <= (oracle._f.func(x) + oracle.grad(x) @ (y - x)
                                     + l / 2 * (np.linalg.norm(y - x) ** 2)):
                x = y
                break
            line_iter = line_iter + 1 if is_storing_iter else None
            l = 2 * l

        l = max(l / 2, L_0)

    return _do_check_result(oracle, x, tolerance, start_time, history, display, line_iter)


def newton(oracle : BarrierOracle, x_0, tolerance=1e-5, max_iter=100,
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
    if tolerance <= 0.:
        tolerance = 1e-32

    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x = np.copy(x_0)
    start_time = time()
    grad_norm_0 = np.linalg.norm(oracle.grad(x))

    def get_alpha(x, d):
        alpha_max = oracle.get_alpha_max(x, d)
        if alpha_max > 1. and line_search_tool.is_correct(oracle, x, d, 1.):
            return 1.

        return line_search_tool.line_search(oracle, x, d, alpha_max)

    def has_nans(*args):
        for arg in args:
            if np.isnan(arg).any() or np.isinf(arg).any():
                return True

        return False

    for _ in range(max_iter):
        func = oracle.func(x)
        grad = oracle.grad(x)
        hess = oracle.hess(x)
        grad_norm = np.linalg.norm(grad)
        if has_nans(func, grad, hess):
            return x, 'computational_error', history

        if grad_norm ** 2 <= tolerance * (grad_norm_0 ** 2):
            _log_if_needed(display, 'Newton descent done, x =',
                           x, 'f(x) =', func)
            _fill_history_if_needed(history, func, grad_norm, x, start_time)
            return x, 'success', history
            # could be saddle point, and we can try to use solve_saddle implemented above

        try:
            c, low = cho_factor(hess)
            if has_nans(c, low):
                return x, 'computational_error', history

            d = cho_solve((c, low), -grad)
        except:
            _log_if_needed(
                display, 'Failure of solving linear system with Hessian matrix')
            return x, 'newton_direction_error', history

        alpha = get_alpha(x, d)
        if alpha is None:
            return x, 'computational_error', history

        _fill_history_if_needed(history, func, grad_norm, x, start_time)

        x = x + alpha * d

    _fill_history_if_needed(history, oracle.func(
        x), np.linalg.norm(oracle.grad(x)), x, start_time)

    return _do_check_result(oracle, x, tolerance, grad_norm_0, history, display)


def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5,
                         tolerance_inner=1e-8, max_iter=100,
                         max_iter_inner=20, t_0=1, gamma=10,
                            c1=1e-4, lasso_duality_gap=None,
                            trace=False, display=False, method=newton, method_kwargs={}):
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
    method: callable method to solve iteration

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
    line_search_options = {'method': 'Armijo', 'c1': c1}
    oracle = BarrierOracle(A, b, reg_coef, t_0)
    if tolerance <= 0.:
        tolerance = 1e-32

    history = defaultdict(list) if trace else None
    start_time = time()
    x = np.copy(x_0)
    u = np.copy(u_0)
    t = t_0
    if lasso_duality_gap is not None:
        x_ = np.append(x, u)
        Ax_b = A @ x - b
        ldg = lasso_duality_gap(
            x, Ax_b, np.transpose(A) @ Ax_b, b, reg_coef)
        _fill_history_if_needed_duality(
            history, oracle.clear_func(x_), ldg, x, start_time)
        if ldg <= tolerance:
            _log_if_needed(display, 'Barrier method done')
            return (x, u), 'success', history

    for _ in range(max_iter):
        x_, msg, _ = method(oracle, np.append(x, u), tolerance_inner, max_iter_inner,
                            line_search_options=line_search_options, display=display, **method_kwargs)
        if msg != 'success':
            return (None, None), msg, history

        x, u = oracle.devide_vec(x_)
        if lasso_duality_gap is not None:
            Ax_b = A @ x - b
            ldg = lasso_duality_gap(
                x, Ax_b, np.transpose(A) @ Ax_b, b, reg_coef)
            _fill_history_if_needed_duality(
                history, oracle.clear_func(x_), ldg, x, start_time)
            if ldg <= tolerance:
                _log_if_needed(display, 'Barrier method done')
                return (x, u), msg, history
        else:
            _fill_history_if_needed_duality(
                history, oracle.clear_func(x_), None, x, start_time)

        t *= gamma
        oracle.set_t(t)

    if lasso_duality_gap is not None:
        return (x, u), 'iterations_exceeded', history

    return (x, u), msg, history
