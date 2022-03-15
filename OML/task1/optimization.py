from collections import defaultdict, deque
# Use this for effective implementation of L-BFGS
from matplotlib.pyplot import hist
from scipy.linalg import cho_factor, cho_solve

import numpy as np
import time

from oracles import BaseSmoothOracle

from utils import get_line_search_tool

def is_not_neg_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)


def _log_if_needed(should_display : bool, *args, **kwargs):
    if should_display:
        print(*args, **kwargs)


def _fill_history_if_needed(history, func, grad_norm, x, start_time):
    if history is None:
        return

    history['func'].append(func)
    history['grad_norm'].append(grad_norm)
    history['time'].append(time.time() - start_time)
    if x.shape[0] <= 2:
        history['x'].append(x)


def solve_saddle(oracle, x, line_search_tool, start_time, history=None, display=False):
    func = oracle.func(x)
    hess = oracle.hess(x)
    grad = oracle.grad(x)
    grad_norm = np.linalg.norm(grad)
    if is_not_neg_def(hess):
        _log_if_needed(display, 'Gradient descent done, x =', x, 'f(x) =', func)
        return x, 'success'
    else:
        eigenvalues, eigenvectors = np.linalg.eig(hess)
        neg_vector = eigenvectors[0]
        for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors):
            if eigenvalue < 0:
                neg_vector = eigenvector

        alpha = line_search_tool.line_search(oracle, x, neg_vector)
        if alpha is None:
            return x, 'computational_error'
        _fill_history_if_needed(history, func, grad_norm, x, start_time)
        x = x - alpha * neg_vector
        _log_if_needed(display, 'Step on x =', x, 'func =', func, 'was performed by hessian')
        return x, 'should_continue'


def do_check_result(oracle, x, tolerance, grad_norm_0, history=None, display=False, is_check_hess=False):
    if np.linalg.norm(oracle.grad(x)) ** 2 >= tolerance * (grad_norm_0 ** 2):
        _log_if_needed(display, 'Gradient descent couldnt satisfy tolerance after_max_iter',
                       'iterations. Cause: gradient greater then tolerance after loop')
        return x, 'iterations_exceeded', history

    if is_check_hess and not is_not_neg_def(oracle.hess(x)):
        _log_if_needed(display, 'Gradient descent couldnt satisfy tolerance after',
                       'iterations. Cause: hessian isnt positive after loop.')
        return x, 'iterations_exceeded', history

    return x, 'success', history


def gradient_descent(oracle: BaseSmoothOracle, x_0, tolerance=1e-5, max_iter=10000,
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
    if tolerance <= 0.:
        tolerance = 1e-32

    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x = np.copy(x_0)
    start_time = time.time()
    grad_norm_0 = np.linalg.norm(oracle.grad(x))

    _log_if_needed(display, 'Starting with x =', x)

    for _ in range(max_iter):
        func = oracle.func(x)
        grad = -oracle.grad(x)
        grad_norm = np.linalg.norm(grad)

        # Case gradient is low
        if grad_norm ** 2 <= tolerance * (grad_norm_0 ** 2):
            _log_if_needed(display, 'Gradient descent done, x =', x, 'f(x) =', func)
            _fill_history_if_needed(history, func, grad_norm, x, start_time)
            return x, 'success', history
            # could be saddle point, and we can try to use solve_saddle implemented above

        alpha = line_search_tool.line_search(oracle, x, grad)
        if alpha is None:
            return x, 'computational_error', history
        _fill_history_if_needed(history, func, grad_norm, x, start_time)

        x = x + alpha * grad

    _fill_history_if_needed(history, oracle.func(x), np.linalg.norm(oracle.grad(x)), x, start_time)

    return do_check_result(oracle, x, tolerance, grad_norm_0, history, display)


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
    if tolerance <= 0.:
        tolerance = 1e-32

    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x = np.copy(x_0)
    start_time = time.time()
    grad_norm_0 = np.linalg.norm(oracle.grad(x))

    def get_alpha(x, d):
        if line_search_tool.is_correct(oracle, x, d, 1.):
            return 1.

        return line_search_tool.line_search(oracle, x, d)

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
            _log_if_needed(display, 'Gradient descent done, x =', x, 'f(x) =', func)
            _fill_history_if_needed(history, func, grad_norm, x, start_time)
            return x, 'success', history
            # could be saddle point, and we can try to use solve_saddle implemented above

        try:
            c, low = cho_factor(hess)
            if has_nans(c, low):
                return x, 'computational_error', history

            d = cho_solve((c, low), -grad)
        except:
            _log_if_needed(display, 'Failure of solving linear system with Hessian matrix')
            return x, 'newton_direction_error', history

        alpha = get_alpha(x, d)
        if alpha is None:
            return x, 'computational_error', history

        _fill_history_if_needed(history, func, grad_norm, x, start_time)

        x = x + alpha * d

    _fill_history_if_needed(history, oracle.func(x), np.linalg.norm(oracle.grad(x)), x, start_time)

    return do_check_result(oracle, x, tolerance, grad_norm_0, history, display)
