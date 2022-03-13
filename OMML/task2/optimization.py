from time import time
import warnings
from collections import deque, defaultdict
import numpy as np
from numpy.linalg import norm
import scipy
import scipy.sparse
import scipy.optimize
from torch import threshold

from utils import get_line_search_tool


def _log_if_needed(should_display : bool, *args, **kwargs):
    if should_display:
        print(*args, **kwargs)


def has_nans(*args):
    for arg in args:
        if np.isnan(arg).any() or np.isinf(arg).any():
            return True

    return False


def _fill_history_if_needed(history, func, grad_norm, x, start_time):
    if history is None:
        return

    history['func'].append(func)
    history['grad_norm'].append(grad_norm)
    history['time'].append(time() - start_time)
    if x.shape[0] <= 2:
        history['x'].append(x)


def _fill_history_if_needed_v2(history, residual_norm, x, start_time):
    if history is None:
        return

    history['residual_norm'].append(residual_norm)
    history['time'].append(time() - start_time)
    if x.shape[0] <= 2:
        history['x'].append(x)


def do_check_result(oracle, x, tolerance, grad_norm_0, history=None, display=False):
    if np.linalg.norm(oracle.grad(x)) ** 2 >= tolerance * (grad_norm_0 ** 2):
        _log_if_needed(display, 'Gradient descent couldnt satisfy tolerance after_max_iter',
                       'iterations. Cause: gradient greater then tolerance after loop')
        return x, 'iterations_exceeded', history

    return x, 'success', history


def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

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
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    if tolerance <= 0.:
        tolerance = 1e-32

    history = defaultdict(list) if trace else None
    max_iter = int(1e12) if max_iter is None else max_iter
    start_time = time()
    x_k = np.copy(x_0)
    Ax_k = matvec(x_k)
    g_k = Ax_k - b
    d_k = -g_k
    g_k_norm = np.linalg.norm(g_k)
    exit_expr = np.linalg.norm(b) * tolerance

    _log_if_needed(display, 'Starting with x =', x_k)
    _fill_history_if_needed_v2(history, g_k_norm, x_k, start_time)

    for _ in range(max_iter):
        if g_k_norm < exit_expr:
            _log_if_needed(display, 'Done inside max iteration with x =', x_k)
            return x_k, 'success', history

        Ad_k = matvec(d_k)
        koef_x = g_k_norm ** 2 / (Ad_k @ d_k)
        Ax_k = Ax_k + koef_x * Ad_k
        x_k = x_k + koef_x * d_k
        g_k = Ax_k - b
        d_k = -g_k + (g_k @ g_k) / (g_k_norm ** 2) * d_k
        g_k_norm = np.linalg.norm(g_k)
        _fill_history_if_needed_v2(history, g_k_norm, x_k, start_time)

    if np.linalg.norm(g_k) < exit_expr:
        _log_if_needed(display, 'Done at last iteration with x =', x_k)
        return x_k, 'success', history

    _log_if_needed(display, 'Iterations exceeded with x =', x_k)
    return x_k, 'iterations_exceeded', history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden-Fletcher-Goldfarb-Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
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
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    if tolerance <= 0.:
        tolerance = 1e-32

    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x = np.copy(x_0)
    start_time = time()
    grad_norm_0 = np.linalg.norm(oracle.grad(x))
    h_k = []

    def get_alpha(x, d):
        if line_search_tool.is_correct(oracle, x, d, 1.):
            return 1.

        return line_search_tool.line_search(oracle, x, d)
    
    def bfgs_multiply(v, h, gamma):
        if len(h) == 0:
            return gamma * v
        s, y = h[-1]
        z = bfgs_multiply(v - ((v @ s / (y @ s)) * y), h[:-1], gamma)
        return z + ((s @ v - (y @ z)) / (y @ s)) * s

    def lbfgs_direction(h, grad):
        if len(h) == 0:
            return -grad
        s, y = h[-1]
        return bfgs_multiply(-grad, h, y @ s / (y @ y))

    for _ in range(max_iter):
        func = oracle.func(x)
        grad = oracle.grad(x)
        grad_norm = np.linalg.norm(grad)
        if has_nans(func, grad):
            return x, 'computational_error', history

        if grad_norm ** 2 <= tolerance * (grad_norm_0 ** 2):
            _log_if_needed(display, 'Gradient descent done, x =', x, 'f(x) =', func)
            _fill_history_if_needed(history, func, grad_norm, x, start_time)
            return x, 'success', history

        d = lbfgs_direction(h_k, grad)

        alpha = get_alpha(x, d)
        if alpha is None:
            return x, 'computational_error', history

        _fill_history_if_needed(history, func, grad_norm, x, start_time)

        x = x + alpha * d
        h_k.append((alpha * d, oracle.grad(x) - grad))
        if len(h_k) > memory_size:
            h_k = h_k[1:]

    _fill_history_if_needed(history, oracle.func(x), np.linalg.norm(oracle.grad(x)), x, start_time)

    return do_check_result(oracle, x, tolerance, grad_norm_0, history, display)


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500, 
                        line_search_options=None, linear_solver_options=None,
                        display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    linear_solver_options : dict or None
        Dictionary with parameters for newton's system solver.
        NOTE: Specify it by yourself if you need to setup inner CG method.
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
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    if tolerance <= 0.:
        tolerance = 1e-32

    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x = np.copy(x_0)
    start_time = time()
    grad_norm_0 = np.linalg.norm(oracle.grad(x))

    def get_alpha(x, d):
        if line_search_tool.is_correct(oracle, x, d, 1.):
            return 1.

        return line_search_tool.line_search(oracle, x, d)
    
    def d_solver(hess_vec, x, grad, max_iter=int(1e10)):
        def hess_vec_v(v):
            return hess_vec(x, v)

        d = -grad
        threshold_ = .5
        expr = np.min((threshold_, np.sqrt(np.linalg.norm(grad))))
        for _ in range(max_iter):
            d, msg, __ = conjugate_gradients(hess_vec_v, -grad, d, expr, display=display)
            if msg != 'success':
                _log_if_needed(display, 'Problem in conjugate gradients')
                return d, msg
            
            if grad @ d < 0:
                return d, msg

            threshold_ /= 10
            expr = np.min((threshold_, np.sqrt(np.linalg.norm(grad))))

        return d, 'iterations_exceeded'


    for _ in range(max_iter):
        func = oracle.func(x)
        grad = oracle.grad(x)
        grad_norm = np.linalg.norm(grad)
        if has_nans(func, grad):
            return x, 'computational_error', history

        if grad_norm ** 2 <= tolerance * (grad_norm_0 ** 2):
            _log_if_needed(display, 'Gradient descent done, x =', x, 'f(x) =', func)
            _fill_history_if_needed(history, func, grad_norm, x, start_time)
            return x, 'success', history

        d, msg = d_solver(oracle.hess_vec, x, grad)
        if msg != 'success':
            return x, 'computational_error', history

        alpha = get_alpha(x, d)
        if alpha is None:
            return x, 'computational_error', history

        _fill_history_if_needed(history, func, grad_norm, x, start_time)

        x = x + alpha * d

    _fill_history_if_needed(history, oracle.func(x), np.linalg.norm(oracle.grad(x)), x, start_time)

    return do_check_result(oracle, x, tolerance, grad_norm_0, history, display)
