from collections import defaultdict
import numpy as np
from time import time


def _log_if_needed(should_display: bool, *args, **kwargs):
    if should_display:
        print(*args, **kwargs)


def _fill_history_if_needed_duality(history, func, oracle, x, start_time):
    if history is None:
        return

    history['func'].append(func)
    if hasattr(oracle, 'duality_gap'):
        history['duality_gap'].append(oracle.duality_gap(x))
    history['time'].append(time() - start_time)
    if x.shape[0] <= 2:
        history['x'].append(x)


def check_duality(oracle, x, tolerance):
    if not hasattr(oracle, 'duality_gap'):
        return False
    if oracle.duality_gap(x) <= tolerance:
        return True
    return False


def do_check_result(oracle, x, tolerance, start_time, history=None, display=False):
    _fill_history_if_needed_duality(
        history, oracle.func(x), oracle, x, start_time)
    if hasattr(oracle, 'duality_gap') and not check_duality(oracle, x, tolerance):
        _log_if_needed(display, 'Method couldnt satisfy tolerance after_max_iter',
                       'iterations. Cause: gradient greater then tolerance after loop')
        return x, 'iterations_exceeded', history

    return x, 'success', history


def subgradient_method(oracle, x_0, tolerance=1e-2, max_iter=1000, alpha_0=1,
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

    for i in range(max_iter):
        func = oracle.func(x)
        grad = -oracle.subgrad(x)
        grad_norm = np.linalg.norm(grad)

        if check_duality(oracle, x, tolerance):
            log('Gradient descent done, x =', x, 'f(x) =', func)
            _fill_history_if_needed_duality(
                history, func, oracle, x, start_time)
            return x, 'success', history

        _fill_history_if_needed_duality(history, func, oracle, x, start_time)
        x = x + (alpha_0 / (np.sqrt(i + 1) * grad_norm)) * grad

    return do_check_result(oracle, x, tolerance, start_time, history, display)


def proximal_gradient_method(oracle, x_0, L_0=1.0, tolerance=1e-5,
                             max_iter=1000, trace=False, display=False):
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
    for _ in range(max_iter):
        func = oracle.func(x)
        _fill_history_if_needed_duality(history, func, oracle, x, start_time)
        if check_duality(oracle, x, tolerance):
            log('Gradient descent done, x =', x, 'f(x) =', func)
            return x, 'success', history

        while True:
            y = oracle.prox(x - oracle.grad(x) / l, 1 / l)
            if oracle.func(y) <= (func + oracle.grad(x) @ (y - x)
                                  + l / 2 * (np.linalg.norm(y - x) ** 2)):
                x = y
                break
            l = 2 * l

        l = max(l / 2, L_0)

    return do_check_result(oracle, x, tolerance, start_time, history, display)
