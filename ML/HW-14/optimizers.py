from importlib_metadata import distribution
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Union, Optional
import sklearn
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from sklearn.gaussian_process import GaussianProcessRegressor
from distributions import BaseDistribution, NumericalDistribution
from scipy.stats import norm


class BaseOptimizer(BaseEstimator, ABC):
    '''
    A base class for all hyperparameter optimizers
    '''

    def __init__(self, estimator: BaseEstimator, param_distributions: Dict[str, BaseDistribution],
                 scoring: Optional[Callable[[np.ndarray, np.ndarray], float]] = None, cv: int = 3, num_runs: int = 100,
                 num_dry_runs: int = 5, num_samples_per_run: int = 20, n_jobs: Optional[int] = None,
                 verbose: bool = False, random_state: Optional[int] = None):
        '''
        Params:
          - estimator: sklearn model instance
          - param_distributions: a dictionary of parameter distributions,
            e.g. param_distributions['num_epochs'] = IntUniformDistribution(100, 200)
          - scoring: sklearn scoring object, see
            https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            if left None estimator must have 'score' attribute
          - cv: number of folds to cross-validate
          - num_runs: number of iterations to fit hyperparameters
          - num_dry_runs: number of dry runs (i.e. random strategy steps) to gather initial statistics
          - num_samples_per_run: number of hyperparameters set to sample each iteration
          - n_jobs: number of parallel processes to fit algorithms
          - verbose: whether to print debugging information (you can configure debug as you wish)
          - random_state: RNG seed to control reproducibility
        '''
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.scoring = scoring
        self.cv = cv
        self.num_runs = num_runs
        self.num_samples_per_run = num_samples_per_run
        self.num_dry_runs = num_dry_runs
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.reset()

    def reset(self):
        '''
        Reset fields used for fitting
        '''
        self.splitter = None
        self.best_score = None
        self.best_params = None
        self.best_estimator = None
        self.params_history = {
            name: np.ndarray([]) for name in self.param_distributions
        }
        self.scores_history = np.array([])

    def sample_params(self) -> Dict[str, np.array]:
        '''
        Sample self.num_samples_per_run set of hyperparameters
        Returns:
          - sampled_params: dict of arrays of parameter samples,
            e.g. sampled_params['num_epochs'] = np.array([178, 112, 155])
        '''
        sampled_params = {}
        for name, distr in self.param_distributions.items():
            sampled_params[name] = distr.sample(
                self.num_samples_per_run)
        return sampled_params

    @abstractmethod
    def select_params(self, params_history: Dict[str, np.ndarray], scores_history: np.ndarray,
                      sampled_params: Dict[str, np.ndarray]) -> Dict[str, Any]:
        '''
        Select new set of parameters according to a specific search strategy
        Params:
          - params_history: list of hyperparameter values from previous interations
          - scores_history: corresponding array of CV scores
          - sampled_params: dict of arrays of parameter samples to select from
        Returns:
          - new_params: a dict of new hyperparameter values
        '''
        msg = f'method \"select_params\" not implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)

    def cross_validate(self, X: np.ndarray, y: Optional[np.ndarray],
                       params: Dict[str, Any]) -> float:
        '''
        Calculate cross-validation score for a set of params
        Consider using estimator.set_params() and sklearn.model_selection.cross_validate()
        Also use self.splitter as a cv parameter in cross_validate
        Params:
          - X: object features
          - y: object labels
          - params: a set of params to score
        Returns:
          - score: mean cross-validation score
        '''
        return np.mean(cross_validate(
            self.estimator.set_params(**params), X, y, cv=self.splitter, scoring=self.scoring, verbose=self.verbose
        )['test_score'])

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> BaseEstimator:
        '''
        Find the best set of hyperparameters with a specific search strategy
        using cross-validation and fit self.best_estimator on whole training set
        Params:
          - X_train: array of train features of shape (num_samples, num_features)
          - y_train: array of train labels of shape (num_samples, )
            if left None task is unsupervised
        Returns:
          - self: (sklearn standard convention)
        '''
        np.random.seed(self.random_state)
        self.reset()
        if y_train is not None and np.issubdtype(y_train.dtype, np.integer):
            self.splitter = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                            random_state=self.random_state)
        else:
            self.splitter = KFold(n_splits=self.cv, shuffle=True,
                                  random_state=self.random_state)

        for _ in range(self.num_runs):
            for _ in range(self.num_samples_per_run):
                run_params = self.select_params(
                    self.params_history, self.scores_history, self.sample_params())
                score = self.cross_validate(X_train, y_train, run_params)
                self.params_history += run_params
                self.scores_history += score
                if self.best_score < score:
                    self.best_score = score
                    self.best_params = run_params

        self.best_estimator = self.estimator.set_params(**self.best_params)
        self.best_estimator.fit(X_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        '''
        Generate a prediction using self.best_estimator
        Params:
          - X_test: array of test features of shape (num_samples, num_features)
        Returns:
          - y_pred: array of test predictions of shape (num_samples, )
        '''
        if self.best_estimator is None:
            raise ValueError('Optimizer not fitted yet')

        return self.best_estimator.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        '''
        Generate a probability prediction using self.best_estimator
        Params:
          - X_test: array of test features of shape (num_samples, num_features)
        Returns:
          - y_pred: array of test probabilities of shape (num_samples, num_classes)
        '''
        if self.best_estimator is None:
            raise ValueError('Optimizer not fitted yet')

        if not hasattr(self.best_estimator, 'predict_proba'):
            raise ValueError('Estimator does not support predict_proba')

        return self.best_estimator.predict_proba(X_test)


class RandomSearchOptimizer(BaseOptimizer):
    '''
    An optimizer implementing random search strategy
    '''

    def __init__(self, estimator: BaseEstimator, param_distributions: Dict[str, BaseDistribution],
                 scoring: Optional[Callable[[np.ndarray, np.ndarray], float]] = None, cv: int = 3, num_runs: int = 100,
                 n_jobs: Optional[int] = None, verbose: bool = False, random_state: Optional[int] = None):
        super().__init__(
            estimator, param_distributions, scoring, cv=cv,
            num_runs=num_runs, num_dry_runs=0, num_samples_per_run=1,
            n_jobs=n_jobs, verbose=verbose, random_state=random_state
        )

    def select_params(self, params_history: Dict[str, np.ndarray], scores_history: np.ndarray,
                      sampled_params: Dict[str, np.ndarray]) -> Dict[str, Any]:
        new_params = {}
        pos = np.random.randint(self.num_samples_per_run)
        for name, param_list in sampled_params.items():
            new_params[name] = param_list[pos]

        return new_params


class GPOptimizer(BaseOptimizer):
    '''
    An optimizer implementing gaussian process strategy
    '''
    @staticmethod
    def calculate_expected_improvement(y_star: float, mu: np.ndarray,
                                       sigma: np.ndarray) -> np.ndarray:
        '''
        Calculate EI values for passed parameters of normal distribution
        hint: consider using scipy.stats.norm
        Params:
          - y_star: optimal (maximal) score value
          - mu: array of mean values of normal distribution of size (num_samples_per_run, )
          - sigma: array of std values of normal distribution of size (num_samples_per_run, )
        Retuns:
          - ei: array of EI values of size (num_samples_per_run, )
        '''
        return (
            sigma / np.sqrt(2 * np.pi) * np.exp(-(((y_star - mu) / sigma) ** 2) / 2) +
            (mu - y_star) * (1 - norm.cdf(y_star, mu, sigma))
        )

    def transform_params(self, params: Dict[str, np.ndarray]) -> tuple:
        res = []
        cnt = 0
        for key in params:
            distr = self.param_distributions[key]
            if hasattr(distr, 'scale'):
                value = params[key]
                res += [distr.scale(value)]
                cnt += 1

        return np.transpose(res), cnt

    def get_categorical_mu_sigma(self, params_history: Dict[str, np.ndarray],
                                 scores_history: np.ndarray,
                                 sampled_params: Dict[str, np.ndarray]) -> tuple:
        res_mus = []
        res_sigmas = []
        y_star = np.max(scores_history)
        for key in params_history:
            distr = self.param_distributions[key]
            if hasattr(distr, 'scale'):
                continue

            mus = []
            sigmas = []
            cat_map = {
                'mu': {},
                'sigma': {},
            }

            param_sampled = sampled_params[key]
            param_history = params_history[key]
            set_cat = set(param_sampled) | set(param_history)

            for cat in set_cat:
                scores_cat_hist = scores_history[param_history == cat]
                if len(scores_cat_hist) == 0:
                    cat_map['mu'][cat] = y_star
                    cat_map['sigma'][cat] = 1.
                else:
                    cat_map['mu'][cat] = scores_cat_hist.mean()
                    cat_map['sigma'][cat] = (
                        1. + ((scores_cat_hist - cat_map['mu'][cat]) ** 2).sum()
                    ) / (1. + len(scores_cat_hist))

            for cat in sampled_params[key]:
                mus += [cat_map['mu'][cat]]
                sigmas += [cat_map['sigma'][cat]]

            res_mus += [mus]
            res_sigmas += [sigmas]

        return np.array(res_mus).mean(axis=0), np.array(res_sigmas).mean(axis=0)

    def get_best_pos(self, params_history: Dict[str, np.ndarray], scores_history: np.ndarray,
                     sampled_params: Dict[str, np.ndarray]) -> int:
        if len(scores_history) < self.num_dry_runs:
            return np.random.randint(self.num_samples_per_run)

        transormed_params, cnt_num = self.transform_params(params_history)
        if cnt_num > 0:
            gpr = GaussianProcessRegressor()
            gpr.fit(transormed_params, scores_history)
            mu, sigma = gpr.predict(self.transform_params(
                sampled_params)[0], return_std=True)

        y_star = np.max(scores_history)

        mu_cat, sigma_cat = self.get_categorical_mu_sigma(
            params_history, scores_history, sampled_params)
        if mu_cat.shape[0] > 0:
            if mu is None:
                mu = mu_cat
                sigma = sigma_cat
            else:
                mu = (mu * cnt_num + mu_cat * (len(params_history) - cnt_num)) / len(params_history)
                sigma = (sigma * cnt_num + sigma_cat * (len(params_history) - cnt_num)) / len(params_history)

        return np.argmax(self.calculate_expected_improvement(y_star, mu, sigma))

    def select_params(self, params_history: Dict[str, np.ndarray], scores_history: np.ndarray,
                      sampled_params: Dict[str, np.ndarray]) -> Dict[str, Any]:
        new_params = {}
        pos = self.get_best_pos(params_history, scores_history, sampled_params)
        for name, param_list in sampled_params.items():
            new_params[name] = param_list[pos]

        return new_params


class TPEOptimizer(BaseOptimizer):
    '''
    An optimizer implementing tree-structured Parzen estimator strategy
    '''

    def __init__(self, estimator: BaseEstimator, param_distributions: Dict[str, BaseDistribution],
                 scoring: Optional[Callable[[np.ndarray, np.ndarray], float]] = None, cv: int = 3, num_runs: int = 100,
                 num_dry_runs: int = 5, num_samples_per_run: int = 20, gamma: float = 0.75,
                 n_jobs: Optional[int] = None, verbose: bool = False, random_state: Optional[int] = None):
        '''
        Params:
          - gamma: scores quantile used for history splitting
        '''
        super().__init__(
            estimator, param_distributions, scoring, cv=cv, num_runs=num_runs,
            num_dry_runs=num_dry_runs, num_samples_per_run=num_samples_per_run,
            n_jobs=n_jobs, verbose=verbose, random_state=random_state
        )
        self.gamma = gamma

    @staticmethod
    def estimate_log_density(scaled_params_history: np.ndarray,
                             scaled_sampled_params: np.ndarray, bandwidth: float):
        '''
        Estimate log density of sampled numerical hyperparameters based on
        numerical hyperparameters history subset
        Params:
          - scaled_params_history: array of scaled numerical hyperparameters history subset
            of size (subset_size, num_numerical_params)
          - scaled_sampled_params: array of scaled sampled numerical hyperparameters
            of size (num_samples_per_run, num_numerical_params)
          - bandwidth: bandwidth for KDE
        Returns:
          - log_density: array of estimated log probabilities of size (num_samples_per_run, )
        '''
        log_density = np.zeros(self.num_samples_per_run)
        # Your code here (⊃｡•́‿•̀｡)⊃
        return log_density

    def select_params(self, params_history: Dict[str, np.ndarray], scores_history: np.ndarray,
                      sampled_params: Dict[str, np.ndarray]) -> Dict[str, Any]:
        new_params = {}
        # Your code here (⊃｡•́‿•̀｡)⊃
        return new_params
