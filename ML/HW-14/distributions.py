import numpy as np
from typing import Union
from abc import ABC, abstractmethod

class BaseDistribution(ABC):
    '''
    A base class for all hyperparameter distributions
    '''
    @abstractmethod
    def sample(self, size: int) -> np.ndarray:
        '''
        Generate "size" samples from the distribution
        Params:
          - size: number of samples
        Returns:
          - sample: generated samples np.array of shape (size, )
        '''
        msg = f'method \"sample\" not implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)


class NumericalDistribution(BaseDistribution, ABC):
    '''
    A base class for all numerical hyperparameter distributions
    '''
    @abstractmethod
    def scale(self, sample: np.array) -> np.ndarray:
        '''
        Scale sample from distribution to [0, 1] uniform range
        Params:
          - sample: np.array sample drawn from the distribution of arbitrary shape
        Returns:
          - scaled_sample: np.array scaled sample of the same shape
        '''
        msg = f'method \"scale\" not implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)


class CategoricalDistribution(BaseDistribution):
    '''
    A categorical distribution over a finite set of elements
    '''
    def __init__(self, categories: Union[list, tuple, np.array]):
        '''
        Params:
          - categories: finite set of categories
          - p: probabilitites of categories (uniform by default)
        '''
        super().__init__()
        self.categories = np.array(categories)
        self.p = np.ones(len(categories)) / len(categories)

    def sample(self, size: int) -> np.ndarray:
        return np.random.choice(self.categories, size=size, p=self.p)


class UniformDistribution(NumericalDistribution):
    '''
    A uniform continuous distribution x ~ U[low, high]
    '''
    def __init__(self, low: float, high: float):
        '''
        Params:
          - low: distribution lower bound
          - high: distribution upper bound
        '''
        super().__init__()
        assert high > low
        self.low = low
        self.high = high

    def sample(self, size: int) -> np.ndarray:
        return np.random.rand(size) * (self.high - self.low) + self.low

    def scale(self, sample: np.array) -> np.ndarray:
        return (sample - self.low) / (self.high - self.low)


class LogUniformDistribution(NumericalDistribution):
    '''
    A log-uniform continuous distribution x ~ exp(U[ln(a), ln(b)])
    '''
    def __init__(self, low: float, high: float):
        '''
        Params:
          - low: distribution lower bound
          - high: distribution upper bound
        '''
        super().__init__()
        assert high > low and low > 0
        self.uniform_distribution = UniformDistribution(np.log(low), np.log(high))

    def sample(self, size: int) -> np.ndarray:
        return np.exp(self.uniform_distribution.sample(size))

    def scale(self, sample: np.array) -> np.ndarray:
        return self.uniform_distribution.scale(np.log(sample))


class IntUniformDistribution(NumericalDistribution):
    '''
    A uniform discrete distribution x ~ U[{low, low + 1, ..., high}]
    '''
    def __init__(self, low: int, high: int):
        '''
        Params:
          - low: distribution lower bound
          - high: distribution upper bound (included in range)
        '''
        super().__init__()
        assert type(low) == int and type(high) == int
        assert high >= low
        self.low = low
        self.high = high

    def sample(self, size: int) -> np.ndarray:
        return np.random.randint(self.low, self.high, size=size)

    def scale(self, sample: np.array) -> np.ndarray:
        return (sample - self.low) / (self.high - self.low)


class IntLogUniformDistribution(NumericalDistribution):
    '''
    A log-uniform discrete distribution x ~ int(exp(U[ln(a), ln(b)]))
    '''
    def __init__(self, low: int, high: int):
        '''
        Params:
          - low: distribution lower bound
          - high: distribution upper bound (included in range)
        '''
        super().__init__()
        assert type(low) == int and type(high) == int
        assert high > low and low > 0
        self.log_distribution = LogUniformDistribution(low, high)

    def sample(self, size: int) -> np.ndarray:
        return self.log_distribution.sample(size).astype(int)

    def scale(self, sample: np.array) -> np.ndarray:
        return self.log_distribution.scale(sample)
