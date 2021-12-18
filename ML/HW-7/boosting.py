from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

sns.set(style='darkgrid')

def Bootstrap(size, train_size):
    train_idxs = np.random.randint(0, size, int(size * train_size))
    return train_idxs, np.array(list(set(np.arange(size)) - set(train_idxs)))

def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class Boosting:
    def __init__(
            self,
            base_model_class = DecisionTreeRegressor,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            is_plotting: bool = False,
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: LearningRate = LearningRate(lambda_=learning_rate)
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.is_plotting: bool = is_plotting

        self.history = {
            'loss': [],
            'auc_roc': []
        }

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        self.loss_derivative2 = lambda y, z: y ** 2 * self.sigmoid(-y * z) * (1 - self.sigmoid(-y * z))


    def find_optimal_gamma(self, y, old_predictions, new_predictions):
        gammas = np.linspace(start=0, stop=self.learning_rate(), num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]


    def fit_new_base_model(self, x, y, predictions):
        self.models.append(self.base_model_class(**self.base_model_params))
        train, _ = Bootstrap(y.shape[0], self.subsample)
        self.models[-1].fit(x[train], -self.loss_derivative(y[train], predictions[train]))
        self.gammas.append(self.find_optimal_gamma(y, predictions, self.models[-1].predict(x)))
        predictions += self.gammas[-1] * self.models[-1].predict(x)


    def is_stop(self, round, valid_predictions, x_valid, y_valid):
        pred = self.gammas[-1] * self.models[-1].predict(x_valid)
        valid_predictions += pred
        self.history['loss'] += [self.loss_fn(y_valid, valid_predictions)]
        self.history['auc_roc'] += [self.score(x_valid, y_valid)]
        return self.early_stopping_rounds is not None and self.early_stopping_rounds <= round and \
            self.history['loss'][-2] < self.history['loss'][-1]


    def plot_validation_scores(self):
        _, axs = plt.subplots(1, len(self.history), figsize=(16, 8))
        for i, (name, val) in enumerate(self.history.items()):
            axs[i].plot(range(len(val)), val)
            axs[i].set_title('Test ' + name + ' over estimators')
            axs[i].set_xlabel('Count of estimators')
            axs[i].set_ylabel(name)
        plt.show()


    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        for round in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)

            if self.is_stop(round, valid_predictions, x_valid, y_valid):
                break

        if self.is_plotting:
            self.plot_validation_scores()


    def predict_proba(self, x):
        predictions = np.zeros((x.shape[0], 2))
        for gamma, model in zip(self.gammas, self.models):
            predictions[:, 1] += gamma * model.predict(x)

        predictions = self.sigmoid(predictions)
        predictions[:, 0] = 1 - predictions[:, 1]
        return predictions


    def predict(self, x):
        return self.predict_proba(x)[:, 1]


    def score(self, x, y):
        return score(self, x, y)


    @property
    def feature_importances_(self):
        res = np.zeros(self.models[0].feature_importances_.shape)
        for model, gamma in zip(self.models, self.gammas):
            res += model.feature_importances_ * gamma
        res = res / res.sum()
        return res

