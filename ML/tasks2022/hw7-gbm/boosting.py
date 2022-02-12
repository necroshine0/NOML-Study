from collections import defaultdict

import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
sns.set(style='darkgrid')

from drawings import plot_everything
    
def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        self.loss_derivative2 = lambda y, z: y ** 2 * self.sigmoid(-y * z) * (1 - self.sigmoid(-y * z))

        
    def fit_new_base_model(self, x, y, predictions):
        
        inds = np.random.choice(x.shape[0], int(x.shape[0] * self.subsample), replace=True)
        X_boot, y_boot = x[inds], y[inds]
        S = -self.loss_derivative(y[inds], predictions[inds])
        new_md = self.base_model_class(**self.base_model_params).fit(X_boot, S)
        self.models.append(new_md)
        
        if len(self.gammas) == 0:
            self.gammas.append(1.)
        else:
            new_predictions = self.models[-1].predict(x)
            self.gammas.append(
                self.find_optimal_gamma(y_boot, predictions[inds], new_predictions[inds])
            )

    def fit(self, x_train, y_train, x_valid, y_valid):

        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        breakcount = 0
        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_train)
            valid_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_valid)

            self.history['loss_t'].append(self.loss_fn(y_train, train_predictions))
            self.history['loss_v'].append(self.loss_fn(y_valid, valid_predictions))
            self.history['auc_t'].append(score(self, x_train, y_train))
            self.history['auc_v'].append(score(self, x_valid, y_valid))

            if self.early_stopping_rounds is not None:
                if self.history['loss_v'][-1] >= self.history['loss_v'][-2]:
                    breakcount += 1
                else:
                    breakcount = 0
                    
                if breakcount == self.early_stopping_rounds:
                    break

        if self.plot:
            plot_everything(self.history,
                            y_train, train_predictions,
                            y_valid, valid_predictions
                           )
            
    def predict_proba(self, x):
        probs = 0
        for gamma, model in zip(self.gammas, self.models):
            probs += self.learning_rate * gamma * model.predict(x)
        probs = self.sigmoid(probs)
        return np.array([1. - probs, probs]).T
            
    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        pass
