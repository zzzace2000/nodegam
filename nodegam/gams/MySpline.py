"""GAM baselines adapted from https://github.com/zzzace2000/GAMs_models/."""


import numpy as np
import pandas as pd
import pygam
from pygam import LogisticGAM, LinearGAM, f, s

from .EncodingBase import OnehotEncodingRegressorMixin, OnehotEncodingClassifierMixin
from .base import MyExtractLogOddsMixin, MyFitMixin
from .utils import sigmoid


class MySplineMixin(MyExtractLogOddsMixin):
    def __init__(self, model_cls, search=True, search_lam=None, max_iter=500,
        n_splines=50, fit_binary_feat_as_factor_term=False, cat_features=None, **kwargs):
        super().__init__()

        self.model_cls = model_cls
        self.search = search
        self.search_lam = search_lam
        self.max_iter = max_iter
        self.n_splines = n_splines
        self.fit_binary_feat_as_factor_term = fit_binary_feat_as_factor_term
        self.cat_features = cat_features
        self.kwargs = kwargs

        if self.search_lam is None:
            self.search_lam = np.logspace(-3, 3, 15)

        if not self.fit_binary_feat_as_factor_term:
            self.model = self.model_cls(
                max_iter=self.max_iter, n_splines=self.n_splines, **self.kwargs)
        else:
            formulas = []
            for idx, feat_name in enumerate(self.feature_names):
                num_unique_x = len(self.X_values_counts[feat_name])
                if num_unique_x < 2:
                    continue

                if num_unique_x == 2:
                    formulas.append(f(idx))
                else:
                    formulas.append(s(idx))

            the_formula = formulas[0]
            for term in formulas[1:]:
                the_formula += term

            self.model = self.model_cls(the_formula, max_iter=self.max_iter,
                                        n_splines=self.n_splines, **self.kwargs)

    def fit(self, X, y, **kwargs):
        return self._fit(X, y, **kwargs)

    def _fit(self, X, y, mylam=None, **kwargs):
        if isinstance(X, pd.DataFrame):
            X = X.values

        if not self.search:
            # Just fit the model with this lam
            return self.model.fit(X, y, **kwargs)

        if mylam is None:
            mylam = self.search_lam

        # do a grid search over here
        try:
            print('search range from %f to %f' % (mylam[0], mylam[-1]))
            self.model.gridsearch(X, y, lam=mylam, **kwargs)
        except (np.linalg.LinAlgError, pygam.utils.OptimizationError) as e:
            print('Get the following error:' , str(e), '\nRetry the grid search')
            if hasattr(self.model, 'coef_'):
                del self.model.coef_

            self._fit(X, y, mylam=mylam[1:], **kwargs)

        if not hasattr(self.model, 'statistics_'): # Does not finish the training
            raise Exception('Training fails.')

        return self

    def get_lam(self):
        """Return the lambda penalty."""
        return self.model.lam[0][0]

    def get_params(self, *args, **kwargs):
        """Return the parameters."""
        return dict(
            search=self.search,
            search_lam=self.search_lam,
            **self.model.get_params(*args, **kwargs),
        )

    def set_params(self, *args, **kwargs):
        if 'search' in kwargs:
            self.search = kwargs['search']
        if 'search_lam' in kwargs:
            self.search_lam = kwargs['search_lam']

        self.model.set_params(*args, **kwargs)
        return self


class MySplineLogisticGAMBase(MyFitMixin, MySplineMixin):
    def __init__(self, **kwargs):
        super().__init__(model_cls=LogisticGAM, **kwargs)

    def _my_predict_logodds(self, X):
        """It's used in the base class MyExtractLogOddsMixin."""
        if isinstance(X, pd.DataFrame):
            X = X.values

        logit = self.model._linear_predictor(X)
        return logit

    def predict_proba(self, X):
        """Predict Probability.

        Args:
            X (pandas dataframe): inputs.

        Returns:
            prob (numpy array): the probability of both classes with shape [N, 2].
        """
        logit = self._my_predict_logodds(X)
        # Use stable sigmoid instead of the unstable packages
        prob = sigmoid(logit)

        # Back to sklearn format
        return np.vstack([1. - prob, prob]).T


class MySplineLogisticGAM(OnehotEncodingClassifierMixin, MySplineLogisticGAMBase):
    """Logistic Spline for binary classification with one-hot encoding for cat features.

    Args:
        search (bool): if True, it searches the best lam penalty for the model.
        search_lam (list or numpy array): the range of lam penalty to search. If None, it is
            set to np.linspace(-3, 3, 15).
        max_iter (int): maximum interations to train.
        n_splines (int): number of splines. Default: 50.
        cat_features (list): the column names of the categorical features. Default: None.
    """
    pass


class MySplineGAMBase(MyFitMixin, MySplineMixin):
    def __init__(self, **kwargs):
        super().__init__(model_cls=LinearGAM, **kwargs)

    def predict(self, X):
        """Predict regression target.

        Args:
            X (pandas dataframe): inputs.

        Returns:
            prob (numpy array): the prediction of shape [N].
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)


class MySplineGAM(OnehotEncodingRegressorMixin, MySplineGAMBase):
    """Spline for Regression with one-hot encoding for cat features.

    Args:
        search (bool): if True, it searches the best lam penalty for the model.
        search_lam (list or numpy array): the range of lam penalty to search. If None, it is
            set to np.linspace(-3, 3, 15).
        max_iter (int): maximum interations to train.
        n_splines (int): number of splines. Default: 50.
        cat_features (list): the column names of the categorical features. Default: None.
    """
    pass
