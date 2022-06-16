"""GAM baselines adapted from https://github.com/zzzace2000/GAMs_models/."""


import numpy as np
import pandas as pd

import pygam
from pygam import LogisticGAM, LinearGAM, f, s
from .base import MyGAMPlotMixinBase, MyExtractLogOddsMixin, MyFitMixin
from .EncodingBase import OnehotEncodingRegressorMixin, OnehotEncodingClassifierMixin
from .utils import sigmoid, get_X_values_counts


class MySplineMixin(MyExtractLogOddsMixin):
    def __init__(self, model_cls=LogisticGAM, search=True, search_lam=None, max_iter=500, 
        n_splines=50, fit_binary_feat_as_factor_term=False, **kwargs):
        super().__init__()

        self.model_cls = model_cls
        self.search = search
        self.search_lam = search_lam
        self.max_iter = max_iter
        self.n_splines = n_splines
        self.fit_binary_feat_as_factor_term = fit_binary_feat_as_factor_term
        self.kwargs = kwargs

        if self.search_lam is None:
            self.search_lam = np.logspace(-3, 3, 15)

    def fit(self, X, y, **kwargs):
        # Call the base's fit to create X_values_counts and feature_names
        # super().fit(X, y, **kwargs)

        return self._fit(X, y, **kwargs)

    def _fit(self, X, y, mylam=None, **kwargs):
        if isinstance(X, pd.DataFrame):
            X = X.values

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

    # def extract_log_odds(self, log_odds):
    #     # Take the bias
    #     offset = self.model.coef_[-1]
    #     log_odds['offset']['y_val'] += offset

    #     for feat_idx, feat_name in enumerate(self.feature_names):
    #         x = log_odds[feat_name]['x_val']

    #         # Create a data that only has that feature column not as 0!
    #         X = np.ones((len(x), len(self.feature_names)))
    #         X[:, feat_idx] = x

    #         log_odds[feat_name]['y_val'] = self.model._linear_predictor(X, term=feat_idx)

    def get_lam(self):
        # Now only do 1-number search for lam
        return self.model.lam[0][0]

    def get_params(self, *args, **kwargs):
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

    @property
    def param_distributions(self):
        return {
            'n_splines': [10, 25, 35, 50],
        }


class MySplineLogisticGAMBase(MyFitMixin, MySplineMixin):
    ''' Since it uses the gridsearch as an entry point. Take GAM as an member '''
    def __init__(self, **kwargs):
        super().__init__(model_cls=LogisticGAM, **kwargs)

    def _my_predict_logodds(self, X):
        ''' only used in the base class MyExtractLogOddsMixin '''
        if isinstance(X, pd.DataFrame):
            X = X.values

        logit = self.model._linear_predictor(X)
        return logit

    def predict_proba(self, X):
        logit = self._my_predict_logodds(X)
        # Use stable sigmoid instead of the unstable packages
        prob = sigmoid(logit)

        # Back to sklearn format
        return np.vstack([1. - prob, prob]).T

class MySplineLogisticGAM(OnehotEncodingClassifierMixin, MySplineLogisticGAMBase):
    pass


class MySplineGAMBase(MyFitMixin, MySplineMixin):
    ''' Since it uses the gridsearch as an entry point. Take GAM as an member '''
    def __init__(self, **kwargs):
        super().__init__(model_cls=LinearGAM, **kwargs)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)


class MySplineGAM(OnehotEncodingRegressorMixin, MySplineGAMBase):
    pass


# class MyNewSplineMixin(object):
#     def __init__(self, search=False, max_iter=500, **kwargs):
#         super().__init__(max_iter=max_iter, **kwargs)

#         self.search = search

#     def fit(self, X, y, weights=None, lam=None, **kwargs):
#         if isinstance(X, pd.DataFrame):
#             X = X.values

#         if not self.search:
#             return super().fit(X, y, **kwargs)

#         # do a grid search over here
#         if lam is None:
#             lam = np.logspace(0, 2.5, 10) \
#                 if isinstance(self, LogisticGAM) \
#                 else np.logspace(-3, 2, 10)

#         try:
#             if self.search:
#                 self.gridsearch(X, y, lam=lam, return_scores=True, **kwargs)
#             else:
#                 super().fit(X, y, **kwargs)
#         except np.linalg.LinAlgError as e:
#             print('Get the following error:' , str(e), '\nRetry the grid search')
#             if hasattr(self, 'coef_'):
#                 del self.coef_

#             self.fit(X, y, lam=lam[1:], **kwargs)
#         except pygam.utils.OptimizationError as e:
#             print('Get the following error:' , str(e), '\nRetry fitting with grid search with larger lambda')
#             if hasattr(self, 'coef_'):
#                 del self.coef_
#             self.fit(X, y, lam=lam[1:], **kwargs)

#         return self

#     def extract_log_odds(self, log_odds):
#         # Take the bias
#         offset = self.coef_[-1]
#         log_odds['offset']['y_val'] += offset

#         for feat_idx, feat_name in enumerate(self.feature_names):
#             x = log_odds[feat_name]['x_val']

#             # Create a data that only has that feature column not as 0!
#             X = np.zeros((len(x), len(self.feature_names)))
#             X[:, feat_idx] = x

#             log_odds[feat_name]['y_val'] = self._linear_predictor(X, term=feat_idx)

#     def get_lam(self):
#         # Now only do 1-number search for lam
#         return self.lam[0][0]


# class MyNewLogisticSpline(MyGAMPlotMixinBase, MyNewSplineMixin, LogisticGAM):
#     def predict_proba(self, X):
#         if isinstance(X, pd.DataFrame):
#             X = X.values

#         # Use stable sigmoid instead of the unstable packages
#         from models_utils import sigmoid
#         logit = self.model._linear_predictor(X)
#         prob = sigmoid(logit)

#         # Back to sklearn format
#         return np.vstack([1. - prob, prob]).T

# class MyNewLinearSpline(MyGAMPlotMixinBase, MyNewSplineMixin, LinearGAM):
#     def predict(self, X):
#         if isinstance(X, pd.DataFrame):
#             X = X.values
#         return self.model.predict(X)
