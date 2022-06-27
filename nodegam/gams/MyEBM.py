"""GAM baselines adapted from https://github.com/zzzace2000/GAMs_models/."""


import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

from .EncodingBase import LabelEncodingRegressorMixin, LabelEncodingClassifierMixin, \
    OnehotEncodingRegressorMixin, OnehotEncodingClassifierMixin
from .base import MyCommonBase, MyFitMixin


class MyExplainableBoostingMixin(MyCommonBase):
    def fit(self, X, y):
        result = super().fit(X, y)

        # Fix the feature_name inconsistencies in EBM model
        if 'feature_0' in self.feature_names:
            self.feature_names = ['f%d' % idx for idx in range(len(self.feature_names))]
        return result

    def get_GAM_df(self, x_values_lookup=None):
        ebm_global = self.explain_global()
        overall_importance = ebm_global.data()['scores']

        results = [{
            'feat_name': 'offset',
            'feat_idx': -1,
            'x': None,
            'y': np.full(1, self.intercept_),
            'importance': -1,
        }]

        for feat_idx, feat_name in enumerate(self.feature_names):
            tmp = ebm_global.data(feat_idx)
            if tmp['type'] == 'interaction':
                break

            bins, y = tmp['names'], tmp['scores']

            bins = np.array(bins)[~np.isnan(bins)]
            x = (bins[:-1] + bins[1:]) / 2 # Choose the middle pt as x

            if y is None:
                y = np.zeros(len(x))
            y = np.array(y)

            y_std = None
            if 'lower_bounds' in tmp and tmp['lower_bounds'] is not None:
                y_std = y - np.array(tmp['lower_bounds'])

            # interpolate, since sometimes each split would not have the same unique value of x
            if x_values_lookup is not None:
                x_val = x_values_lookup[feat_name]
                if len(x_val) != len(x) or np.any(x_val != x):
                    raise NotImplementedError('Can not handle the interpolation due to interpret '
                                              'version changes.')

            results.append(dict(
                feat_name=feat_name,
                feat_idx=feat_idx,
                x=x,
                y=y,
                importance=overall_importance[feat_idx],
                **{k: v for k, v in [('y_std', y_std)] if v is not None},
            ))

        return pd.DataFrame(results)


class MyExplainableBoostingClassifier(LabelEncodingClassifierMixin, MyExplainableBoostingMixin,
                                      ExplainableBoostingClassifier):
    pass


class MyExplainableBoostingRegressor(LabelEncodingRegressorMixin, MyExplainableBoostingMixin,
                                     ExplainableBoostingRegressor):
    pass


class MyOnehotExplainableBoostingClassifier(OnehotEncodingClassifierMixin, MyFitMixin,
                                            MyExplainableBoostingMixin,
                                            ExplainableBoostingClassifier):
    pass


class MyOnehotExplainableBoostingRegressor(OnehotEncodingRegressorMixin, MyFitMixin,
                                           MyExplainableBoostingMixin,
                                           ExplainableBoostingRegressor):
    pass