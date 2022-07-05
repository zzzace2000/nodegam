"""GAM baselines adapted from https://github.com/zzzace2000/GAMs_models/."""

import numpy as np
import pandas as pd

from .utils import get_X_values_counts

eps = np.finfo(np.float32).eps


class MyCommonBase(object):
    @property
    def is_GAM(self):
        """Returns True if it's a GAM."""
        return True

    @property
    def param_distributions(self):
        return None


class MyExtractLogOddsMixin(MyCommonBase):
    """Extract the output from the underlying model.

    It uses the predict function to extract the log odds from the underlying model. It is useful
    to deal with a black-box model that is hard to extract the marginal plot from it. It can then
    use "get_GAM_df(self, x_values_lookup=None)" to extract.

    Requirement:
        the cls needs to implement one of:
        1) predict(): this is for regression model.
        2) predict_proba(): this is for binary classification.
    """

    def _extract_log_odds(self, log_odds):
        split_lens = [len(log_odds[f_name]['x_val']) for f_name in self.feature_names]
        cum_lens = np.cumsum(split_lens)

        all_X = np.ones((1 + np.sum(split_lens), len(self.feature_names)), dtype='float64')

        for f_idx, (feature_name, s_idx, e_idx) in enumerate(
            zip(self.feature_names, [0] + cum_lens[:-1].tolist(), cum_lens)
        ):
            x = log_odds[feature_name]['x_val']

            all_X[(1 + s_idx):(1 + e_idx), f_idx] = x

        if hasattr(self, '_my_predict_logodds'): # for MySpline to use
            score = self._my_predict_logodds(all_X)
        elif hasattr(self, 'predict_proba'):
            prob = self.predict_proba(all_X)[:, 1]

            prob = np.clip(prob, eps, 1. - eps)
            score = np.log(prob) - np.log(1. - prob)
        elif hasattr(self, 'predict'):
            score = self.predict(all_X)
        else:
            raise NotImplementedError('No predict_proba or predict function implemented')

        log_odds['offset']['y_val'] = score[0]
        score[1:] -= score[0]

        ys = np.split(score[1:], np.cumsum(split_lens[:-1]))
        for f_idx, feature_name in enumerate(self.feature_names):
            log_odds[feature_name]['y_val'] = ys[f_idx]

    def get_GAM_df(self, x_values_lookup=None, center=True):
        """Get the GAM dataframe.

        Args:
            x_values_lookup (dict): the unique values of X for each feature. If passed, the outputs
                of the GAM model w.r.t. these x values are extracted. Useful to get a coarser graph
                when there are too many unique values in a feature.
            center (bool): if True, it centers each GAM graph to 0 by moving its mean to the
                intercept term.

        Returns:
            df (pandas dataframe): a GAM dataframe where each row represents a GAM term with the
                inputs x, outputs y, and feature importance.
        """
        assert self.is_GAM, 'Only supports when the model is a GAM'

        # Use the X_values_counts to produce the Xs
        log_odds = {'offset': {'y_val': 0.}}
        for feat_name in self.feature_names:
            all_xs = list(self.X_values_counts[feat_name].keys())

            if x_values_lookup is not None:
                passed_xs = list(x_values_lookup[feat_name])
                if len(all_xs) != len(passed_xs) or np.any(all_xs != passed_xs):
                    all_xs = np.unique(all_xs + passed_xs)

            log_odds[feat_name] = {
                'x_val': np.array(all_xs),
                'y_val': np.zeros(len(all_xs)),
            }

        self._extract_log_odds(log_odds)

        # Centering and importances
        for feat_idx, feat_name in enumerate(self.feature_names):
            v = log_odds[feat_name]

            model_y_val = v['y_val']
            if x_values_lookup is not None:
                model_xs, passed_xs = np.array(list(self.X_values_counts[feat_name].keys())), \
                                      np.array(x_values_lookup[feat_name])

                if len(model_xs) != len(passed_xs) or np.any(model_xs != passed_xs):
                    y_lookup = pd.Series(v['y_val'], v['x_val'])

                    log_odds[feat_name]['x_val'] = passed_xs
                    log_odds[feat_name]['y_val'] = y_lookup[passed_xs].values

                    model_y_val = y_lookup[model_xs].values
            
            # Calculate importance
            weights = np.array(list(self.X_values_counts[feat_name].values()))
            weighted_mean = np.average(model_y_val, weights=weights)
            importance = np.average(np.abs(model_y_val - weighted_mean), weights=weights)
            log_odds[feat_name]['importance'] = importance

            # Centering
            log_odds[feat_name]['y_val'] -= weighted_mean
            log_odds['offset']['y_val'] += weighted_mean

        results = [{
            'feat_name': 'offset',
            'feat_idx': -1,
            'x': None,
            'y': np.full(1, log_odds['offset']['y_val']),
            'importance': -1,
        }]

        for feat_idx, feat_name in enumerate(self.feature_names):
            results.append({
                'feat_name': feat_name,
                'feat_idx': feat_idx,
                'x': log_odds[feat_name]['x_val'],
                'y': np.array(log_odds[feat_name]['y_val']),
                'importance': log_odds[feat_name]['importance'],
            })

        return pd.DataFrame(results)


class MyFitMixin(object):
    """My Mixin to record the feature names and counts when called fit().

    It overides the fit() to record the self.feature_names and self.X_value_counts. It would call
    the super().fit() if there exists such function or just silently returns if not.
    """

    def fit(self, X, y, **kwargs):
        assert isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray)

        self.feature_names = ['f%d' % idx for idx in range(X.shape[1])] \
            if isinstance(X, np.ndarray) else list(X.columns)

        self.X_values_counts = get_X_values_counts(X, self.feature_names)
        return super().fit(X, y, **kwargs)


class MyGAMPlotMixinBase(MyFitMixin, MyExtractLogOddsMixin):
    pass
