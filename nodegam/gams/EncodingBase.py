"""GAM baselines adapted from https://github.com/zzzace2000/GAMs_models/."""
from typing import Union

import numpy as np
import pandas as pd


class EncodingBase(object):
    """A base class for handling label or onehot encoding."""
    def get_GAM_df(self, x_values_lookup=None, **kwargs):
        # Make x_values_lookup as onehot encoding!
        if x_values_lookup is not None:
            x_values_lookup = self._convert_x_values_lookup(x_values_lookup)

        # Get the original DF
        df = super().get_GAM_df(x_values_lookup, **kwargs)

        # Used in the bagging. As we already revert it back, no need to do it here.
        if hasattr(self, 'not_revert') and self.not_revert:
            return df

        # change it back to non-onehot encoding df
        return self.revert_dataframe(df)

    def _convert_x_values_lookup(self, x_values_lookup=None):
        raise NotImplementedError()

    def revert_dataframe(self, df):
        raise NotImplementedError()
        

class LabelEncodingFitMixin(EncodingBase):
    def _convert_x_values_lookup(self, x_values_lookup=None):
        need_label_encoding = hasattr(self, 'cat_columns') and len(self.cat_columns) > 0 \
                              and x_values_lookup is not None
        if not need_label_encoding:
            return x_values_lookup
        
        x_values_lookup = x_values_lookup.copy()
        self.cat_x_values_lookup = {c: x_values_lookup[c] for c in self.cat_columns}

        for c in self.cat_columns:
            val = self.cat_to_num_dict[c][x_values_lookup[c]].values
            x_values_lookup[c] = val[~np.isnan(val)]
        return x_values_lookup

    def revert_dataframe(self, df):
        need_label_encoding = hasattr(self, 'cat_columns') and len(self.cat_columns) > 0
        if not need_label_encoding:
            return df

        df_lookup = df.set_index('feat_name')
        for c in self.cat_columns:
            df_lookup.at[c, 'x'] = self.num_to_cat_dict[c][df_lookup.loc[c, 'x']].values

            if not hasattr(self, 'cat_x_values_lookup'):
                continue

            row = df_lookup.loc[c]
            orig_x = self.cat_x_values_lookup[c]
            if len(row.x) == len(orig_x) and np.all(np.array(row.x) == np.array(orig_x)):
                continue

            cat_x = list(row.x) + list(orig_x)
            cat_y = list(row.y) + [0.] * len(orig_x)

            final_x, index = np.unique(cat_x, return_index=True)
            final_y = np.array(cat_y)[index]

            df_lookup.at[c, 'x'] = final_x
            df_lookup.at[c, 'y'] = final_y
            if 'y_std' in df_lookup:
                cat_y_std = list(row.y_std) + [0.] * len(orig_x)
                df_lookup.at[c, 'y_std'] = np.array(cat_y_std)[index]
        
        df = df_lookup.reset_index()
        return df

    def fit(self, X, y, **kwargs):
        if isinstance(X, pd.DataFrame): # in bagging, the coming X is from numpy. Don't transform
            self.my_fit(X, y)
            X = self.my_transform(X)

        return super().fit(X, y, **kwargs)

    def my_fit(self, X, y):
        self.cat_columns = X.columns[X.dtypes == object].values.tolist()
        self.num_to_cat_dict, self.cat_to_num_dict = {}, {}

        for c in self.cat_columns:
            tmp = X[c].astype('category').cat
            self.num_to_cat_dict[c] = pd.Series(tmp.categories)
            self.cat_to_num_dict[c] = pd.Series(range(len(tmp.categories)),
                                                index=tmp.categories.values)
        return X

    def my_transform(self, X):
        X = X.copy()
        for c in self.cat_columns:
            x_value = X[c]
            # If some category values not in the training set, replace them with the most
            # frequent value.
            not_in_category = (~x_value.isin(set(self.cat_to_num_dict[c].index))).copy()
            if not_in_category.any():
                x_value.loc[not_in_category] = x_value.mode().iloc[0]
            val = self.cat_to_num_dict[c][x_value]
            X.loc[:, c] = val.values
        return X


class LabelEncodingClassifierMixin(LabelEncodingFitMixin):
    def predict_proba(self, X):
        # in bagging, the coming X is from numpy. Don't transform
        if isinstance(X, pd.DataFrame) and hasattr(self, 'cat_columns') \
                and len(self.cat_columns) > 0:
            X = self.my_transform(X)
        return super().predict_proba(X)


class LabelEncodingRegressorMixin(LabelEncodingFitMixin):
    def predict(self, X):
        # in bagging, the coming X is from numpy. Don't transform
        if isinstance(X, pd.DataFrame) and hasattr(self, 'cat_columns') \
                and len(self.cat_columns) > 0:
            X = self.my_transform(X)
        return super().predict(X)


class OnehotEncodingFitMixin(EncodingBase):
    def _convert_x_values_lookup(self, x_values_lookup=None):
        need_label_encoding = \
            hasattr(self, 'cat_columns') \
            and len(self.cat_columns) > 0 \
            and x_values_lookup is not None
        if not need_label_encoding:
            return x_values_lookup
        
        x_values_lookup = x_values_lookup.copy()
        # record it
        self.cat_x_values_lookup = {c: x_values_lookup[c] for c in self.cat_columns}

        for c in self.cat_columns:
            del x_values_lookup[c]
        
        for feat_name in self.feature_names:
            if feat_name not in x_values_lookup:
                x_values_lookup[feat_name] = np.array(list(self.X_values_counts[feat_name].keys()))
        
        return x_values_lookup

    def revert_dataframe(self, df):
        """Move the old onehot-encoding df to new non-onehot encoding one."""
        need_label_encoding = hasattr(self, 'cat_columns') and len(self.cat_columns) > 0
        if not need_label_encoding:
            return df

        overall_logic_kept = None

        onehot_features = []
        for c in self.cat_columns:
            logic = df.feat_name.apply(lambda x: x.startswith(c + '_'))
            overall_logic_kept = logic if overall_logic_kept is None \
                else (Union[logic, overall_logic_kept])
            
            filtered = df[logic].copy()
            filtered['new_y_val'] = filtered.y.apply(lambda x: (x[1] - x[0]) if len(x) == 2 else 0.)

            # Record it into the X_values_counts
            if c not in self.X_values_counts:
                values = [self.X_values_counts[f][1] if 1 in self.X_values_counts[f] else 0
                          for f in filtered.feat_name]
                keys = filtered.feat_name.apply(lambda x: x[(len(c)+1):])
                self.X_values_counts[c] = dict(zip(keys, values))

            filtered['proportion'] = list(self.X_values_counts[c].values())

            offset = np.average(filtered.new_y_val.values, weights=filtered.proportion.values)
            filtered.new_y_val -= offset
            
            importance = np.average(np.abs(filtered.new_y_val.values),
                                    weights=filtered.proportion.values)
            
            # Use indep Gaussian to estimate y_std
            if 'y_std' in filtered:
                new_y_std = filtered.y_std.apply(
                    lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2) if len(x) == 2 else 0.)
            
            onehot_features.append(dict(
                feat_name=c, 
                feat_idx=None, 
                x=filtered.feat_name.apply(lambda x: x[(len(c)+1):]).values.tolist(), 
                y=filtered.new_y_val.values.tolist(),
                importance=importance,
                **({'y_std': new_y_std.values.tolist()} if 'y_std' in filtered else {})
            ))

        onehot_df = pd.DataFrame(onehot_features)

        # Handle the case the incoming x_values_lookup having more features than the model
        if hasattr(self, 'cat_x_values_lookup'):
            for idx, c in enumerate(self.cat_columns):
                row = onehot_df.iloc[idx]
                orig_x = self.cat_x_values_lookup[c]
                
                if len(row.x) == len(orig_x) and np.all(np.array(row.x) == np.array(orig_x)):
                    continue

                cat_x = list(row.x) + list(orig_x)
                cat_y = list(row.y) + [0.] * len(orig_x)

                final_x, index = np.unique(cat_x, return_index=True)
                final_y = np.array(cat_y)[index]
                onehot_df.at[idx, 'x'] = final_x
                onehot_df.at[idx, 'y'] = final_y
                if 'y_std' in onehot_df:
                    cat_y_std = list(row.y_std) + [0.] * len(orig_x)
                    onehot_df.at[idx, 'y_std'] = np.array(cat_y_std)[index]
        
        newdf = pd.concat([df[~overall_logic_kept], onehot_df], axis=0)
        newdf.feat_idx = [-1] + list(range(newdf.shape[0] - 1))
        newdf = newdf.reset_index(drop=True)

        return newdf

    def fit(self, X, y, **kwargs):
        if isinstance(X, pd.DataFrame): # in bagging, the coming X is from numpy. Don't transform
            self.cat_columns = X.columns[X.dtypes == object].values.tolist()
            X = pd.get_dummies(X)

        return super().fit(X, y, **kwargs)

    def _transform_X_to_fit_model_feats(self, X):
        """Do one-hot encoding."""
        X = pd.get_dummies(X)
        if len(X.columns) == len(self.feature_names) and np.all(X.columns == self.feature_names):
            return X

        new_X = np.zeros((X.shape[0], len(self.feature_names)))
        for feat_idx, feat_name in enumerate(self.feature_names):
            if feat_name in X:
                new_X[:, feat_idx] = X[feat_name].values

        return pd.DataFrame(new_X, columns=self.feature_names)

    def predict(self, X):
        # in bagging, the coming X is from numpy. Don't transform
        if isinstance(X, pd.DataFrame) and hasattr(self, 'cat_columns') \
                and len(self.cat_columns) > 0:
            X = self._transform_X_to_fit_model_feats(X)

        return super().predict(X)


class OnehotEncodingClassifierMixin(OnehotEncodingFitMixin):
    def predict_proba(self, X):
        # in bagging, the coming X is from numpy. Don't transform
        if isinstance(X, pd.DataFrame) and hasattr(self, 'cat_columns') \
                and len(self.cat_columns) > 0:
            X = self._transform_X_to_fit_model_feats(X)

        return super().predict_proba(X)


class OnehotEncodingRegressorMixin(OnehotEncodingFitMixin):
    pass
