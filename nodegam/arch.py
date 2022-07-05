"""The architecture of the models.

This file includes the NODE (ODSTBlock), NODE-GAM (GAMBlock), and NODE-GAM with attention
(GAMAttBlock).
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from . import nn_utils
from .odst import ODST, GAM_ODST, GAMAttODST
from .utils import process_in_chunks, Timer


class ODSTBlock(nn.Sequential):
    """Original NODE model adapted from https://github.com/Qwicen/node."""

    def __init__(self, in_features, num_trees, num_layers, num_classes=1,
                 addi_tree_dim=0,
                 output_dropout=0.0, init_bias=True, add_last_linear=True,
                 last_dropout=0., l2_lambda=0., **kwargs):
        """Neural Oblivious Decision Ensembles (NODE).

        Args:
            in_features: The input dimension of dataset.
            num_trees: How many ODST trees in a layer.
            num_layers: How many layers of trees.
            num_classes: How many classes to predict. It's the output dim.
            addi_tree_dim: Additional dimension for the outputs of each tree. If the value x > 0,
                each tree outputs a (1 + x) dimension of vector.
            output_dropout: The dropout rate on the output of each tree.
            init_bias: If set to True, it adds a trainable bias to the output of the model.
            add_last_linear: If set to True, add a last linear layer to sum outputs of all trees.
            last_dropout: If add_last_layer is True, then it adds a dropout on the weight og last
                linear year.
            l2_lambda: Add a l2 penalty on the outputs of trees.
            kwargs: The kwargs for initializing odst trees.
        """
        layers = self.create_layers(in_features, num_trees, num_layers,
                                    tree_dim=num_classes + addi_tree_dim,
                                    **kwargs)
        super().__init__(*layers)
        self.num_layers, self.num_trees, self.num_classes, self.addi_tree_dim = \
            num_layers, num_trees, num_classes, addi_tree_dim
        self.output_dropout = output_dropout
        self.init_bias = init_bias
        self.add_last_linear = add_last_linear
        self.last_dropout = last_dropout
        self.l2_lambda = l2_lambda

        val = torch.tensor(0.) if num_classes == 1 \
            else torch.full([num_classes], 0., dtype=torch.float32)
        self.bias = nn.Parameter(val, requires_grad=init_bias)

        self.last_w = None
        if add_last_linear or addi_tree_dim < 0:
            # Happens when more outputs than intermediate tree dim
            self.last_w = nn.Parameter(torch.empty(
                num_layers * num_trees * (num_classes + addi_tree_dim),
                num_classes))
            nn.init.xavier_uniform_(self.last_w)

        # Record which params need gradient
        self.named_params_requires_grad = set()
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.named_params_requires_grad.add(name)

    def create_layers(self, in_features, num_trees, num_layers, tree_dim, **kwargs):
        """Create layers of oblivious trees.

        Args:
            in_features: The dim of input features.
            num_trees: The number of trees in a layer.
            num_layers: The number of layers.
            tree_dim: The output dimension of each tree.
            kwargs: The kwargs for initializing odst trees.
        """
        layers = []
        for i in range(num_layers):
            oddt = ODST(in_features, num_trees, tree_dim=tree_dim, **kwargs)
            in_features = in_features + num_trees * tree_dim
            layers.append(oddt)
        return layers

    def forward(self, x, return_outputs_penalty=False, feature_masks=None):
        """Model prediction.

        Args:
            x: The input features.
            return_outputs_penalty: If True, it returns the output l2 penalty.
            feature_masks: Only used in the pretraining. If passed, the outputs of trees belonging
                to masked features (masks==1) is zeroed. This is like dropping out features directly.
        """
        outputs = self.run_with_layers(x)

        num_output_trees = self.num_layers * self.num_trees
        outputs = outputs.view(*outputs.shape[:-1], num_output_trees,
                               self.num_classes + self.addi_tree_dim)

        # During pretraining, we mask the outputs of trees
        if feature_masks is not None:
            assert not self[0].ga2m, 'Not supported for ga2m for now!'
            with torch.no_grad():
                tmp = torch.cat([l.get_feature_selectors() for l in self],
                                dim=1)
                # ^-- [in_features, layers * num_trees, 1]
                op_masks = torch.einsum('bi,ied->bed', feature_masks, tmp)
            outputs = outputs * (1. - op_masks)

        # We can do weighted sum instead of just simple averaging
        if self.last_w is not None:
            last_w = self.last_w
            if self.training and self.last_dropout > 0.:
                last_w = F.dropout(last_w, self.last_dropout)
            result = torch.einsum(
                'bd,dc->bc',
                outputs.reshape(outputs.shape[0], -1),
                last_w
            ).squeeze_(-1)
        else:
            outputs = outputs[..., :self.num_classes]
            # ^--[batch_size, num_trees, num_classes]
            result = outputs.mean(dim=-2).squeeze_(-1)

        result += self.bias

        if return_outputs_penalty:
            # Average over batch, num_outputs_units
            output_penalty = self.calculate_l2_penalty(outputs)
            return result, output_penalty
        return result

    def calculate_l2_penalty(self, outputs):
        """Calculate l2 penalty."""
        return self.l2_lambda * (outputs ** 2).mean()

    def run_with_layers(self, x):
        initial_features = x.shape[-1]

        for layer in self:
            layer_inp = x
            h = layer(layer_inp)
            if self.training and self.output_dropout:
                h = F.dropout(h, self.output_dropout)
            x = torch.cat([x, h], dim=-1)

        outputs = x[..., initial_features:]
        return outputs

    def set_bias(self, y_train):
        """Set the bias term for GAM output as logodds of y.

        It's unnecessary to run since we can just use a learnable bias.
        """

        y_cls, counts = np.unique(y_train, return_counts=True)
        bias = np.log(counts / np.sum(counts))
        if len(bias) == 2:
            bias = bias[1] - bias[0]

        self.bias.data = torch.tensor(bias, dtype=torch.float32)

    def freeze_all_but_lastw(self):
        for name, param in self.named_parameters():
            if param.requires_grad and 'last_w' not in name:
                param.requires_grad = False

    def unfreeze(self):
        for name, param in self.named_parameters():
            if name in self.named_params_requires_grad:
                param.requires_grad = True

    def get_num_trees_assigned_to_each_feature(self):
        """Get the number of trees assigned to each feature per layer.

        It's helpful for logging. Just to see how many trees focus on some features.

        Returns:
            Counts of trees with shape of [num_layers, num_input_features (in_features)].
        """
        if type(self) is ODSTBlock:
            return None

        num_trees = [l.get_num_trees_assigned_to_each_feature() for l in self]
        counts = torch.stack(num_trees)
        return counts

    @classmethod
    def load_model_by_hparams(cls, args, ret_step_callback=False):
        """Helper function to generate a model instance based on hyperparameters.

        Args:
            args: The arguments from argparse. It specifies all hyperparameters.
        """
        if isinstance(args, dict):
            args = argparse.Namespace(**args)
        assert args.arch == 'ODST', 'Wrong arch: ' + args.arch

        model = ODSTBlock(
            in_features=args.in_features,
            num_trees=args.num_trees,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            addi_tree_dim=args.addi_tree_dim + getattr(args, 'data_addi_tree_dim', 0),
            depth=args.depth,
            choice_function=nn_utils.entmax15,
            init_bias=True,
            output_dropout=args.output_dropout,
            last_dropout=args.last_dropout,
            colsample_bytree=args.colsample_bytree,
            bin_function=nn_utils.entmoid15,
            add_last_linear=getattr(args, 'add_last_linear', False),
            l2_lambda=args.l2_lambda,
        )

        if not ret_step_callback:
            return model

        return model, []

    @classmethod
    def add_model_specific_args(cls, parser):
        """Add argparse arguments."""
        parser.add_argument("--colsample_bytree", type=float, default=1.,
                            help="The random proportion of features allowed in each tree. The same "
                                 "argument as in xgboost package. If less than 1, for each tree, "
                                 "it will only choose a fraction of features to train.")
        parser.add_argument("--output_dropout", type=float, default=0.,
                            help='The dropout rate on the output of each tree.')
        parser.add_argument("--last_dropout", type=float, default=0.5,
                            help="The dropout rate on the last linear layer.")
        parser.add_argument("--add_last_linear", type=int, default=1,
                            help="If 1, add the last linear layer to aggregate the output of trees. "
                                 "If 0, do an average of all outputs of trees.")

        for action in parser._actions:
            if action.dest == 'lr':
                # Change the default LR to be 1e-3
                action.default = 1e-3
        return parser

    @classmethod
    def get_model_specific_rs_hparams(cls):
        """Specify the range of hyperparameter search."""
        ch = np.random.choice

        rs_hparams = {
            'seed': dict(short_name='s',
                         gen=lambda args: int(np.random.randint(100))),
            'num_layers': dict(short_name='nl', gen=lambda args: int(ch([2, 3, 4, 5]))),
            'num_trees': dict(short_name='nt',
                              gen=lambda args: int(ch([500, 1000, 2000, 4000])) // args.num_layers),
            'addi_tree_dim': dict(short_name='td', gen=lambda args: int(ch([0]))),
            'depth': dict(short_name='d', gen=lambda args: int(ch([2, 4, 6]))),
            'output_dropout': dict(short_name='od', gen=lambda args: ch([0., 0.1, 0.2])),
            'colsample_bytree': dict(short_name='cs', gen=lambda args: ch([1., 0.5, 0.1])),
            'lr': dict(short_name='lr', gen=lambda args: ch([0.01, 0.005])),
            'l2_lambda': dict(short_name='la', gen=lambda args: float(ch([1e-5, 1e-6, 0.]))),
            'add_last_linear': dict(
                short_name='ll',
                gen=lambda args: (int(ch([0, 1]))),
            ),
            'last_dropout': dict(short_name='ld',
                                 gen=lambda args: (
                                     0. if not args.add_last_linear
                                     else ch([0., 0.1, 0.2, 0.3]))),
        }
        return rs_hparams

    @classmethod
    def add_model_specific_results(cls, results, args):
        """Add or modify the output of csv recording."""
        results['depth'] = args.depth
        return results


class GAMAdditiveMixin(object):
    """All Functions related to extracting GAM and GA2M graphs from the model."""

    def extract_additive_terms(self, X, norm_fn=lambda x: x, y_mu=0., y_std=1., device='cpu',
                               batch_size=1024, tol=1e-3, purify=True):
        """Extract the additive terms in the GAM/GA2M model to plot the graphs.

        To extract the main and interaction terms, it runs the model on all possible input values
        and get the predicted value of each additive term. Then it returns a mapping of x and
        model's outputs y in a dataframe for each term.

        Args:
            X: Input 2d array (pandas). Note that it is the unpreprocessed data.
            norm_fn: The data preprocessing function (E.g. quantile normalization) before feeding
                into the model. Inputs: pandas X. Outputs: preprocessed outputs.
            y_mu, y_std: The outputs of the model will be multiplied by y_std and then shifted by
                y_mu. It's useful in regression problem where target y is normalized to mean 0 and
                std 1. Default: 0, 1.
            device: Use which device to run the model. Default: 'cpu'.
            batch_size: Batch size.
            tol: The tolerance error for the interaction purification that moves mass from
                interactions to mains (see the "purification" of the paper).
            purify: If True, we move all effects of the interactions to main effects.

        Returns:
            A pandas table that records all main and interaction terms. The columns include::
            feat_name: The feature name. E.g. "Hour".
            feat_idx: The feature index. E.g. 2.
            x: The unique values of the feature. E.g. [0.5, 3, 4.7].
            y: The values of the output. E.g. [-0.2, 0.3, 0.5].
            importance: The feature importance. It's calculated as the weighted average of
                the absolute value of y weighted by the counts of each unique value.
            counts: The counts of each unique value in the data. E.g. [20, 10, 3].
        """
        assert self.num_classes == 1, 'Has not support > 2 classes. But should be easy.'
        assert isinstance(X, pd.DataFrame)
        self.eval()

        vals, counts, terms = self._run_and_extract_vals_counts(
            X, device, batch_size, norm_fn=norm_fn, y_mu=y_mu, y_std=y_std)

        if purify:
            # Doing centering: do the pairwise purification
            with Timer('Purify interactions to main effects'):
                self._purify_interactions(vals, counts, tol=tol)

        # Center the main effect
        with Timer('Center main effects'):
            vals[-1] += (self.bias.data.item())
            for t in vals:
                # If it's an interaction term or the bias term, continue.
                if isinstance(t, tuple) or t == -1:
                    continue

                weights = counts[t].values
                avg = np.average(vals[t].values, weights=weights)

                vals[-1] += avg
                vals[t] -= avg

        # Organize data frame. Initialize with the bias term.
        results = [{
            'feat_name': 'offset',
            'feat_idx': -1,
            'x': None,
            'y': np.full(1, vals[-1]),
            'importance': -1,
            'counts': None,
        }]

        with Timer('Construct table'):
            for t in tqdm(vals):
                if t == -1:
                    continue

                if not isinstance(t, tuple):
                    x = vals[t].index.values
                    y = vals[t].values
                    count = counts[t].values
                    tmp = np.argsort(x)
                    x, y, count = x[tmp], y[tmp], count[tmp]
                else:
                    # Make 2d back to 1d
                    # tmp_count = counts[t].stack()
                    tmp = vals[t].stack()
                    tmp_count = counts[t].values.reshape(-1)
                    selected_entry = ((tmp.values != 0) | (tmp_count > 0))
                    tmp = tmp[selected_entry]
                    x = tmp.index.values
                    y = tmp.values
                    count = tmp_count[selected_entry]

                imp = np.average(np.abs(np.array(y)), weights=np.array(count))
                results.append({
                    'feat_name': (X.columns[t] if not isinstance(t, tuple)
                                  else f'{X.columns[t[0]]}_{X.columns[t[1]]}'),
                    'feat_idx': t,
                    'x': x.tolist(),
                    'y': y.tolist(),
                    'importance': imp,
                    'counts': count.tolist(),
                })

            df = pd.DataFrame(results)
            df['tmp'] = df.feat_idx.apply(
                lambda x: x[0] * 1e10 + x[1] * 1e5 if isinstance(x,
                                                                 tuple) else int(
                    x))
            df = df.sort_values('tmp').drop('tmp', axis=1)
            df = df.reset_index(drop=True)
        return df

    def run_with_additive_terms(self, x):
        """Run the models but return the outputs of each main and interaction term.

        Run the models. But instead of summing all the tree outputs, we return the aggregate outputs
        under each main or interaction term for each example.

        Args:
            x: Inputs to the model. A Pytorch Tensor of [batch_size, in_features].
    
        Returns:
            A tensor with shape [batch_size, num_unique_terms, output_dim] where
            'num_unique_terms' is the total number of main and interaction effects, and
            'output_dim' is the output_dim (num_classes). Usually 1.
        """
        outputs = self.run_with_layers(x)
        td = self.num_classes + self.addi_tree_dim
        outputs = outputs.view(*outputs.shape[:-1],
                               self.num_layers * self.num_trees, td)
        # ^--[batch_size, layers*num_trees, tree_dim]

        terms, inv = self.get_additive_terms(return_inverse=True)
        # ^-- (list of unique terms, [layers*num_trees])

        if self.last_w is not None:
            inv = inv.unsqueeze_(-1).expand(-1, td).reshape(-1)
            # ^-- [layers*num_trees*tree_dim] binary features

            new_w = inv.new_zeros(inv.shape[0], len(terms), self.num_classes,
                                  dtype=torch.float32)
            # ^-- [layers*num_trees*tree_dim, uniq_terms, num_classes]
            val = self.last_w.unsqueeze(1).expand(-1, len(terms), -1)
            # ^-- [layers*num_trees*tree_dim, uniq_terms, num_classes]
            idx = inv.unsqueeze_(-1).unsqueeze_(-1).expand(-1, 1,
                                                           self.num_classes)
            # ^-- [layers*num_trees*tree_dim, num_classes]
            new_w.scatter_(1, idx, val)

            result = torch.einsum(
                'bd,duc->buc', outputs.reshape(outputs.shape[0], -1), new_w
            )
        else:
            outputs = outputs[..., :self.num_classes]
            # ^--[batch_size, layers*num_trees, num_classes]

            new_w = inv.new_zeros(inv.shape[0], len(terms), dtype=torch.float32)
            new_w.scatter_(1, inv.unsqueeze_(-1), 1. / inv.shape[0])
            # ^-- [layers*num_trees*tree_dim, uniq_terms]

            result = torch.einsum('bdc,du->buc', outputs, new_w)
        return result

    def _run_and_extract_vals_counts(self, X, device, batch_size, norm_fn=lambda x: x, y_mu=0.,
                                     y_std=1.):
        """Run the models and return the value of model's outputs and the counts.

        It runs the model on all inputs X, and returns the models's output and the counts of each
        input value for each term.

        Args:
            X: Input 2d array (pandas). Note that it is the unnormalized data.
            norm_fn: The data preprocessing function (E.g. quantile normalization) before feeding
                into the model. Inputs: pandas X. Outputs: preprocessed outputs.
            y_mu, y_std: The outputs of the model will be multiplied by y_std and then shifted by
                y_mu. It's useful in regression problem where target y is normalized to mean 0 and
                std 1. Default: 0, 1.
            device: Use which device to run the model. Default: 'cpu'.
            batch_size: Batch size.
            tol: The tolerance error for the interaction purification that moves mass from
                interactions to mains (see the "purification" of the paper).
            purify: If True, we move all effects of the interactions to main effects.

        Returns:
            vals (dict of dict): A dict that has keys as feature index and value as another dict
                that maps the unique value of input X to the output of the model. For example, if a
                model learns 2 main effects for features 1 and 2, and an interaction term between
                features 1 and 2, we could have::
                {1: {0: -0.2, 1: 0.3, 2: 1},
                 2: {1: 0.3, 2: -0.5},
                 (1, 2): {(0, 1): 1, (0, 2): 0.3, (1, 1): -1, (1, 2): -0.3, (2, 1): 0, (2, 2): 1}}.
            counts (dict of dict): Same format as `vals` but the values are the counts in the data.
                It has a dict that has keys as feature index and value as another dict that maps
                the unique value of input X to the counts of occurence in the data. For example::
                {1: {0: 10, 1: 100, 2: 90},
                 2: {1: 80, 2: 120},
                 (1, 2): {(0, 1): 10, (0, 2): 50, (1, 1): 100, (1, 2): 10, (2, 1): 20, (2, 2): 10}}.
            terms (list): all the main and interaction terms. E.g. [1, 2, (2, 3)].
        """
        with Timer('Run values through model'), torch.no_grad():
            results = self._run_vals_with_additive_term_with_batch(
                X, device, batch_size, norm_fn=norm_fn, y_std=y_std)

        # Extract all additive term results
        with Timer('Extract values'):
            vals, counts, terms = self._extract_vals_counts(results, X)
            vals[-1] = y_mu
        return vals, counts, terms

    def _run_vals_with_additive_term_with_batch(self, X, device, batch_size, norm_fn=lambda x: x,
                                                y_std=1.):
        """Run the models with additive terms using mini-batch.

        It calls self.run_with_additive_terms() with mini-batch.

        Args:
            X: Input 2d array (pandas). Note that it is the unnormalized data.
            device: Use which device to run the model. Default: 'cpu'.
            batch_size: Batch size.
            norm_fn: The data preprocessing function (E.g. quantile normalization) before feeding
                into the model. Inputs: pandas X. Outputs: preprocessed outputs.
            y_std: The outputs of the model will be multiplied by y_std. It's useful in regression
                problem where target y is normalized to std 1. Default: 1.

        Returns:
            results (numpy array): The model's output of each term. A numpy tensor of shape
                [num_data, num_unique_terms, output_dim] where 'num_unique_terms' is the total
                number of main and interaction effects, and 'output_dim' is the output_dim
                (num_classes). Usually 1.
        """

        results = process_in_chunks(
            lambda x: self.run_with_additive_terms(torch.tensor(norm_fn(x), device=device)),
            X.values, batch_size=batch_size)
        results = results.cpu().numpy()
        results = (results * y_std)
        return results

    def _extract_vals_counts(self, results, X):
        """Extracts the values and counts based on the outputs of models with additive terms.

        Args:
            results: The model's outputs of self._run_vals_with_additive_term_with_batch. It's a
                numpy tensor of shape [num_data, num_unique_terms, output_dim] that represents the
                model's output of each data on each additive term.
            X: The inputs of the data.

        Returns:
            vals (dict of dict): A dict that has keys as feature index and value as another dict
                that maps the unique value of input X to the output of the model. For example, if a
                model learns 2 main effects for features 1 and 2, and an interaction term between
                features 1 and 2, we could have::
                {1: {0: -0.2, 1: 0.3, 2: 1},
                 2: {1: 0.3, 2: -0.5},
                 (1, 2): {(0, 1): 1, (0, 2): 0.3, (1, 1): -1, (1, 2): -0.3, (2, 1): 0, (2, 2): 1}}.
            counts (dict of dict): Same format as `vals` but the values are the counts in the data.
                It has a dict that has keys as feature index and value as another dict that maps
                the unique value of input X to the counts of occurence in the data. For example::
                {1: {0: 10, 1: 100, 2: 90},
                 2: {1: 80, 2: 120},
                 (1, 2): {(0, 1): 10, (0, 2): 50, (1, 1): 100, (1, 2): 10, (2, 1): 20, (2, 2): 10}}.
            terms (list): all the main and interaction terms. E.g. [1, 2, (2, 3)].
        """
        terms = self.get_additive_terms()

        vals, counts = {}, {}
        for idx, t in enumerate(tqdm(terms)):
            if not isinstance(t, tuple): # main effect term
                index = X.iloc[:, t]
                scores = pd.Series(results[:, idx, 0], index=index)

                tmp = scores.groupby(level=0).agg(['count', 'first'])
                vals[t] = tmp['first']
                counts[t] = tmp['count']
            else:
                tmp = pd.Series(results[:, idx, 0],
                                index=pd.MultiIndex.from_frame(X.iloc[:, list(t)]))

                # One groupby to return both vals and counts
                tmp2 = tmp.groupby(level=[0, 1]).agg(['count', 'first'])

                the_vals = tmp2['first']
                the_counts = tmp2['count']

                vals[t] = the_vals.unstack(level=-1).fillna(0.)
                counts[t] = the_counts.unstack(level=-1).fillna(0).astype(int)

        # For each interaction tuple (i, j), initialize the main effect term i and j since they
        # will have some values during the purification.
        for t in terms:
            if not isinstance(t, tuple):
                continue

            for i in t:
                if i in vals:
                    continue
                a = X.iloc[:, i]
                the_counts = a.groupby(a).agg(['count'])
                counts[i] = the_counts['count']
                vals[i] = the_counts['count'].copy()
                vals[i][:] = 0.

        return vals, counts, terms

    def _purify_interactions(self, vals, counts, tol=1e-3):
        """Purify the interaction term to move the mass from interaction to the main effect.

        See the Supp. D in the paper for details. It modifies the vals in-place for arguments vals.

        Args:
            vals (dict of dict): A dict that has keys as feature index and value as another dict
                that maps the unique value of input X to the output of the model. For example, if a
                model learns 2 main effects for features 1 and 2, and an interaction term between
                features 1 and 2, we could have::
                {1: {0: -0.2, 1: 0.3, 2: 1},
                 2: {1: 0.3, 2: -0.5},
                 (1, 2): {(0, 1): 1, (0, 2): 0.3, (1, 1): -1, (1, 2): -0.3, (2, 1): 0, (2, 2): 1}}.
            counts (dict of dict): Same format as `vals` but the values are the counts in the data.
                It has a dict that has keys as feature index and value as another dict that maps
                the unique value of input X to the counts of occurence in the data. For example::
                {1: {0: 10, 1: 100, 2: 90},
                 2: {1: 80, 2: 120},
                 (1, 2): {(0, 1): 10, (0, 2): 50, (1, 1): 100, (1, 2): 10, (2, 1): 20, (2, 2): 10}}.
        """
        for t in vals:
            # If it's not an interaction term, continue.
            if not isinstance(t, tuple):
                continue

            # Continue purify the interactions until the purified average value is smaller than tol.
            biggest_epsilon = np.inf
            while biggest_epsilon > tol:
                biggest_epsilon = -np.inf

                avg = (vals[t] * counts[t]).sum(axis=1).values / counts[t].sum(axis=1).values
                if np.max(np.abs(avg)) > biggest_epsilon:
                    biggest_epsilon = np.max(np.abs(avg))

                vals[t] -= avg.reshape(-1, 1)
                vals[t[0]] += avg

                avg = (vals[t] * counts[t]).sum(axis=0).values / counts[
                    t].sum(axis=0).values
                if np.max(np.abs(avg)) > biggest_epsilon:
                    biggest_epsilon = np.max(np.abs(avg))

                vals[t] -= avg.reshape(1, -1)
                vals[t[1]] += avg

    def get_additive_terms(self, return_inverse=False):
        """Get the additive terms in the GAM/GA2M model.

        It returns all the main and interaction effects in the NodeGAM.

        Args:
            return_inverse (bool): If True, it returns the map back from each additive term to the
                index of trees. It's useful to check which tree focuses on which feature set.

        Returns:
            tuple_terms (list): A list of integer or tuple that represents all the additive terms it
                learns. E.g. [2, 4, (2, 3), (1, 4)].
        """
        fs = torch.cat([l.get_feature_selectors() for l in self], dim=1).sum(dim=-1)
        fs[fs > 0.] = 1.
        # ^-- [in_features, layers*num_trees] binary features

        result = torch.unique(fs, dim=1, sorted=True, return_inverse=return_inverse)
        # ^-- ([in_features, uniq_terms], [layers*num_trees])

        terms = result
        if isinstance(result, tuple):  # return inverse=True
            terms = result[0]

        # To make additive terms human-readable, it transforms the one-hot vector into an integer,
        # and a 2-hot vector (interaction) into a tuple of integer.
        tuple_terms = self.convert_onehot_vector_to_integers(terms)

        if isinstance(result, tuple):
            return tuple_terms, result[1]
        return tuple_terms

    def convert_onehot_vector_to_integers(self, terms):
        """Make onehot or multi-hot vectors into a list of integers or tuple.

        Args:
            terms (Pytorch tensor): a one-hot matrix with each column has only one entry as 1.
                Shape: [in_features, uniq_GAM_terms].

        Returns:
            tuple_terms (list): A list of integers or tuples of all the GAM terms.
        """
        r_idx, c_idx = torch.nonzero(terms, as_tuple=True)
        tuple_terms = []
        for c in range(terms.shape[1]):
            n_interaction = (c_idx == c).sum()

            if n_interaction > 2:
                print(
                    f'WARNING: it is not a GA2M with a {n_interaction}-way term. '
                    f'Ignore this term.')
                continue
            if n_interaction == 1:
                tuple_terms.append(int(r_idx[c_idx == c].item()))
            elif n_interaction == 2:
                tuple_terms.append(tuple(r_idx[c_idx == c][:2].cpu().numpy()))
        return tuple_terms


class GAMBlock(GAMAdditiveMixin, ODSTBlock):
    """Node-GAM model."""

    def __init__(self, in_features, num_trees, num_layers, num_classes=1, addi_tree_dim=0,
                 output_dropout=0.0, init_bias=True, add_last_linear=True, last_dropout=0.,
                 l2_lambda=0., l2_interactions=0., l1_interactions=0., **kwargs):
        """Initialization of Node-GAM.

        Args:
            in_features: The input dimension of dataset.
            num_trees: How many ODST trees in a layer.
            num_layers: How many layers of trees.
            num_classes: How many classes to predict. It's the output dim.
            addi_tree_dim: Additional dimension for the outputs of each tree. If the value x > 0,
                each tree outputs a (1 + x) dimension of vector.
            output_dropout: The dropout rate on the output of each tree.
            init_bias: If set to True, it adds a trainable bias to the output of the model.
            add_last_linear: If set to True, add a last linear layer to sum outputs of all trees.
            last_dropout: If add_last_layer is True, it adds a dropout on the weight og last
                linear year.
            l2_lambda: Add a l2 penalty on the outputs of trees.
            l2_interactions: Penalize the l2 magnitude of the output of trees that have
                pairwise interactions. Default: 0.
            l1_interactions: Penalize the l1 magnitude of the output of trees that have
                pairwise interactions. Default: 0.
            kwargs (dict): The arguments for underlying GAM ODST trees.
        """
        super().__init__(
            in_features=in_features,
            num_trees=num_trees,
            num_layers=num_layers,
            num_classes=num_classes,
            addi_tree_dim=addi_tree_dim,
            output_dropout=output_dropout,
            init_bias=init_bias,
            add_last_linear=add_last_linear,
            last_dropout=last_dropout,
            l2_lambda=l2_lambda,
            **kwargs)
        self.l2_interactions = l2_interactions
        self.l1_interactions = l1_interactions

        self.inv_is_interaction = None

    def create_layers(self, in_features, num_trees, num_layers, tree_dim, **kwargs):
        """Create layers.

        Args:
            in_features: The input dimension (feature).
            num_trees: Number of trees in a layer.
            num_layers: Number of layers.
            tree_dim: The dimension of the tree's output. Usually equal to num of classes.
            kwargs (dict): The arguments for underlying GAM ODST trees.
        """
        layers = []
        for i in range(num_layers):
            # Last layer only has num_classes dim
            oddt = GAM_ODST(in_features, num_trees, tree_dim=tree_dim, **kwargs)
            layers.append(oddt)
        return layers

    def calculate_l2_penalty(self, outputs):
        """Calculate the penalty of the trees' outputs.

        It helps regularize the model.

        Args:
            outputs: The outputs of trees. A tensor of shape [batch_size, num_trees, tree_dim].
        """
        # Normal L2 weight decay on outputs
        penalty = super().calculate_l2_penalty(outputs)

        # If trees are still learning which features to take, skip the interaction penalty
        if not self[0].choice_function.is_deterministic:
            return penalty

        # Search and cache which term is interaction
        if self.inv_is_interaction is None:
            with torch.no_grad():
                terms, inv = self.get_additive_terms(return_inverse=True)
            idx_is_interactions = [i for i, t in enumerate(terms) if isinstance(t, tuple)]
            if len(idx_is_interactions) == 0:
                return penalty

            inv_is_interaction = inv.new_zeros(*inv.shape, dtype=torch.bool)
            for idx in idx_is_interactions:
                inv_is_interaction |= (inv == idx)
            self.inv_is_interaction = inv_is_interaction

        outputs_interactions = outputs[:, self.inv_is_interaction, :]
        if self.l2_interactions > 0.:
            penalty += self.l2_interactions * torch.mean(
                outputs_interactions ** 2)
        if self.l1_interactions > 0.:
            penalty += self.l1_interactions * torch.mean(
                torch.abs(outputs_interactions))

        return penalty

    def run_with_layers(self, x, return_fs=False):
        """Run the examples through the layers of trees.

        Args:
            x: The input tensor of shape [batch_size, in_features].
            return_fs: If True, it returns the feature selectors of each tree.

        Returns:
            outputs: The trees' outputs [batch_size, num_trees, tree_dim].
            prev_feature_selectors: Only returns when return_fs is True, this returns the feature
                selector of each ODST tree of shape [in_features, num_trees, tree_depth].
        """
        initial_features = x.shape[-1]
        prev_feature_selectors = None
        for layer in self:
            layer_inp = x
            h, feature_selectors = layer(
                layer_inp, prev_feature_selectors=prev_feature_selectors,
                return_feature_selectors=True)
            if self.training and self.output_dropout:
                h = F.dropout(h, self.output_dropout)
            x = torch.cat([x, h], dim=-1)

            prev_feature_selectors = feature_selectors \
                if prev_feature_selectors is None \
                else torch.cat([prev_feature_selectors, feature_selectors], dim=1)

        outputs = x[..., initial_features:]
        if return_fs:
            return outputs, prev_feature_selectors
        return outputs

    @classmethod
    def load_model_by_hparams(cls, args, ret_step_callback=False):
        """Load the initialized model by its hyperparameters.

        Args:
            args: The arguments of the model. Can passed in a dictionary or a namespace.
        """
        if isinstance(args, dict):
            args = argparse.Namespace(**args)

        assert args.arch in ['GAM', 'GAMAtt'], 'Wrong arch: ' + args.arch

        # If it's not GA2M, make sure the l2/l1 interaction is set to 0.
        if not getattr(args, 'ga2m', 0):
            assert getattr(args, 'l2_interactions', 0.) == 0., \
                'No L2 penalty should be set for interaction'
            assert getattr(args, 'l1_interactions', 0.) == 0., \
                'No L1 penalty should be set for interaction'

        # Initialize choice function (default entmax)
        choice_fn = getattr(nn_utils, args.choice_fn)(
            max_temp=1., min_temp=args.min_temp, steps=args.anneal_steps)

        # Temperature annealing for entmoid
        bin_function = nn_utils.entmoid15
        kwargs = dict(
            in_features=args.in_features,
            num_trees=args.num_trees,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            addi_tree_dim=args.addi_tree_dim + getattr(args, 'data_addi_tree_dim', 0),
            depth=args.depth,
            choice_function=choice_fn,
            bin_function=bin_function,
            output_dropout=args.output_dropout,
            last_dropout=getattr(args, 'last_dropout', 0.),
            colsample_bytree=args.colsample_bytree,
            selectors_detach=args.selectors_detach,
            init_bias=True,
            add_last_linear=getattr(args, 'add_last_linear', False),
            save_memory=getattr(args, 'save_memory', 0),
            ga2m=getattr(args, 'ga2m', 0),
            l2_lambda=args.l2_lambda,
            l2_interactions=getattr(args, 'l2_interactions', 0.),
        )

        if args.arch in ['GAMAtt'] and 'dim_att' in args:
            kwargs['dim_att'] = args.dim_att

        model = cls(**kwargs)
        if not ret_step_callback:
            return model

        step_callbacks = [choice_fn.temp_step_callback]
        return model, step_callbacks

    @classmethod
    def add_model_specific_args(cls, parser):
        """Add argparse arguments."""
        parser = super().add_model_specific_args(parser)
        parser.add_argument("--min_temp", type=float, default=1e-2, help="The min temperature.")
        parser.add_argument("--anneal_steps", type=int, default=4000,
                            help="Temp annealing schedule decays from max to min temp in 4k steps.")

        parser.add_argument("--choice_fn", default='EM15Temp',
                            help="Use SoftMax, GumbelSoftMax or EntMax on the choice function of "
                                 "trees.",
                            choices=['GSMTemp', 'SMTemp', 'EM15Temp'])

        parser.add_argument("--selectors_detach", type=int, default=0,
                            help="if 1, the selector will be detached before passing into the "
                                 "next layer. This will save GPU memory in the large dataset "
                                 "(e.g. Epsilon).")

        # Use GA2M
        parser.add_argument("--ga2m", type=int, default=0, help="If 1, train a GA2M. If 0, GAM.")
        parser.add_argument("--l2_interactions", type=float, default=0.,
                            help="Add L2 penalty on the interactions to decrease the learned "
                                 "interaction effects. It does not improve in my exp.")

        # Change default value
        for action in parser._actions:
            if action.dest == 'lr':
                action.default = 0.01
            elif action.dest == 'lr_warmup_steps':
                action.default = 500
            elif action.dest == 'lr_decay_steps':
                action.default = 5000
            elif action.dest == 'early_stopping_rounds':
                action.default = 11000

        return parser

    @classmethod
    def get_model_specific_rs_hparams(cls):
        """Specify the range of hyperparameter search."""
        ch = np.random.choice

        def colsample_bytree_gen(args):
            if args.dataset == 'compas':  # At least 1, 2 features
                if not args.ga2m:
                    return ch([1., 0.5, 0.1])
                return ch([1., 0.5, 0.2])

            if not args.ga2m:
                return ch([0.5, 0.1, 1e-5])
            return ch([1., 0.5, 0.2, 0.1])

        rs_hparams = {
            'seed': dict(short_name='s', gen=lambda args: int(np.random.randint(100))),
            'num_layers': dict(short_name='nl',
                               gen=lambda args: int(ch([2, 3, 4, 5]))),
            'num_trees': dict(short_name='nt',
                              # gen=lambda args: int(ch([4096, 8192, 16384, 32768, 32768*2]))),
                              gen=lambda args: int(ch([500, 1000, 2000,
                                                       4000])) // args.num_layers),
            'depth': dict(short_name='d', gen=lambda args: int(ch([2, 4, 6]))),
            'output_dropout': dict(short_name='od', gen=lambda args: ch([0., 0.1, 0.2])),
            'last_dropout': dict(short_name='ld',
                                 gen=lambda args: (
                                     0. if not args.add_last_linear
                                     else ch([0., 0.15, 0.3]))),
            'colsample_bytree': dict(short_name='cs', gen=colsample_bytree_gen),
            'lr': dict(short_name='lr', gen=lambda args: ch([0.01, 0.005])),
            'l2_lambda': dict(short_name='la', gen=lambda args: float(ch([1e-5, 1e-6, 0.]))),
            'add_last_linear': dict(
                short_name='ll',
                gen=lambda args: int(ch([0, 1])),
            ),
        }
        return rs_hparams

    @classmethod
    def add_model_specific_results(cls, results, args):
        """Record some model attributes into the csv result."""
        # Record annealing steps
        results['anneal_steps'] = args.anneal_steps
        return results


class GAMAttBlock(GAMBlock):
    """Node-GAM with attention model."""

    def create_layers(self, in_features, num_trees, num_layers, tree_dim, **kwargs):
        """Create layers of oblivious trees.

        Args:
            in_features: The dim of input features.
            num_trees: The number of trees in a layer.
            num_layers: The number of layers.
            tree_dim: The output dimension of each tree.
            kwargs: The kwargs for initializing GAMAtt ODST trees.
        """
        layers = []
        prev_in_features = 0
        for i in range(num_layers):
            # Last layer only has the dimension equal to num_classes
            oddt = GAMAttODST(in_features, num_trees, tree_dim=tree_dim,
                              prev_in_features=prev_in_features, **kwargs)
            layers.append(oddt)
            prev_in_features += num_trees * tree_dim
        return layers

    @classmethod
    def add_model_specific_args(cls, parser):
        """Add argparse arguments."""
        parser = super().add_model_specific_args(parser)
        parser.add_argument("--dim_att", type=int, default=8,
                            help="The dimension of attention embedding to reduce # parameters.")
        return parser

    @classmethod
    def get_model_specific_rs_hparams(cls):
        """Specify the range of hyperparameter search."""
        rs_hparams = super().get_model_specific_rs_hparams()
        ch = np.random.choice
        rs_hparams.update({
            'dim_att': dict(short_name='da', gen=lambda args: int(ch([8, 16, 24]))),
        })
        return rs_hparams
