import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm

from .odst import ODST, GAM_ODST, GAMAttODST, GAMAtt2ODST, GAMAtt3ODST
from . import nn_utils
from .utils import process_in_chunks, Timer


class ODSTBlock(nn.Sequential):
    def __init__(self, input_dim, layer_dim, num_layers, num_classes=1, addi_tree_dim=0,
                 max_features=None, output_dropout=0.0, flatten_output=True,
                 last_as_output=False, init_bias=False, add_last_linear=False,
                 last_dropout=0., l2_lambda=0., **kwargs):
        layers = self.create_layers(input_dim, layer_dim, num_layers,
                                    tree_dim=num_classes + addi_tree_dim,
                                    max_features=max_features,
                                    **kwargs)
        super().__init__(*layers)
        self.num_layers, self.layer_dim, self.num_classes, self.addi_tree_dim = \
            num_layers, layer_dim, num_classes, addi_tree_dim
        self.max_features, self.flatten_output = max_features, flatten_output
        self.output_dropout = output_dropout
        self.last_as_output = last_as_output
        self.init_bias = init_bias
        self.add_last_linear = add_last_linear
        self.last_dropout = last_dropout
        self.l2_lambda = l2_lambda

        if init_bias:
            val = torch.tensor(0.) if num_classes == 1 \
                else torch.full([num_classes], 0., dtype=torch.float32)
            self.bias = nn.Parameter(val, requires_grad=False)

        self.last_w = None
        if add_last_linear or addi_tree_dim < 0:
            # Happens when more outputs than intermediate tree dim
            self.last_w = nn.Parameter(torch.empty(
                num_layers * layer_dim * (num_classes + addi_tree_dim), num_classes))
            nn.init.xavier_uniform_(self.last_w)

        # Record which params need gradient
        self.named_params_requires_grad = set()
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.named_params_requires_grad.add(name)

    def create_layers(self, input_dim, layer_dim, num_layers,
                      tree_dim, max_features=None, **kwargs):
        layers = []
        for i in range(num_layers):
            # Last layer only has num_classes dim
            oddt = ODST(input_dim, layer_dim, tree_dim=tree_dim, flatten_output=True,
                        **kwargs)
            input_dim = min(input_dim + layer_dim * tree_dim, max_features or float('inf'))
            layers.append(oddt)
        return layers

    def forward(self, x, return_outputs_penalty=False, feature_masks=None):
        '''
        feature_masks: Only used in the pretraining. If passed, the outputs of trees
        belonging to masked features (masks==1) is zeroed.
        This is like dropping out features directly.
        '''
        outputs = self.run_with_layers(x)

        if not self.flatten_output:
            num_output_trees = self.layer_dim if self.last_as_output \
                else self.num_layers * self.layer_dim
            outputs = outputs.view(*outputs.shape[:-1], num_output_trees,
                                   self.num_classes + self.addi_tree_dim)

        # During pretraining, we mask the outputs of trees
        if feature_masks is not None:
            assert not self[0].ga2m, 'Not supported for ga2m for now!'
            with torch.no_grad():
                tmp = torch.cat([l.get_feature_selectors() for l in self], dim=1)
                # ^-- [input_dim, layers * num_trees, 1]
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

        if self.init_bias:
            result += self.bias.data

        if return_outputs_penalty:
            # Average over batch, num_outputs_units
            output_penalty = self.get_penalty(outputs)
            return result, output_penalty
        return result

    def get_penalty(self, outputs):
        return self.l2_lambda * (outputs ** 2).mean()

    def run_with_layers(self, x):
        initial_features = x.shape[-1]

        for layer in self:
            layer_inp = x
            if self.max_features is not None:
                tail_features = min(self.max_features, layer_inp.shape[-1]) - initial_features
                if tail_features != 0:
                    layer_inp = torch.cat([layer_inp[..., :initial_features], layer_inp[..., -tail_features:]], dim=-1)
            h = layer(layer_inp)
            if self.training and self.output_dropout:
                h = F.dropout(h, self.output_dropout)
            x = torch.cat([x, h], dim=-1)

        outputs = h if self.last_as_output else x[..., initial_features:]
        return outputs

    def set_bias(self, y_train):
        ''' Set the bias term for GAM output as logodds of y. '''
        assert self.init_bias

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
        '''
        Return num of trees assigned to each feature in GAM.
        Return a vector of size equal to the input_dim
        '''
        if type(self) is ODSTBlock:
            return None

        num_trees = [l.get_num_trees_assigned_to_each_feature() for l in self]
        return torch.stack(num_trees)

    @classmethod
    def load_model_by_hparams(cls, args, ret_step_callback=False):
        if isinstance(args, dict):
            args = argparse.Namespace(**args)
        assert args.arch == 'ODST', 'Wrong arch: ' + args.arch

        model = ODSTBlock(
            input_dim=args.input_dim,
            layer_dim=args.num_trees,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            addi_tree_dim=args.addi_tree_dim + getattr(args, 'data_addi_tree_dim', 0),
            depth=args.depth, flatten_output=False,
            choice_function=nn_utils.entmax15,
            init_bias=(getattr(args, 'init_bias', False)
                       and args.problem == 'classification'),
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
        parser.add_argument("--colsample_bytree", type=float, default=1.)
        parser.add_argument("--output_dropout", type=float, default=0.)
        parser.add_argument("--add_last_linear", type=int, default=1)
        parser.add_argument("--last_dropout", type=float, default=0.)

        for action in parser._actions:
            if action.dest == 'lr':
                action.default = 1e-3
            # if action.dest == 'batch_size':
            #     action.default = 1024
        return parser

    @classmethod
    def get_model_specific_rs_hparams(cls):
        ch = np.random.choice

        rs_hparams = {
            'seed': dict(short_name='s', gen=lambda args: int(np.random.randint(100))),
            'num_layers': dict(short_name='nl',
                               gen=lambda args: int(ch([2, 3, 4, 5]))),
            'num_trees': dict(short_name='nt',
                              # gen=lambda args: int(ch([4096, 8192, 16384, 32768, 32768*2]))),
                              gen=lambda args: int(ch([500, 1000, 2000, 4000])) // args.num_layers),
            'addi_tree_dim': dict(short_name='td',
                                  gen=lambda args: int(ch([0, 1, 2]))),
            # gen=lambda args: 0),
            'depth': dict(short_name='d', gen=lambda args: int(ch([2, 4, 6]))),
            'output_dropout': dict(short_name='od',
                                   gen=lambda args: ch([0., 0.1, 0.2])),
            'colsample_bytree': dict(short_name='cs', gen=lambda args: ch([1., 0.5, 0.1])),
            'lr': dict(short_name='lr', gen=lambda args: ch([0.01, 0.005])),
            'l2_lambda': dict(short_name='la',
                              gen=lambda args: float(ch([1e-5, 1e-6, 0.]))),
            'add_last_linear': dict(
                short_name='ll',
                gen=lambda args: (int(ch([0, 1]))),
            ),
            'last_dropout': dict(short_name='ld',
                                 gen=lambda args: (0. if not args.add_last_linear
                                                   else ch([0., 0.1, 0.2, 0.3]))),
        }
        return rs_hparams

    @classmethod
    def add_model_specific_results(cls, results, args):
        results['depth'] = args.depth
        return results


class GAMAdditiveMixin(object):
    def run_with_additive_terms(self, x):
        outputs = self.run_with_layers(x)
        td = self.num_classes + self.addi_tree_dim
        outputs = outputs.view(*outputs.shape[:-1], self.num_layers * self.layer_dim, td)
        # ^--[batch_size, layers*num_trees, tree_dim]

        terms, inv = self.get_additive_terms(return_inverse=True)
        # ^-- (list of unique terms, [layers*num_trees])

        if self.last_w is not None:
            inv = inv.unsqueeze_(-1).expand(-1, td).reshape(-1)
            # ^-- [layers*num_trees*tree_dim] binary features

            new_w = inv.new_zeros(inv.shape[0], len(terms), self.num_classes, dtype=torch.float32)
            # ^-- [layers*num_trees*tree_dim, uniq_terms, num_classes]
            val = self.last_w.unsqueeze(1).expand(-1, len(terms), -1)
            # ^-- [layers*num_trees*tree_dim, uniq_terms, num_classes]
            idx = inv.unsqueeze_(-1).unsqueeze_(-1).expand(-1, 1, self.num_classes)
            # ^-- [layers*num_trees*tree_dim, num_classes]
            new_w.scatter_(1, idx, val)

            result = torch.einsum(
                'bd,duc->buc', outputs.reshape(outputs.shape[0], -1), new_w
            )
        else:
            outputs = outputs[..., :self.num_classes]
            # ^--[batch_size, layers*num_trees, num_classes]

            new_w = inv.new_zeros(inv.shape[0], len(terms), dtype=torch.float32)
            # idx = inv.unsqueeze_(-1)
            new_w.scatter_(1, inv.unsqueeze_(-1), 1. / inv.shape[0])
            # ^-- [layers*num_trees*tree_dim, uniq_terms]

            result = torch.einsum('bdc,du->buc', outputs, new_w)
        return result

    def extract_additive_terms(self, X, norm_fn=lambda x: x, y_mu=0., y_std=1.,
                               device='cpu', batch_size=1024, tol=1e-3,
                               purify=True, min_purify_counts=0, samples_per_bin=-1):
        '''
        X: input 2d array (pandas)
        interactions: a list of interaction term. E.g. [[0, 1], [0, 2]]
        predict_type: choose from ["binary_logodds", "binary_prob", "regression"]
            This corresponds to which predict_fn to pass in.
        '''
        assert self.num_classes == 1, 'Has not support > 2 classes. But should be easy.'
        assert isinstance(X, pd.DataFrame)
        self.eval()

        with Timer('Run and extract values', remove_start_msg=False):
            vals, counts, terms = self._run_and_extract_vals_counts(
                X, norm_fn=None, y_mu=0., y_std=1., device=device, batch_size=batch_size)

        if purify:
            # Doing centering: do the pairwise purification
            with Timer('Purify interactions to main effects'):
                self._purify_interactions(
                    X, terms, vals, counts,
                    tol=tol, min_purify_counts=min_purify_counts,
                    samples_per_bin=samples_per_bin,
                )

        # Center the main effect
        with Timer('Center main effects'):
            vals[-1] += (0. if not self.init_bias else self.bias.data.item())
            for idx, t in enumerate(terms):
                if isinstance(t, tuple):  # main term
                    continue

                weights = counts[t].values
                avg = np.average(vals[t].values, weights=weights)

                vals[-1] += avg
                vals[t] -= avg

        # Organize data frame
        results = [{
            'feat_name': 'offset',
            'feat_idx': -1,
            'x': None,
            'y': np.full(1, vals[-1]),
            'importance': -1,
            'counts': None,
        }]

        with Timer('Construct table'):
            for t in vals:
                if t == -1:
                    continue
                x = list(vals[t].index)
                y = vals[t].values.tolist()
                count = counts[t].values.tolist()

                if not isinstance(t, tuple):
                    tmp = np.argsort(x)
                    x, y, count = np.array(x)[tmp], np.array(y)[tmp], np.array(count)[tmp]

                imp = np.average(np.abs(np.array(y)), weights=np.array(count))
                results.append({
                    'feat_name': (X.columns[t] if not isinstance(t, tuple)
                                  else f'{X.columns[t[0]]}*{X.columns[t[1]]}'),
                    'feat_idx': t,
                    'x': x,
                    'y': y,
                    'importance': imp,
                    'counts': count,
                })

            df = pd.DataFrame(results)
            df['tmp'] = df.feat_idx.apply(
                lambda x: x[0] * 1e10 + x[1] * 1e5 if isinstance(x, tuple) else int(x))
            df = df.sort_values('tmp').drop('tmp', axis=1)
            df = df.reset_index(drop=True)
        return df

    def _run_and_extract_vals_counts(self, X, device, batch_size,
                                     norm_fn=lambda x: x, y_mu=0., y_std=1.):
        with Timer('Run the additive terms'):
            with torch.no_grad():
                terms = self.get_additive_terms()
                results = process_in_chunks(
                    lambda x: self.run_with_additive_terms(
                        torch.tensor(norm_fn(x), device=device)),
                    X.values, batch_size=batch_size)
                results = results.cpu().numpy()

                results = (results * y_std)

        # Extract all additive term results
        with Timer('Extract values'):
            vals, counts = {}, {}
            vals[-1] = y_mu

            for idx, t in enumerate(tqdm(terms)):
                if isinstance(t, tuple):
                    index = pd.MultiIndex.from_frame(X.iloc[:, list(t)])
                else:
                    index = X.iloc[:, t]
                scores = pd.Series(results[:, idx, 0], index=index)

                tmp = scores.groupby(level=0)
                vals[t] = tmp.first() # The rest element should be the same
                counts[t] = tmp.count()

            # Just in case some main effects are not chosen!
            for i in range(X.shape[1]):
                if i in vals:
                    continue
                index = X.iloc[:, i]
                scores = pd.Series(np.zeros(X.shape[0]), index=index)

                tmp = scores.groupby(level=0)
                counts[i] = tmp.count()
                vals[i] = np.zeros_like(counts[i]) #tmp.apply(lambda x: 0.)
        return vals, counts, terms

    def _purify_interactions(self, X, terms, vals, counts, tol=1e-3, min_purify_counts=0,
                             samples_per_bin=-1):
        for idx, t in enumerate(terms):
            if not isinstance(t, tuple):  # only interactions
                continue

            uniq_pair = X.iloc[:, list(t)].drop_duplicates().values
            uniq_x0, count_x0 = np.unique(X.iloc[:, t[0]].values, return_counts=True)
            uniq_x1, count_x1 = np.unique(X.iloc[:, t[1]].values, return_counts=True)
            if samples_per_bin is None or samples_per_bin == -1 \
                    or len(uniq_x0) * len(uniq_x1) * samples_per_bin <= X.shape[0] \
                    or (X.dtypes.iloc[t[0]] is np.dtype(object) and X.dtypes.iloc[t[1]] is np.dtype(object)):

                biggest_epsilon = np.inf
                while biggest_epsilon > tol:
                    biggest_epsilon = -np.inf

                    # Calculate the main term for first 1
                    for u in uniq_x0[count_x0 >= min_purify_counts]:
                        pairs = uniq_pair[uniq_pair[:, 0] == u]
                        all_val = np.array([vals[t][tuple(p)] for p in pairs])
                        all_counts = np.array([counts[t][tuple(p)] for p in pairs])
                        avg = np.average(all_val, weights=all_counts)

                        if np.abs(avg) > biggest_epsilon:
                            biggest_epsilon = np.abs(avg)

                        vals[t[0]][u] += avg
                        for p in pairs:
                            vals[t][tuple(p)] -= avg

                    for v in uniq_x1[count_x1 >= min_purify_counts]:
                        pairs = uniq_pair[uniq_pair[:, 1] == v]
                        all_val = np.array([vals[t][tuple(p)] for p in pairs])
                        all_counts = np.array([counts[t][tuple(p)] for p in pairs])
                        avg = np.average(all_val, weights=all_counts)

                        if np.abs(avg) > biggest_epsilon:
                            biggest_epsilon = np.abs(avg)
                        vals[t[1]][v] += avg
                        for p in pairs:
                            vals[t][tuple(p)] -= avg
            else:
                col_data1, col_data2 = X.iloc[:, t[0]].values, X.iloc[:, t[1]].values
                bins1 = bins2 = int(np.sqrt(X.shape[0] / samples_per_bin))

                print(t, bins1, bins2)

                digitized1 = self.quantile_digitize(col_data1, bins1)
                digitized2 = self.quantile_digitize(col_data2, bins2)

                # cache 1 and 2
                uniq_dig1, uniq_dig2 = np.unique(digitized1), np.unique(digitized2)

                # Take out interaction values
                the_vals, the_counts = {0: {}, 1: {}, (0, 1): {}}, {0: {}, 1: {}, (0, 1): {}}
                for the_bin1 in uniq_dig1:
                    for u in np.unique(col_data1[(digitized1 == the_bin1)]):
                        for the_bin2 in uniq_dig2:
                            for v in np.unique(col_data2[(digitized2 == the_bin2)]):
                                if (u, v) not in vals[t]:
                                    continue
                                the_vals[(0, 1)][(the_bin1, the_bin2)] = \
                                    the_vals[(0, 1)].get((the_bin1, the_bin2), 0.) + vals[t][(u, v)]
                                the_counts[(0, 1)][(the_bin1, the_bin2)] = \
                                    the_counts[(0, 1)].get((the_bin1, the_bin2), 0.) + counts[t][(u, v)]

                orig_it_vals = copy.deepcopy(the_vals[(0, 1)])

                biggest_epsilon = np.inf
                while biggest_epsilon > tol:
                    biggest_epsilon = -np.inf

                    # Calculate the main term for first 1
                    for the_bin1 in uniq_dig1:
                        all_val, all_counts = [], []
                        for the_bin2 in uniq_dig2:
                            if (the_bin1, the_bin2) not in the_vals[(0, 1)]:
                                continue
                            all_val.append(the_vals[(0, 1)][(the_bin1, the_bin2)])
                            all_counts.append(the_counts[(0, 1)][(the_bin1, the_bin2)])
                        if len(all_counts) == 0 or np.sum(all_counts) < min_purify_counts:
                            continue
                        avg = np.average(np.array(all_val), weights=np.array(all_counts))

                        if np.abs(avg) > biggest_epsilon:
                            biggest_epsilon = np.abs(avg)
                        the_vals[0][the_bin1] = the_vals[0].get(the_bin1, 0.) + avg
                        for the_bin2 in uniq_dig2:
                            if (the_bin1, the_bin2) not in the_vals[(0, 1)]:
                                continue
                            the_vals[(0, 1)][(the_bin1, the_bin2)] -= avg

                    # Center the 2nd
                    for the_bin2 in uniq_dig2:
                        all_val, all_counts = [], []
                        for the_bin1 in uniq_dig1:
                            if (the_bin1, the_bin2) not in the_vals[(0, 1)]:
                                continue
                            all_val.append(the_vals[(0, 1)][(the_bin1, the_bin2)])
                            all_counts.append(the_counts[(0, 1)][(the_bin1, the_bin2)])

                        if len(all_counts) == 0 or np.sum(all_counts) < min_purify_counts:
                            continue
                        avg = np.average(np.array(all_val), weights=np.array(all_counts))

                        if np.abs(avg) > biggest_epsilon:
                            biggest_epsilon = np.abs(avg)
                        the_vals[1][the_bin2] = the_vals[1].get(the_bin2, 0.) + avg
                        for the_bin1 in uniq_dig1:
                            if (the_bin1, the_bin2) not in the_vals[(0, 1)]:
                                continue
                            the_vals[(0, 1)][(the_bin1, the_bin2)] -= avg

                # Add it back to actual vals
                for the_bin1 in uniq_dig1:
                    for u in np.unique(col_data1[(digitized1 == the_bin1)]):
                        vals[t[0]][u] += the_vals[0][the_bin1]
                        for the_bin2 in uniq_dig2:
                            for v in np.unique(col_data2[(digitized2 == the_bin2)]):
                                if (u, v) not in vals[t]:
                                    continue
                                vals[t][(u, v)] -= \
                                    (orig_it_vals[(the_bin1, the_bin2)]
                                     - the_vals[(0, 1)][(the_bin1, the_bin2)])

                for the_bin2 in uniq_dig2:
                    for v in np.unique(col_data2[(digitized2 == the_bin2)]):
                        vals[t[1]][v] += the_vals[1][the_bin2]

    def quantile_digitize(self, col_data, max_n_bins=None):
        uniq_vals, uniq_idx = np.unique(col_data, return_inverse=True)
        if max_n_bins is None or len(uniq_vals) <= max_n_bins:
            return uniq_idx

        bins = np.unique(
            np.quantile(
                uniq_vals, q=np.linspace(0, 1, max_n_bins + 1),
            )
        )

        _, bin_edges = np.histogram(col_data, bins=bins)
        digitized = np.digitize(col_data, bin_edges, right=False)
        digitized[digitized == 0] = 1
        digitized -= 1
        return digitized

    def get_additive_terms(self, return_inverse=False):
        fs = torch.cat([l.get_feature_selectors() for l in self], dim=1).sum(dim=-1)
        fs[fs > 0.] = 1.
        # ^-- [input_dim, layers*num_trees] binary features

        result = torch.unique(fs, dim=1, sorted=True, return_inverse=return_inverse)
        # ^-- ([input_dim, uniq_terms], [layers*num_trees])

        terms = result
        if isinstance(result, tuple): # return inverse=True
            terms = result[0]

        # Make additive terms human-readable: make it as integer or tuple
        tuple_terms = self.get_tuple_terms(terms)

        if isinstance(result, tuple):
            return tuple_terms, result[1]
        return tuple_terms

    def get_tuple_terms(self, terms):
        r_idx, c_idx = torch.nonzero(terms, as_tuple=True)
        tuple_terms = []
        for c in range(terms.shape[1]):
            n_interaction = (c_idx == c).sum()

            if n_interaction > 2:
                print(f'WARNING: it is not a GA2M with a {n_interaction}-way term. '
                      f'Ignore this term.')
                continue
            if n_interaction == 1:
                tuple_terms.append(int(r_idx[c_idx == c].item()))
            elif n_interaction == 2:
                tuple_terms.append(tuple(r_idx[c_idx == c][:2].cpu().numpy()))
        return tuple_terms


class GAMAdditiveMixin2(GAMAdditiveMixin):
    def extract_additive_terms(self, X, norm_fn=lambda x: x, y_mu=0., y_std=1.,
                               device='cpu', batch_size=1024, tol=1e-3,
                               purify=True, min_purify_counts=0, samples_per_bin=-1):
        '''
        X: input 2d array (pandas)
        interactions: a list of interaction term. E.g. [[0, 1], [0, 2]]
        predict_type: choose from ["binary_logodds", "binary_prob", "regression"]
            This corresponds to which predict_fn to pass in.
        '''
        assert self.num_classes == 1, 'Has not support > 2 classes. But should be easy.'
        assert isinstance(X, pd.DataFrame)
        self.eval()

        # with Timer('Run and extract values', remove_start_msg=False):
        vals, counts, terms = self._run_and_extract_vals_counts(
            X, device, batch_size, norm_fn=norm_fn, y_mu=y_mu, y_std=y_std)

        if purify:
            # Doing centering: do the pairwise purification
            with Timer('Purify interactions to main effects'):
                self._purify_interactions(
                    X, terms, vals, counts,
                    tol=tol, min_purify_counts=min_purify_counts,
                    samples_per_bin=samples_per_bin,
                )

        # Center the main effect
        with Timer('Center main effects'):
            vals[-1] += (0. if not self.init_bias else self.bias.data.item())
            for idx, t in enumerate(terms):
                if isinstance(t, tuple):  # main term
                    continue

                # weights = np.array(list(counts[t].values()))
                # avg = np.average(np.array(list(vals[t].values())), weights=weights)
                weights = counts[t].values
                avg = np.average(vals[t].values, weights=weights)

                vals[-1] += avg
                vals[t] -= avg

        # Organize data frame
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
                lambda x: x[0] * 1e10 + x[1] * 1e5 if isinstance(x, tuple) else int(x))
            df = df.sort_values('tmp').drop('tmp', axis=1)
            df = df.reset_index(drop=True)
        return df

    def _run_and_extract_vals_counts(self, X, device, batch_size,
                                     norm_fn=lambda x: x, y_mu=0., y_std=1.):
        with Timer('Run values through model'), torch.no_grad():
            results = self._run_vals_with_additive_term_with_batch(
                X, device, batch_size, norm_fn=norm_fn, y_mu=y_mu, y_std=y_std)

        # Extract all additive term results
        with Timer('Extract values'):
            vals, counts, terms = self._extract_vals_counts(results, X)
            vals[-1] = y_mu
        return vals, counts, terms

    def _run_vals_with_additive_term_with_batch(self, X, device, batch_size,
                                                norm_fn=lambda x: x, y_mu=0., y_std=1.):
        results = process_in_chunks(
            lambda x: self.run_with_additive_terms(
                torch.tensor(norm_fn(x), device=device)),
            X.values, batch_size=batch_size)
        results = results.cpu().numpy()
        results = (results * y_std)
        return results

    def _extract_vals_counts(self, results, X):
        terms = self.get_additive_terms()

        vals, counts = {}, {}
        for idx, t in enumerate(tqdm(terms)):
            if not isinstance(t, tuple):
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

        # In case only interaction effect is chosen but not main effect
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

    def _purify_interactions(self, X, terms, vals, counts, tol=1e-3, min_purify_counts=0,
                             samples_per_bin=-1):
        for idx, t in enumerate(terms):
            if not isinstance(t, tuple):  # only interactions
                continue

            if True:
                biggest_epsilon = np.inf
                while biggest_epsilon > tol:
                    biggest_epsilon = -np.inf

                    avg = (vals[t] * counts[t]).sum(axis=1).values / counts[t].sum(axis=1).values
                    if np.max(np.abs(avg)) > biggest_epsilon:
                        biggest_epsilon = np.max(np.abs(avg))

                    vals[t] -= avg.reshape(-1, 1)
                    vals[t[0]] += avg

                    avg = (vals[t] * counts[t]).sum(axis=0).values / counts[t].sum(axis=0).values
                    if np.max(np.abs(avg)) > biggest_epsilon:
                        biggest_epsilon = np.max(np.abs(avg))

                    vals[t] -= avg.reshape(1, -1)
                    vals[t[1]] += avg


class GAMBlock(GAMAdditiveMixin2, ODSTBlock):
    def __init__(self, *args, l2_interactions=0., l1_interactions=0., **kwargs):
        super().__init__(*args, **kwargs)
        self.l2_interactions = l2_interactions
        self.l1_interactions = l1_interactions

        self.inv_is_interaction = None

    def create_layers(self, input_dim, layer_dim, num_layers,
                      tree_dim, max_features=None, **kwargs):
        layers = []
        for i in range(num_layers):
            # Last layer only has num_classes dim
            oddt = GAM_ODST(input_dim, layer_dim, tree_dim=tree_dim,
                            flatten_output=True, **kwargs)
            layers.append(oddt)
        return layers

    def get_penalty(self, outputs):
        # Normal L2 weight decay on outputs
        penalty = super().get_penalty(outputs)
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
            penalty += self.l2_interactions * torch.mean(outputs_interactions ** 2)
        if self.l1_interactions > 0.:
            penalty += self.l1_interactions * torch.mean(torch.abs(outputs_interactions))

        return penalty

    def run_with_layers(self, x, return_fs=False):
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

        outputs = h if self.last_as_output else x[..., initial_features:]
        if return_fs:
            return outputs, prev_feature_selectors
        return outputs

    @classmethod
    def load_model_by_hparams(cls, args, ret_step_callback=False):
        if isinstance(args, dict):
            args = argparse.Namespace(**args)
        assert args.arch in ['GAM', 'GAMAtt', 'GAMAtt2', 'GAMAtt3'], 'Wrong arch: ' + args.arch
        if not getattr(args, 'ga2m', 0):
            assert getattr(args, 'l2_interactions', 0.) == 0., \
                'No L2 penalty should be set for interaction'
            assert getattr(args, 'l1_interactions', 0.) == 0., \
                'No L1 penalty should be set for interaction'

        choice_fn = getattr(nn_utils, args.choice_fn)(
            max_temp=1., min_temp=args.min_temp, steps=args.anneal_steps)

        # Temperature annealing for entmoid
        bin_function = nn_utils.entmoid15
        args.entmoid_min_temp = getattr(args, 'entmoid_min_temp', 1.)
        if args.entmoid_min_temp < 1.:
            bin_function = nn_utils.EMoid15Temp(
                min_temp=args.entmoid_min_temp, steps=args.entmoid_anneal_steps)

        kwargs = dict(
            input_dim=args.input_dim,
            layer_dim=args.num_trees,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            addi_tree_dim=args.addi_tree_dim + getattr(args, 'data_addi_tree_dim', 0),
            depth=args.depth,
            flatten_output=False,
            choice_function=choice_fn,
            bin_function=bin_function,
            output_dropout=args.output_dropout,
            last_dropout=getattr(args, 'last_dropout', 0.),
            colsample_bytree=args.colsample_bytree,
            selectors_detach=args.selectors_detach,
            fs_normalize=args.fs_normalize,
            last_as_output=args.last_as_output,
            init_bias=(getattr(args, 'init_bias', False)
                       and args.problem == 'classification'),
            add_last_linear=getattr(args, 'add_last_linear', False),
            save_memory=getattr(args, 'save_memory', 0),
            ga2m=getattr(args, 'ga2m', 0),
            l2_lambda=args.l2_lambda,
            l2_interactions=getattr(args, 'l2_interactions', 0.),
            l1_interactions=getattr(args, 'l1_interactions', 0.),
        )

        if args.arch in ['GAMAtt', 'GAMAtt2', 'GAMAtt3'] and 'dim_att' in args:
            kwargs['dim_att'] = args.dim_att

        model = cls(**kwargs)
        if not ret_step_callback:
            return model

        step_callbacks = [choice_fn.temp_step_callback]
        if args.entmoid_min_temp < 1.:
            step_callbacks.append(bin_function.temp_step_callback)
        return model, step_callbacks

    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument("--colsample_bytree", type=float, default=1.)
        parser.add_argument("--output_dropout", type=float, default=0.)
        parser.add_argument("--last_dropout", type=float, default=0.)
        parser.add_argument("--last_as_output", type=int, default=0)
        parser.add_argument("--min_temp", type=float, default=1e-2)
        parser.add_argument("--anneal_steps", type=int, default=4000)

        parser.add_argument("--choice_fn", default='EM15Temp',
                            help="Choose the dataset.",
                            choices=['GSMTemp', 'SMTemp', 'EM15Temp'])

        parser.add_argument("--entmoid_min_temp", type=float, default=1.,
                            help="If smaller than 1, the shape function becomes jumpy.")
        parser.add_argument("--entmoid_anneal_steps", type=int, default=4000,
                            help="If smaller than 1, the shape function becomes jumpy.")

        parser.add_argument("--selectors_detach", type=int, default=0)
        parser.add_argument("--fs_normalize", type=int, default=1)
        parser.add_argument("--init_bias", type=int, default=1)
        parser.add_argument("--add_last_linear", type=int, default=1)

        # Use GA2M
        parser.add_argument("--ga2m", type=int, default=0)
        parser.add_argument("--l2_interactions", type=float, default=0.)
        parser.add_argument("--l1_interactions", type=float, default=0.)

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
        ch = np.random.choice
        def colsample_bytree_gen(args):
            if args.dataset == 'compas': # At least 1, 2 features
                if not args.ga2m:
                    return ch([1., 0.5, 0.1])
                return ch([1., 0.5, 0.2])

            if not args.ga2m:
                return ch([0.5, 0.1, 1e-5])
            return ch([1., 0.5, 0.2, 0.1])

        rs_hparams = {
            # 'arch': dict(short_name='', gen=lambda args: np.random.choice(['GAM', 'GAMAtt'])),
            'seed': dict(short_name='s', gen=lambda args: int(np.random.randint(100))),
            # 'seed': dict(short_name='s', gen=lambda args: 2),  # Fix seed; see other hparams
            'num_layers': dict(short_name='nl',
                               gen=lambda args: int(ch([2, 3, 4, 5]))),
            'num_trees': dict(short_name='nt',
                              # gen=lambda args: int(ch([4096, 8192, 16384, 32768, 32768*2]))),
                              gen=lambda args: int(ch([500, 1000, 2000, 4000])) // args.num_layers),
            'addi_tree_dim': dict(short_name='td',
                                  gen=lambda args: int(ch([0, 1, 2]))),
                                  # gen=lambda args: 0),
            'depth': dict(short_name='d', gen=lambda args: int(ch([2, 4, 6]))),
            'output_dropout': dict(short_name='od',
                                   gen=lambda args: ch([0., 0.1, 0.2])),
            'last_dropout': dict(short_name='ld',
                                 gen=lambda args: (0. if not args.add_last_linear
                                                   else ch([0., 0.15, 0.3]))),
            'colsample_bytree': dict(short_name='cs', gen=colsample_bytree_gen),
            'lr': dict(short_name='lr', gen=lambda args: ch([0.01, 0.005])),
            # 'last_as_output': dict(short_name='lo', gen=lambda args: int(ch([0, 1]))),
            'last_as_output': dict(short_name='lo', gen=lambda args: 0),
            # 'anneal_steps': dict(short_name='an', gen=lambda args: int(ch([2000, 4000, 6000]))),
            'l2_lambda': dict(short_name='la',
                              gen=lambda args: float(ch([1e-5, 1e-6, 0.]))),
            'pretrain': dict(short_name='pt'),
            'pretraining_ratio': dict(
                short_name='pr',
                # gen=lambda args: float(ch([0.1, 0.15, 0.2])) if args.pretrain else 0),
                # gen=lambda args: 0.15 if args.pretrain else 0,
            ),
            'masks_noise': dict(
                short_name='mn',
                # gen=lambda args: float(ch([0., 0.1, 0.2])) if args.pretrain else 0),
                gen=lambda args: 0.1 if args.pretrain else 0),
            'opt_only_last_layer': dict(
                short_name='ol',
                # gen=lambda args: (int(ch([0, 1])) if args.pretrain else 0)),
                gen=lambda args: 0),
            'add_last_linear': dict(
                short_name='ll',
                gen=lambda args: (1 if (args.pretrain or args.arch == 'GAM')
                                  else int(ch([0, 1]))),
            ),
        }
        return rs_hparams

    @classmethod
    def add_model_specific_results(cls, results, args):
        results['anneal_steps'] = args.anneal_steps
        return results


class GAMAttBlock(GAMBlock):
    def create_layers(self, input_dim, layer_dim, num_layers,
                      tree_dim, max_features=None, **kwargs):
        layers = []
        prev_input_dim = 0
        for i in range(num_layers):
            # Last layer only has num_classes dim
            oddt = GAMAttODST(input_dim, layer_dim, tree_dim=tree_dim,
                              flatten_output=True,
                              prev_input_dim=prev_input_dim, **kwargs)
            layers.append(oddt)
            prev_input_dim += layer_dim * tree_dim
        return layers

    @classmethod
    def add_model_specific_args(cls, parser):
        parser = super().add_model_specific_args(parser)
        parser.add_argument("--dim_att", type=int, default=64)
        return parser

    @classmethod
    def get_model_specific_rs_hparams(cls):
        rs_hparams = super().get_model_specific_rs_hparams()
        ch = np.random.choice
        rs_hparams.update({
            'dim_att': dict(short_name='da',
                            gen=lambda args: int(ch([8, 16, 32]))),
            # 'add_last_linear': dict(
            #     short_name='ll',
            #     gen=lambda args: (1 if args.pretrain else int(ch([0, 1]))),
            # ),
            # 'add_last_linear': dict(short_name='ll', gen=lambda args: 1),
            # 'colsample_bytree': dict(short_name='cs',
            #                          gen=lambda args: ch([0.5, 0.1])),
        })
        return rs_hparams


class GAMAtt2Block(GAMAttBlock):
    def create_layers(self, input_dim, layer_dim, num_layers,
                      tree_dim, max_features=None, **kwargs):
        layers = []
        prev_input_dim = 0
        for i in range(num_layers):
            # Last layer only has num_classes dim
            oddt = GAMAtt2ODST(input_dim, layer_dim, tree_dim=tree_dim,
                               flatten_output=True,
                               prev_input_dim=prev_input_dim, **kwargs)
            layers.append(oddt)
            prev_input_dim += layer_dim * tree_dim
        return layers


class GAMAtt3Block(GAMAttBlock):
    def create_layers(self, input_dim, layer_dim, num_layers,
                      tree_dim, max_features=None, **kwargs):
        layers = []
        prev_input_dim = 0
        for i in range(num_layers):
            # Last layer only has num_classes dim
            oddt = GAMAtt3ODST(input_dim, layer_dim, tree_dim=tree_dim,
                               flatten_output=True,
                               prev_input_dim=prev_input_dim, **kwargs)
            layers.append(oddt)
            prev_input_dim += layer_dim * tree_dim
        return layers

