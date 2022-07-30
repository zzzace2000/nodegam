"""This file implements a higher-level NodeGAM model that can just call fit(X, y).

The goal is to provide a simple interface for users who just want to use it like::

    >>> model = NodeGAM()
    >>> model.fit(X, y)
"""
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
from qhoptim.pyt import QHAdam
from scipy.special import softmax
from sklearn.model_selection import train_test_split

from .arch import GAMBlock, GAMAttBlock
from .gams.utils import bin_data
from .mypreprocessor import MyPreprocessor
from .nn_utils import EM15Temp, entmoid15
from .trainer import Trainer
from .utils import iterate_minibatches, process_in_chunks, check_numpy, sigmoid_np
from .vis_utils import vis_GAM_effects


class NodeGAMBase(object):
    """Base class for all NodeGAM."""
    def __init__(
        self,
        # Dataset-related
        in_features,
        cat_features=None,
        validation_size=0.15,
        quantile_dist='normal',
        quantile_noise=1e-3,
        # General
        name=None,
        seed=1377,
        # Model
        arch='GAM',
        ga2m=1,
        num_classes=1,
        num_trees=200,
        num_layers=2,
        depth=3,
        addi_tree_dim=0,
        colsample_bytree=0.5,
        output_dropout=0,
        last_dropout=0.3,
        l2_lambda=0,
        dim_att=8,
        # Optimization
        n_last_checkpoints=5,
        batch_size=2048,
        lr=0.01,
        lr_warmup_steps=100, # Warmup in 100 steps
        lr_decay_steps=300, # Decay LR in 900 steps
        early_stopping_steps=2000, # Early stop in 2k steps
        max_steps=20000, # Max 10k steps
        max_time=20 * 3600, # 20 Hours
        anneal_steps=2000,
        report_frequency=100,
        fp16=0,
        device='cuda',
        objective='negative_auc',
        problem='classification',
        verbose=1,
    ):
        """NodeGAM Base."""
        assert arch in ['GAM', 'GAMAtt'], 'Invalid arch: ' + str(arch)
        assert quantile_dist in ['normal', 'uniform'], 'Invalid dist: ' + str(quantile_dist)
        assert objective in ['ce_loss', 'negative_auc', 'mse', 'error_rate'], \
            'Invalid objective: ' + str(objective)

        if name is None:
            name = 'tmp_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}'.format(*time.gmtime()[:5])

        self.name = name
        self.in_features = in_features
        self.cat_features = cat_features
        self.seed = seed
        self.validation_size = validation_size
        self.arch = arch
        self.ga2m = ga2m
        self.num_classes = num_classes
        self.num_trees = num_trees
        self.num_layers = num_layers
        self.depth = depth
        self.addi_tree_dim = addi_tree_dim
        self.colsample_bytree = colsample_bytree
        self.output_dropout = output_dropout
        self.last_dropout = last_dropout
        self.l2_lambda = l2_lambda
        self.dim_att = dim_att
        self.n_last_checkpoints = n_last_checkpoints
        self.batch_size = batch_size
        self.lr = lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.quantile_dist = quantile_dist
        self.quantile_noise = quantile_noise
        self.early_stopping_steps = early_stopping_steps
        self.max_steps = max_steps
        self.max_time = max_time
        self.anneal_steps = anneal_steps
        self.report_frequency = report_frequency
        self.fp16 = fp16
        self.device = device
        self.objective = objective
        self.problem = problem
        self.verbose = verbose

        self.preprocessor = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Train the model.

        Args:
            X (pandas dataframe): input features.
            y (numpy array): targets.

        Returns:
            train_losses: the training losses of each optimization step.
            val_metrics: the validation losses of optimization under the `report_frequency`.
        """
        # Data
        y_normalize = True if self.problem == 'regression' else False
        self.preprocessor = MyPreprocessor(
            cat_features=self.cat_features,
            y_normalize=y_normalize,  # True if regression, False for classification
            random_state=self.seed,
            quantile_transform=True,
            output_distribution=self.quantile_dist,
            quantile_noise=self.quantile_noise,
        )

        self.preprocessor.fit(X, y)

        # Split into train and val
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=self.validation_size, random_state=self.seed)

        # Transform dataset
        X_train, y_train = self.preprocessor.transform(X_train, y_train)
        X_valid, y_valid = self.preprocessor.transform(X_valid, y_valid)

        # Initialize the architecture
        choice_fn = EM15Temp(max_temp=1., min_temp=0.01, steps=self.anneal_steps)
        the_arch = GAMBlock if self.arch == 'GAM' else GAMAttBlock
        self.model = the_arch(
            in_features=X_train.shape[1],
            num_trees=self.num_trees,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            addi_tree_dim=self.addi_tree_dim,
            depth=self.depth,
            choice_function=choice_fn,
            bin_function=entmoid15,
            output_dropout=self.output_dropout,
            last_dropout=self.last_dropout,
            colsample_bytree=self.colsample_bytree,
            selectors_detach=1, # Save memory
            add_last_linear=True,
            ga2m=self.ga2m,
            l2_lambda=self.l2_lambda,
            **({} if self.arch == 'GAM' else {'dim_att': self.dim_att})
        )

        self.model.to(self.device)

        step_callbacks = [choice_fn.temp_step_callback]
        optimizer_params = {'nus': (0.7, 1.0), 'betas': (0.95, 0.998)}
        trainer = Trainer(
            model=self.model,
            experiment_name=self.name,
            warm_start=True,  # if True, will load latest checkpt in the saved dir logs/${name}
            Optimizer=QHAdam,
            optimizer_params=optimizer_params,
            lr=self.lr,
            lr_warmup_steps=self.lr_warmup_steps,
            verbose=False,
            n_last_checkpoints=self.n_last_checkpoints,
            step_callbacks=step_callbacks,  # Temp annelaing
            fp16=self.fp16,
            problem=self.problem,
        )

        # trigger data-aware init of the model
        with torch.no_grad():
            _ = self.model(torch.as_tensor(X_train[:(2 * self.batch_size)], device=self.device))

        train_losses, val_metrics = [], []
        best_err, best_step_err = np.inf, -1
        prev_lr_decay_step = 0
        is_first = True

        st_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size=self.batch_size,
                                         shuffle=True, epochs=float('inf')):
            metrics = trainer.train_on_batch(*batch, device=self.device)

            train_losses.append(float(metrics['loss']))

            if trainer.step % self.report_frequency == 0:
                trainer.save_checkpoint()
                trainer.remove_old_temp_checkpoints()
                trainer.average_checkpoints(out_tag='avg')
                trainer.load_checkpoint(tag='avg')

                if self.objective == 'negative_auc':
                    err = trainer.evaluate_negative_auc(X_valid, y_valid, device=self.device,
                                                        batch_size=self.batch_size * 2)
                elif self.objective == 'error_rate':
                    err = trainer.evaluate_classification_error(
                        X_valid, y_valid, device=self.device, batch_size=self.batch_size * 2)
                elif self.objective == 'ce_loss':
                    err = trainer.evaluate_ce_loss(
                        X_valid, y_valid, device=self.device, batch_size=self.batch_size * 2)
                elif self.objective == 'mse':
                    err = trainer.evaluate_mse(X_valid, y_valid, device=self.device,
                                               batch_size=self.batch_size * 2)
                    err *= (self.preprocessor.y_std ** 2)

                if err < best_err:
                    best_err = err
                    best_step_err = trainer.step
                    trainer.save_checkpoint(tag='best')
                val_metrics.append(err)

                trainer.load_checkpoint()  # last

                if is_first:
                    self.print(f'Steps\tTrain Err\tVal Metric ({self.objective})')
                    is_first = False
                self.print(f"{trainer.step}\t{round(float(metrics['loss']), 4)}\t{round(float(err), 4)}")

            # Stop training at least after the steps for temperature annealing
            if trainer.step > max(self.anneal_steps, best_step_err) + self.early_stopping_steps:
                self.print('BREAK. There is no improvment for {} steps'.format(self.early_stopping_steps))
                break

            if self.lr_decay_steps > 0 \
                    and trainer.step > max(self.anneal_steps, best_step_err) + self.lr_decay_steps \
                    and trainer.step > (prev_lr_decay_step + self.lr_decay_steps):
                lr_before = trainer.lr
                trainer.decrease_lr(ratio=0.2, min_lr=1e-6)
                prev_lr_decay_step = trainer.step
                self.print('LR: %.2e -> %.2e' % (lr_before, trainer.lr))

            if 0 < self.max_steps < trainer.step:
                self.print('End. It reaches the maximum steps %d' % self.max_steps)
                break

            if (time.time() - st_time) > self.max_time:
                self.print('End. It reaches the maximum run time %d (s)' % self.max_time)
                break

        total_train_time = round(time.time() - st_time, 1)
        self.print(f"Total training time: {total_train_time} seconds")
        self.print("Best step: ", best_step_err)
        self.print("Best Val Metric: ", best_err)

        self.print('Load the best checkpoint.')
        trainer.load_checkpoint(tag='best')

        # Clean up
        self.model.train(False)
        shutil.rmtree(os.path.join('logs', self.name), ignore_errors=True)
        del trainer

        return dict(train_losses=train_losses, val_metrics=val_metrics,
                    total_train_time=total_train_time)

    def get_GAM_df(self, all_X: pd.DataFrame, max_n_bins: int = 256):
        """Extract the GAM dataframe from the model.

        Args:
            all_X: all the input data in X.
            max_n_bins: max number of bins per feature.

        Returns:
            df: a GAM dataframe with each row representing a GAM term.
        """
        assert self.num_classes == 1, 'The GAM visualization has not supported more than 1 class.'

        if max_n_bins is not None and max_n_bins > 0:
            all_X = bin_data(all_X, max_n_bins=max_n_bins)

        df = self.model.extract_additive_terms(all_X,
                                               norm_fn=self.preprocessor.transform,
                                               y_mu=self.preprocessor.y_mu,
                                               y_std=self.preprocessor.y_std,
                                               device=self.device,
                                               batch_size=2 * self.batch_size)
        return df

    def visualize(self, X: pd.DataFrame, max_n_bins: int = 256, show_density: bool = False):
        """Visualize the GAM graph.

        Args:
            all_X: all the input data in X.
            max_n_bins: max number of bins per feature.
            show_density: if True, show the density of data as red colors in the background in the
                main effect plot.

        Returns:
            fig: the figure.
            axes: all the subplots.
            df: the GAM dataframe.
        """
        df = self.get_GAM_df(X, max_n_bins=max_n_bins)

        fig, axes = vis_GAM_effects({'model': df}, show_density=show_density)

        return fig, axes, df

    def print(self, *args):
        if self.verbose:
            print(*args)


class NodeGAMClassifier(NodeGAMBase):
    def __init__(
        self,
        # Dataset-related
        in_features,
        cat_features=None,
        validation_size=0.15,
        quantile_dist='normal',
        quantile_noise=1e-3,
        # General
        name=None,
        seed=1377,
        # Model
        arch='GAM',
        ga2m=1,
        num_classes=1,
        num_trees=200,
        num_layers=2,
        depth=3,
        addi_tree_dim=0,
        colsample_bytree=0.5,
        output_dropout=0,
        last_dropout=0.3,
        l2_lambda=0,
        dim_att=8,
        # Optimization
        n_last_checkpoints=5,
        batch_size=2048,
        lr=0.01,
        lr_warmup_steps=100,  # Warmup in 100 steps
        lr_decay_steps=300,  # Decay LR in 900 steps
        early_stopping_steps=2000,  # Early stop in 2k steps
        max_steps=10000,  # Max 10k steps
        max_time=20 * 3600,  # 20 Hours
        anneal_steps=2000,
        report_frequency=100,
        fp16=0,
        device='cuda',
        objective='ce_loss',
        verbose=1,
    ):
        """A NodeGAM Classfier that follows sklearn interface to train.

        Args:
            in_features (int): number of input features.
            cat_features: the name of categorical features that match the columns of X.
            validation_size: validation size.
            quantile_dist: choose between ['normal', 'uniform']. Data is projected onto this
                distribution. See the flag 'output_dist' of sklearn QuantileTransformer.
            quantile_noise: fits QuantileTransformer on data with added gaussian noise with
                std = :quantile_noise: * data.std; this will cause discrete values to be
                more separable. Please note that this transformation does NOT apply gaussian noise
                to the resulting data, the noise is only applied for QuantileTransformer.fit().

            name: the model's name. It's used to store checkpoints under logs/{name}. If not
                specified, it randomly generates a temperory name.
            seed: random seed.
            arch: choose between ['GAM', 'GAMAtt']. GAMAtt is the architecture with attention. Often
                 GAMAtt is better in large datasets while GAM is better in smaller ones.
            ga2m: if 0, only model GAM. If 1, model GA2M.
            num_classes: number of target classes. If set to 1, it is binary classification. Set to
                > 2 for multi-class classifications, but the visualization is not available yet for
                the multi-class setup.
            num_trees: number of trees per layer.
            num_layers: number of layers of trees.
            depth: depth of the tree. Should be at least 2 if ga2m=1.
            addi_tree_dim: additional dimension of tree's output. Default: 0.
            colsample_bytree: the random proportion of features allowed in each tree. The same
                argument as in xgboost package. If less than 1, for each tree, it will only choose
                a fraction of features to train.
            output_dropout: the dropout rate on the output of each tree.
            last_dropout: the dropout rate on the weight of the last linear layer.
            l2_lambda: the l2 penalty coefficient on the outputs of trees.
            dim_att: the dimension of the attention embedding.

            n_last_checkpoints: number of the most recent checkpoints to take average.
            batch_size: batch size. Should be bigger than 1024.
            lr: the learning rate.
            lr_warmup_steps: warm up the learning rate in the first few steps.
            lr_decay_steps: decrease the learning rate by half if not improving for these steps.
            early_stopping_steps: early stopping if not improving for k steps.
            max_steps: maximum number of steps to optimize.
            max_time: maximum number of time to optimize in seconds.
            anneal_steps: temperature annealing steps. After this step, the EntMax becomes Max.
            report_frequency: how many steps to report.
            fp16: if 1, use fp16 to optimize.
            device='cuda': choose from ['cpu', 'cuda'].
            objective: the evaluation objective. Only used in binary classification i.e.
                `num_classes`=1 . Choose from ['ce_loss', 'negative_auc', 'error_rate']. If
                num_classes > 2 (multi-class classifier), only ['ce_loss', 'error_rate'] is allowed.
            verbose: if 1, print the training progress.
        """
        assert arch in ['GAM', 'GAMAtt'], 'Invalid arch: ' + str(arch)
        assert quantile_dist in ['normal', 'uniform'], 'Invalid dist: ' + str(quantile_dist)
        if num_classes == 2:
            self.print('[Warning] num_classes is set to 2 but it should be 1 in the binary '
                       'classification. Change it to be 1.')
            num_classes = 1

        if num_classes == 1:
            assert objective in ['ce_loss', 'negative_auc', 'error_rate'], \
                f"Invalid objective: {objective}. Chose from ['ce_loss', 'negative_auc', " \
                f"'error_rate']."
        elif num_classes > 1:
            assert objective in ['ce_loss', 'error_rate'], \
                f"Invalid objective: {objective}. Chose from ['ce_loss', 'error_rate']."

        super().__init__(
            in_features=in_features,
            cat_features=cat_features,
            validation_size=validation_size,
            quantile_dist=quantile_dist,
            quantile_noise=quantile_noise,
            name=name,
            seed=seed,
            arch=arch,
            ga2m=ga2m,
            num_classes=num_classes,
            num_trees=num_trees,
            num_layers=num_layers,
            depth=depth,
            addi_tree_dim=addi_tree_dim,
            colsample_bytree=colsample_bytree,
            output_dropout=output_dropout,
            last_dropout=last_dropout,
            l2_lambda=l2_lambda,
            dim_att=dim_att,
            # Optimization
            n_last_checkpoints=n_last_checkpoints,
            batch_size=batch_size,
            lr=lr,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            early_stopping_steps=early_stopping_steps,
            max_steps=max_steps,
            max_time=max_time,
            anneal_steps=anneal_steps,
            report_frequency=report_frequency,
            fp16=fp16,
            device=device,
            objective=objective,
            problem='classification',
            verbose=verbose,
        )

    def predict_proba(self, X: pd.DataFrame):
        """Predict probability.

        Args:
            X: pandas dataframe.

        Returns:
            prob (numpy array): the probability of 2 classes with shape [N, 2].
        """
        assert isinstance(X, pd.DataFrame), 'Has to be a dataframe.'

        logits = self.predict(X)
        if self.num_classes == 1:
            prob = sigmoid_np(logits)
            # Back to sklearn format as shape [N, 2]
            prob = np.vstack([1. - prob, prob]).T
        else:
            prob = softmax(logits, axis=-1)
        return prob

    def predict(self, X: pd.DataFrame):
        """Predict logits.

        Args:
            X (pandas dataframe): Input.

        Returns:
            logits (numpy array): logits.
        """
        assert isinstance(X, pd.DataFrame), 'Has to be a dataframe.'

        self.model.train(False)

        X_tr = self.preprocessor.transform(X)
        X_tr = torch.as_tensor(X_tr, device=self.device)
        with torch.no_grad():
            logits = process_in_chunks(self.model, X_tr, batch_size=self.batch_size)
            logits = check_numpy(logits)
        return logits


class NodeGAMRegressor(NodeGAMBase):
    def __init__(
        self,
        # Dataset-related
        in_features,
        cat_features=None,
        validation_size=0.15,
        quantile_dist='normal',
        quantile_noise=1e-3,
        # General
        name=None,
        seed=1377,
        # Model
        arch='GAM',
        ga2m=1,
        num_trees=200,
        num_layers=2,
        depth=3,
        addi_tree_dim=0,
        colsample_bytree=0.5,
        output_dropout=0,
        last_dropout=0.3,
        l2_lambda=0,
        dim_att=8,
        # Optimization
        n_last_checkpoints=5,
        batch_size=2048,
        lr=0.01,
        lr_warmup_steps=100,
        lr_decay_steps=600,
        early_stopping_steps=2000,
        max_steps=20000,
        max_time=20 * 3600,  # 20 Hours
        anneal_steps=2000,
        report_frequency=100,
        fp16=0,
        device='cuda',
        verbose=1,
    ):
        """A NodeGAM Regressor that follows sklearn interface to train.

        Args:
            in_features (int): number of input features.
            cat_features: the name of categorical features that match the columns of X.
            validation_size: validation size.
            quantile_dist: choose between ['normal', 'uniform']. Data is projected onto this
                distribution. See the flag 'output_dist' of sklearn QuantileTransformer.
            quantile_noise: fits QuantileTransformer on data with added gaussian noise with
                std = :quantile_noise: * data.std; this will cause discrete values to be
                more separable. Please note that this transformation does NOT apply gaussian noise
                to the resulting data, the noise is only applied for QuantileTransformer.fit().

            name: the model's name. It's used to store checkpoints under logs/{name}. If not
                specified, it randomly generates a temperory name.
            seed: random seed.
            arch: choose between ['GAM', 'GAMAtt']. GAMAtt is the architecture with attention. Often
                 GAMAtt is better in large datasets while GAM is better in smaller ones.
            ga2m: if 0, only model GAM. If 1, model GA2M.
            num_trees: number of trees per layer.
            num_layers: number of layers of trees.
            depth: depth of the tree. Should be at least 2 if ga2m=1.
            addi_tree_dim: additional dimension of tree's output. Default: 0.
            colsample_bytree: the random proportion of features allowed in each tree. The same
                argument as in xgboost package. If less than 1, for each tree, it will only choose
                a fraction of features to train.
            output_dropout: the dropout rate on the output of each tree.
            last_dropout: the dropout rate on the weight of the last linear layer.
            l2_lambda: the l2 penalty coefficient on the outputs of trees.
            dim_att: the dimension of the attention embedding.

            n_last_checkpoints: number of the most recent checkpoints to take average.
            batch_size: batch size. Should be bigger than 1024.
            lr: the learning rate.
            lr_warmup_steps: warm up the learning rate in the first few steps.
            lr_decay_steps: decrease the learning rate by half if not improving for these steps.
            early_stopping_steps: early stopping if not improving for k steps.
            max_steps: maximum number of steps to optimize.
            max_time: maximum number of time to optimize in seconds.
            anneal_steps: temperature annealing steps. After this step, the EntMax becomes Max.
            report_frequency: how many steps to report.
            fp16: if 1, use fp16 to optimize.
            device='cuda': choose from ['cpu', 'cuda'].
            verbose: if 1, print the training progress.
        """
        assert arch in ['GAM', 'GAMAtt'], 'Invalid arch: ' + str(arch)
        assert quantile_dist in ['normal', 'uniform'], 'Invalid dist: ' + str(quantile_dist)

        super().__init__(
            in_features=in_features,
            cat_features=cat_features,
            validation_size=validation_size,
            quantile_dist=quantile_dist,
            quantile_noise=quantile_noise,
            name=name,
            seed=seed,
            arch=arch,
            ga2m=ga2m,
            num_trees=num_trees,
            num_layers=num_layers,
            depth=depth,
            addi_tree_dim=addi_tree_dim,
            colsample_bytree=colsample_bytree,
            output_dropout=output_dropout,
            last_dropout=last_dropout,
            l2_lambda=l2_lambda,
            dim_att=dim_att,
            # Optimization
            n_last_checkpoints=n_last_checkpoints,
            batch_size=batch_size,
            lr=lr,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            early_stopping_steps=early_stopping_steps,
            max_steps=max_steps,
            max_time=max_time,
            anneal_steps=anneal_steps,
            report_frequency=report_frequency,
            fp16=fp16,
            device=device,
            objective='mse',
            problem='regression',
            verbose=verbose,
        )

    def predict(self, X: pd.DataFrame):
        """Predict regression.

        Args:
            X: pandas dataframe.

        Returns:
            prediction: numpy array.
        """
        assert isinstance(X, pd.DataFrame), 'Has to be a dataframe.'

        self.model.train(False)

        X_tr = self.preprocessor.transform(X)
        X_tr = torch.as_tensor(X_tr, device=self.device)
        with torch.no_grad():
            prediction = process_in_chunks(self.model, X_tr, batch_size=self.batch_size)
            prediction = check_numpy(prediction)

        # Move back the y normalization
        prediction = (prediction * self.preprocessor.y_std) + self.preprocessor.y_mu

        return prediction
