"""The trainer to optimize the model."""

import glob
import os
import time
from collections import OrderedDict
from copy import deepcopy
from os.path import join as pjoin, exists as pexists

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

try:
	IS_AMP_EXISTS = True
	from apex import amp
except ModuleNotFoundError:
	print('WARNING! The apex is not installed so fp16 is not available.')
	IS_AMP_EXISTS = False

from .utils import get_latest_file, check_numpy, process_in_chunks


class Trainer(nn.Module):
	def __init__(self, model, experiment_name=None, warm_start=False,
				 Optimizer=torch.optim.Adam, optimizer_params={},
				 lr=0.01, lr_warmup_steps=-1, verbose=False,
				 n_last_checkpoints=5, step_callbacks=[], fp16=0,
				 problem='classification', pretraining_ratio=0.15,
				 masks_noise=0.1, opt_only_last_layer=False, freeze_steps=0, **kwargs):
		"""Trainer.

		Args:
			model (torch.nn.Module): the model.
			experiment_name: a path where all logs and checkpoints are saved.
			warm_start: when set to True, loads the last checkpoint.
			Optimizer: function(parameters) -> optimizer. Default: torch.optim.Adam.
			optimizer_params: parameter when intializing optimizer. Usage:
				Optimizer(**optimizer_params).
			verbose: when set to True, produces logging information.
			n_last_checkpoints: the last few checkpoints to do model averaging.
			step_callbacks: function(step). Will be called after each optimization step.
			problem: problem type. Chosen from ['classification', 'regression', 'pretrain'].
			pretraining_ratio: the percentage of feature to mask for reconstruction. Between 0 and
				1. Only used when problem == 'pretrain'.
		"""
		super().__init__()
		self.model = model
		self.verbose = verbose
		self.lr = lr
		self.lr_warmup_steps = lr_warmup_steps

		# When using fp16, there are some params if not filtered out by requires_grad
		# will produce error
		params = [p for p in self.model.parameters() if p.requires_grad]
		if opt_only_last_layer:
			print('Only optimize last layer!')
			params = [self.model.last_w]
		self.opt = Optimizer(params, lr=lr, **optimizer_params)
		self.step = 0
		self.n_last_checkpoints = n_last_checkpoints
		self.step_callbacks = step_callbacks
		self.fp16 = fp16
		self.problem = problem
		self.pretraining_ratio = pretraining_ratio
		self.masks_noise = masks_noise
		self.opt_only_last_layer = opt_only_last_layer
		self.freeze_steps = freeze_steps
		if problem.startswith('pretrain'): # Don't do freeze when pretraining
			self.freeze_steps = 0

		if problem == 'classification':
			# In my datasets I only have binary classification
			self.loss_function = (lambda x, y: F.binary_cross_entropy_with_logits(x, y.float()))
		elif problem == 'regression':
			self.loss_function = (lambda y1, y2: F.mse_loss(y1.float(), y2.float()))
		elif problem.startswith('pretrain'): # Not used
			self.loss_function = None
		else:
			raise NotImplementedError()

		if experiment_name is None:
			experiment_name = 'untitled_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}'.format(*time.gmtime()[:5])
			if self.verbose:
				print('using automatic experiment name: ' + experiment_name)

		self.experiment_path = pjoin('logs/', experiment_name)
		if fp16 and IS_AMP_EXISTS:
			self.model, self.opt = amp.initialize(
				self.model, self.opt, opt_level='O1')
		if warm_start:
			self.load_checkpoint()

	def save_checkpoint(self, tag=None, path=None, mkdir=True, **kwargs):
		assert tag is None or path is None, "please provide either tag or path or nothing, not both"
		if tag is None and path is None:
			tag = "temp_{}".format(self.step)
		if path is None:
			path = pjoin(self.experiment_path, "checkpoint_{}.pth".format(tag))
		if mkdir:
			os.makedirs(os.path.dirname(path), exist_ok=True)

		# Sometimes happen there is a checkpoint already existing. Then overwrite!
		if pexists(path):
			os.remove(path)
		torch.save(OrderedDict([
			('model', self.model.state_dict(**kwargs)),
			('opt', self.opt.state_dict()),
			('step', self.step),
		] + ([] if not (self.fp16 and IS_AMP_EXISTS) else [('amp', amp.state_dict())])), path)
		if self.verbose:
			print("Saved " + path)
		return path

	def load_checkpoint(self, tag=None, path=None, **kwargs):
		assert tag is None or path is None, "please provide either tag or path or nothing, not both"
		if tag is None and path is None:
			path = self.get_latest_file(pjoin(self.experiment_path, 'checkpoint_temp_[0-9]*.pth'))
			if path is None:
				return self

		elif tag is not None and path is None:
			path = pjoin(self.experiment_path, "checkpoint_{}.pth".format(tag))

		checkpoint = torch.load(path)

		self.model.load_state_dict(checkpoint['model'], **kwargs)
		self.opt.load_state_dict(checkpoint['opt'])
		self.step = int(checkpoint['step'])
		if self.fp16 and IS_AMP_EXISTS and 'amp' in checkpoint:
			amp.load_state_dict(checkpoint['amp'])

		# Set the temperature
		for c in self.step_callbacks:
			c(self.step)

		if self.verbose:
			print('Loaded ' + path)
		return self

	def get_latest_file(self, pattern):
		path = get_latest_file(pattern)
		if path is None:
			if self.verbose:
				print('No previous checkpoints found. Train from scratch.')
			return None

		# Remove files not saved correctly
		if os.stat(path).st_size == 0 or len(glob.glob(pattern)) > self.n_last_checkpoints:
			os.remove(path)
			path = self.get_latest_file(pattern)

		return path

	def average_checkpoints(self, tags=None, paths=None, out_tag='avg', out_path=None):
		assert tags is None or paths is None, \
			"please provide either tags or paths or nothing, not both"
		assert out_tag is not None or out_path is not None, \
			"please provide either out_tag or out_path or both, not nothing"
		if tags is None and paths is None:
			paths = self.get_latest_checkpoints(
				pjoin(self.experiment_path, 'checkpoint_temp_[0-9]*.pth'), self.n_last_checkpoints)
		elif tags is not None and paths is None:
			paths = [pjoin(self.experiment_path, 'checkpoint_{}.pth'.format(tag)) for tag in tags]

		checkpoints = [torch.load(path) for path in paths]
		averaged_ckpt = deepcopy(checkpoints[0])
		for key in averaged_ckpt['model']:
			values = [ckpt['model'][key] for ckpt in checkpoints]
			averaged_ckpt['model'][key] = sum(values) / len(values)

		if out_path is None:
			out_path = pjoin(self.experiment_path, 'checkpoint_{}.pth'.format(out_tag))
		torch.save(averaged_ckpt, out_path)

	def get_latest_checkpoints(self, pattern, n_last=None):
		list_of_files = glob.glob(pattern)
		if len(list_of_files) == 0:
			return []

		assert len(list_of_files) > 0, "No latest checkpoint found: " + pattern
		return sorted(list_of_files, key=os.path.getctime, reverse=True)[:n_last]

	def remove_old_temp_checkpoints(self, number_ckpts_to_keep=None):
		if number_ckpts_to_keep is None:
			number_ckpts_to_keep = self.n_last_checkpoints
		paths = self.get_latest_checkpoints(pjoin(self.experiment_path,
												  'checkpoint_temp_[0-9]*.pth'))
		paths_to_delete = paths[number_ckpts_to_keep:]

		for ckpt in paths_to_delete:
			os.remove(ckpt)

	def train_on_batch(self, *batch, device, update=True):
		# Tune temperature in choice function
		for c in self.step_callbacks:
			c(self.step)

		# Tune the learning rate
		if self.lr_warmup_steps > 0 and self.step < self.lr_warmup_steps:
			cur_lr = self.lr * (self.step + 1) / self.lr_warmup_steps
			self.set_lr(cur_lr)

		if self.freeze_steps > 0 and self.step == 0 and update:
			self.model.freeze_all_but_lastw()

		if 0 < self.freeze_steps == self.step:
			self.model.unfreeze()

		x_batch, y_batch = batch
		x_batch = torch.as_tensor(x_batch, device=device)
		if not self.problem.startswith('pretrain'): # Save some memory
			y_batch = torch.as_tensor(y_batch, device=device)

		self.model.train()

		# Read that it's faster...
		for group in self.opt.param_groups:
			for p in group['params']:
				p.grad = None
		# self.opt.zero_grad()

		if not self.problem.startswith('pretrain'): # Normal training
			logits, penalty = self.model(x_batch, return_outputs_penalty=True)
			loss = self.loss_function(logits, y_batch).mean()
		else:
			x_masked, masks, masks_noise = self.mask_input(x_batch)
			feature_masks = masks_noise if self.problem == 'pretrain_recon2' else None
			outputs, penalty = self.model(x_masked, return_outputs_penalty=True,
										  feature_masks=feature_masks)
			loss = self.pretrain_loss(outputs, masks, x_batch)

		loss += penalty

		if self.fp16 and IS_AMP_EXISTS:
			with amp.scale_loss(loss, self.opt) as scaled_loss:
				scaled_loss.backward()
		else:
			loss.backward()

		if update:
			self.opt.step()
			self.step += 1

		return {'loss': loss.item()}

	def mask_input(self, x_batch):
		masks = torch.bernoulli(
			self.pretraining_ratio * torch.ones(x_batch.shape)
		).to(x_batch.device)

		infills = 0.
		# To make it more difficult, 10% of the time we do not mask the inputs! Similar to BERT
		# tricks.
		new_masks = masks
		if self.masks_noise > 0.:
			new_masks = torch.bernoulli((1. - self.masks_noise) * masks)
		x_batch = (1. - new_masks) * x_batch + new_masks * infills
		return x_batch, masks, new_masks

	def pretrain_loss(self, outputs, masks, targets):
		if self.problem.startswith('pretrain_recon'):
			nb_masks = torch.sum(masks, dim=1, keepdim=True)
			nb_masks[nb_masks == 0] = 1
			loss = (((outputs - targets) * masks) ** 2) / nb_masks
			loss = torch.mean(loss)
		else:
			raise NotImplementedError('Unknown problem: ' + self.problem)

		return loss

	def evaluate_pretrain_loss(self, X_test, y_test, device, batch_size=4096):
		X_test = torch.as_tensor(X_test, device=device)
		self.model.train(False)
		with torch.no_grad():
			if self.problem.startswith('pretrain_recon'): # no mask
				outputs = process_in_chunks(self.model, X_test, batch_size=batch_size)
				loss = (((outputs - X_test)) ** 2)
				loss = torch.mean(loss)
			else:
				raise NotImplementedError('Unknown problem: ' + self.problem)

		return loss.item()

	def evaluate_classification_error(self, X_test, y_test, device, batch_size=4096):
		''' This is for evaluation of binary error '''
		X_test = torch.as_tensor(X_test, device=device)
		y_test = check_numpy(y_test)
		self.model.train(False)
		with torch.no_grad():
			logits = process_in_chunks(self.model, X_test, batch_size=batch_size)
			logits = check_numpy(logits)
			error_rate = (y_test != (logits >= 0)).mean()
		return error_rate

	def evaluate_negative_auc(self, X_test, y_test, device, batch_size=4096):
		X_test = torch.as_tensor(X_test, device=device)
		y_test = check_numpy(y_test)
		self.model.train(False)
		with torch.no_grad():
			logits = process_in_chunks(self.model, X_test, batch_size=batch_size)
			logits = check_numpy(logits)
			auc = roc_auc_score(y_test, logits)

		return -auc

	def evaluate_mse(self, X_test, y_test, device, batch_size=4096):
		X_test = torch.as_tensor(X_test, device=device)
		y_test = check_numpy(y_test)
		self.model.train(False)
		with torch.no_grad():
			prediction = process_in_chunks(self.model, X_test, batch_size=batch_size)
			prediction = check_numpy(prediction)
			error_rate = ((y_test - prediction) ** 2).mean()
		error_rate = float(error_rate)  # To avoid annoying JSON unserializable bug
		return error_rate

	def evaluate_multiple_mse(self, X_test, y_test, device, batch_size=4096):
		X_test = torch.as_tensor(X_test, device=device)
		y_test = check_numpy(y_test)
		self.model.train(False)
		with torch.no_grad():
			prediction = process_in_chunks(self.model, X_test, batch_size=batch_size)
			prediction = check_numpy(prediction)
			error_rate = ((y_test - prediction) ** 2).mean(axis=0)
		return error_rate.astype(float).tolist()

	def evaluate_logloss(self, X_test, y_test, device, batch_size=512):
		X_test = torch.as_tensor(X_test, device=device)
		y_test = check_numpy(y_test)
		self.model.train(False)
		with torch.no_grad():
			logits = (process_in_chunks(self.model, X_test, batch_size=batch_size))
			y_test = torch.tensor(y_test, device=device).float()

			logloss = F.binary_cross_entropy_with_logits(logits, y_test).item()
		logloss = float(logloss)  # To avoid annoying JSON unserializable bug
		return logloss

	def decrease_lr(self, ratio=0.1, min_lr=1e-6):
		if self.lr <= min_lr:
			return

		self.lr *= ratio
		if self.lr < min_lr:
			self.lr = min_lr
		self.set_lr(self.lr)

	def set_lr(self, lr):
		for g in self.opt.param_groups:
			g['lr'] = lr
