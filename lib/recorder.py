"""A simple recorder to store the model's training progress."""

import json
import os
from os.path import join as pjoin, exists as pexists

import numpy as np


class Recorder(object):
    def __init__(self, path):
        """A recorder to store the model's training progress.

        Useful to resume training if interuppted by the scheduler. It will reload the record if
        previous record exists.

        Args:
            path: the path to store the record.
        """
        self.path = path
        self.file_path = pjoin(self.path, 'recorder.json')

        self.loss_history, self.err_history = [], []
        self.best_err = float('inf')
        self.best_step_err = 0
        self.step = 0
        self.lr_decay_step = -1
        self.run_time = 0.

        if pexists(self.file_path):
            self.load_record()

    def save_record(self):
        """Save the record."""
        with open(self.file_path, 'w') as op:
            json.dump({
                'best_err': self.best_err,
                'best_step_err': self.best_step_err,
                'step': self.step,
                'lr_decay_step': self.lr_decay_step,
                'run_time': self.run_time,
            }, op)

        np.save(pjoin(self.path, 'loss_history.npy'), self.loss_history)
        np.save(pjoin(self.path, 'err_history.npy'), self.err_history)

    def load_record(self):
        """Load the record."""
        with open(self.file_path) as fp:
            record = json.load(fp)

        if 'loss_history' in record:
            self.loss_history, self.err_history = \
                record['loss_history'], record['err_history']
        elif pexists(pjoin(self.path, 'loss_history.npy')):
            try:
                self.loss_history = np.load(pjoin(self.path, 'loss_history.npy')).tolist()
                self.err_history = np.load(pjoin(self.path, 'err_history.npy')).tolist()
            except ValueError as e:
                print(e)
                print('Encounter problem when loading. Set it to None!')
                self.loss_history = None
                self.err_history = None

        self.best_err = record['best_err']
        self.best_step_err = record['best_step_err']
        self.step = record['step']
        if 'lr_decay_step' in record:
            self.lr_decay_step = record['lr_decay_step']
        if 'run_time' in record:
            self.run_time = record['run_time']

    def clear(self):
        """Remove the record."""
        if pexists(pjoin(self.path, 'loss_history.npy')):
            os.remove(pjoin(self.path, 'loss_history.npy'))
        if pexists(pjoin(self.path, 'err_history.npy')):
            os.remove(pjoin(self.path, 'err_history.npy'))
