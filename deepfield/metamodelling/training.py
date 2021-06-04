"""Training tools."""
from functools import partial

import torch
from torch.utils.data import random_split
import numpy as np

from .utils import lr_lambda
from .losses import standardize_loss_pattern


class NNTrainer:
    """Class for neural-networks training."""
    # pylint: disable=too-many-arguments
    def train_model(self, model, dataset, n_epoch=1, lr=1e-3, parameters_to_optimize=None,
                    loss_pattern=None, log_every=None, evaluate_every=None, test_frac=0, dump_best_parameters=None,
                    optimizer=torch.optim.Adam,
                    scheduler=partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=lr_lambda), **kwargs):
        """Train model on the given dataset.

        Parameters
        ----------
        model: BaseModel
            Model to train.
        dataset: FieldDataset
            Dataset to use.
        n_epoch: int
            Number of iterations through the dataset.
        lr: float
            Learning rate.
        parameters_to_optimize: None or tuple of model's parameters
            Will optimize loss over this set of parameters.
            If None, will over optimize all parameters.
        loss_pattern: tuple or Any
            For more info see losses.standardize_loss_pattern.
        log_every: int or None
            Print logs at each log_every iteration.
            If None, do not print logs.
        evaluate_every: int or None
            Evaluate model on test dataset each evaluate_every iteration.
            If None, evaluations are not used.
        test_frac: float
            Fraction of scenarios put into the test dataset.
        dump_best_parameters: str or None
            Path to dump best model (on test set). Dumps can be made at each evaluation iteration.
            If None, do not dump model automatically.
        optimizer: torch.optim.Optimizer
            Optimizer to use. Default: torch.optim.Adam
        scheduler: torch.optim.lr_scheduler._LRScheduler
            Scheduler to use. Default: torch.optim.lr_scheduler.LambdaLR
        kwargs: dict
            Any additional named arguments for model's forward pass.

        Returns
        -------
        model
            Trained model.
        train_loss_legend: list
            Training loss legend.
        test_loss_legend: list
            Evaluation loss legend.
        """

        train_loss_legend, test_loss_legend = [], []
        min_test_loss, running_loss, i = np.inf, 0, 0
        loss_pattern = standardize_loss_pattern(loss_pattern)

        if evaluate_every is None:
            evaluate_every = log_every if log_every is not None else 1

        dataset, test_dataset = self.split_dataset(dataset, test_frac)

        model.train()
        optimizer, scheduler = self._get_optimizer_and_scheduler(
            parameters_to_optimize if parameters_to_optimize is not None else model.parameters(),
            optimizer, scheduler, lr
        )

        for epoch in range(n_epoch):
            for sample in dataset:
                i += 1
                loss = self._make_training_iter(model, sample, loss_pattern, optimizer, scheduler, **kwargs)
                running_loss += loss
                train_loss_legend.append([i, loss])

                if test_dataset is not None and i % evaluate_every == evaluate_every - 1:
                    test_loss = self._make_evaluation(model, test_dataset, loss_pattern,
                                                      max_iter=evaluate_every, **kwargs)
                    test_loss_legend.append([i, test_loss])
                    # TODO dump dict with meta-information (best_loss, epoch, loss_type etc.)
                    if dump_best_parameters is not None and test_loss < min_test_loss:
                        model.dump(path=dump_best_parameters)
                        min_test_loss = test_loss

                if dump_best_parameters is not None and test_dataset is None:
                    if loss < min_test_loss:
                        model.dump(path=dump_best_parameters)
                        min_test_loss = loss

                if log_every is not None and i % log_every == log_every - 1:
                    self._log_iter(
                        epoch=epoch, iteration=i,
                        train_loss=running_loss / log_every,
                        test_loss=test_loss_legend[-1][1] if test_loss_legend else None
                    )
                    running_loss = 0
        return model, train_loss_legend, test_loss_legend

    @staticmethod
    def _make_training_iter(model, sample, loss_pattern, optimizer, scheduler, **kwargs):
        """Make single training iter."""
        optimizer.zero_grad()
        loss = model.make_training_iter(sample, loss_pattern, **kwargs)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return float(loss.detach())

    @staticmethod
    def _make_evaluation(model, test_dataset, loss_pattern, max_iter=0, **kwargs):
        """Make single evaluation iter."""
        test_loss, i = 0, 0
        for i, sample in enumerate(test_dataset):
            # No more iterations then max_iter
            if i == max_iter - 1:
                break
            test_loss += float(model.make_evaluation_iter(sample, loss_pattern, **kwargs))
        return test_loss / (i + 1)

    @staticmethod
    def split_dataset(dataset, test_frac):
        """Split dataset into test and train."""
        if test_frac > 0:
            test_len = int(test_frac * len(dataset))
            train, test = random_split(dataset=dataset, lengths=[len(dataset) - test_len, test_len])
            return train, test
        return dataset, None

    @staticmethod
    def _get_optimizer_and_scheduler(parameters, optimizer, scheduler, lr):
        """Initialize optimizer and scheduler."""
        optimizer = optimizer(parameters, lr=lr)
        scheduler = scheduler(optimizer) if scheduler is not None else None
        return optimizer, scheduler

    @staticmethod
    def _log_iter(epoch, iteration, train_loss, test_loss=None):
        """Print logs for the given iteration."""
        # TODO: add logging
        print("epoch=%d\titer=%d :" % (epoch + 1, iteration))
        print("\tTrain Loss: %f" % train_loss)
        if test_loss is not None:
            print("\tTest Loss: %f" % test_loss)
