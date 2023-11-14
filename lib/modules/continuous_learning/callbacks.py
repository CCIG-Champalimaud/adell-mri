import lightning.pytorch as pl
from lightning.pytorch import Callback
from typing import List, Union

from .optim import EarlyStopper


class MultiPhaseTraining(Callback):
    """Implements a learning rate routine that allows multiple learning rates
    to be specified over parameter groups. It also has an adaptive check for
    learning rate changes (if no improvement after `patience` epochs, the
    current phase is terminated). Works as a callback for PyTorch Lightning.
    """

    def __init__(
        self,
        learning_rates: List[Union[List[float], str]],
        n_epochs: List[Union[str, int]],
        monitor: str = None,
        patience: int = 10,
        min_delta: float = 0.0,
    ):
        """
        Args:
            learning_rates (List[Union[List[float],str]]): list containing
                lists L_i of learning rates, where each element of L_i is a
                learning rate corresponding to a parameter group in the
                optimizer being used. Instead of a list L_i, can also be
                "stop", causing training to stop.
            n_epochs (List[Union[str,int]]): list containing the epochs
                at which a new phase should begin. Can be int or "adaptive",
                which uses an early stopping-like routine to trigger a change
                to a new phase.
            monitor (str, optional): metric to be monitored if
                n_epochs=="adaptive". Defaults to None.
            patience (int, optional): number of epochs with no improvement in
                `monitor` when n_epochs=="adaptive". Defaults to 10.
            min_delta (float, optional): minimum difference between `monitor`
                and minimum `monitor` to consider that `monitor` is not
                improving. Defaults to 0.0.
        """

        self.learning_rates = learning_rates
        self.n_epochs = n_epochs
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta

        self.assertions()

        self.idx = 0
        self.checker = EarlyStopper(self.patience, self.min_delta)

    def assertions(self):
        """Convienice set of assertions called during initialization."""
        assert len(self.learning_rates) == len(
            self.n_epochs
        ), "len(learning_rates) should be the same as len(n_epochs)"
        assert all(
            [isinstance(x, int) or x == "adaptive" for x in self.n_epochs]
        ), "n_epochs should only have integers or 'adaptive'"
        assert all(
            [isinstance(x, list) or x == "stop" for x in self.learning_rates]
        ), "learning_rates should only contain lists of floats or 'stop'"

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Updates the learning rate for the trainer.

        Args:
            trainer (pl.Trainer): a PyTorch Lightning trainer.
            pl_module (pl.LightningModule): a PyTorch Lightning module
                (not used, for compatibility purposes).
        """
        current_epoch = trainer["epoch"]
        changed = False
        if self.n_epochs[self.idx] == "adaptive":
            v = trainer.callback_metrics.get(self.monitor)
            if self.checker(v) is True:
                changed = True
        else:
            if current_epoch >= self.n_epochs[self.idx]:
                changed = True
        # if there is a change, update the learning rate
        if changed is True:
            self.idx += +1
            self.checker = EarlyStopper(self.patience, self.min_delta)
            opt = self.optimizers()
            lrs = self.learning_rates[self.idx]
            if isinstance(lrs, list):
                for param_group, lr in zip(opt.param_groups, lrs):
                    param_group["lr"] = lr
            elif isinstance(lrs, str):
                if lrs == "stop":
                    trainer.should_stop = True
