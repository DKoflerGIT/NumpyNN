"""Logging callbacks."""

from collections.abc import Iterable
from typing import Any, Literal, Optional

from tqdm import tqdm

from ...utils.tensorboard import SummaryWriter
from .callback import Callback

__all__ = ["History", "ProgressBar", "Tensorboard"]


class History(Callback):
    """Training history saved as a dictionary."""

    def __init__(self) -> None:
        self._cache: dict[str, list[float]] = {}

    def __getitem__(self, key: str) -> list[float]:
        return self._cache[key]

    def _log_stats(self, stats: Iterable[str], trainer_cache: dict[str, Any]) -> None:
        for stat in stats:
            if stat not in trainer_cache:
                return
            if stat in self._cache:
                self._cache[stat].append(trainer_cache[stat])
            else:
                self._cache[stat] = [trainer_cache[stat]]

    def on_step_end(self, trainer_cache: dict[str, Any]) -> None:
        # get stats that contain scores but are not val scores
        scores = [s for s in trainer_cache.keys() if "score" in s and "val_" not in s]
        self._log_stats(["loss", "lr"] + scores, trainer_cache)

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        # get all val stats
        stats = [s for s in trainer_cache.keys() if "val_" in s]
        self._log_stats(stats, trainer_cache)


class ProgressBar(Callback):
    """Callback used for displaying the training progress.

    Parameters
    ----------
    mode : Literal["step", "epoch"], optional
        | Progress bar update mode. Defaults to ``step``.
        | ``epoch``: one progress bar is shown and updated for each epoch.
        | ``step``: a new progress bar is shown per epoch and updated for each step.
    """

    def __init__(self, mode: Literal["step", "epoch"] = "step") -> None:
        self.mode = mode
        self.pbar: Optional[tqdm] = None

    def on_run_start(self, trainer_cache: dict[str, Any]) -> None:
        if self.mode == "epoch":
            self.pbar = tqdm(unit=" epochs", total=trainer_cache["epochs"])

    def on_step_end(self, trainer_cache: dict[str, Any]) -> None:
        if self.pbar is not None and self.mode == "step":
            self.pbar.update()

    def on_epoch_start(self, trainer_cache: dict[str, Any]) -> None:
        if self.mode == "step":
            self.pbar = tqdm(
                desc=f"Epoch {trainer_cache['t']}/{trainer_cache['epochs']}",
                unit=" steps",
                total=trainer_cache["train_steps"],
            )
        elif self.pbar is not None:
            self.pbar.update()

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        self._set_pbar_postfix(trainer_cache)
        if self.pbar is not None and self.mode == "step":
            self.pbar.close()

    def _set_pbar_postfix(self, trainer_cache: dict[str, Any]) -> None:
        stats = []
        for stat in trainer_cache.keys():
            if "loss" not in stat and "score" not in stat:
                continue
            stats.append(f"{stat}={trainer_cache[stat]:.4f}")
        if self.pbar is not None:
            self.pbar.set_postfix_str(", ".join(stats))


class Tensorboard(Callback):
    """Logs statistics to Tensorboard.

    Parameters
    ----------
    logdir : str
        Directory to save Tensorboard logs, e.g. "./runs/my_run".

    See Also
    --------
    :class:`compyute.nn.utils.tensorboard`
    """

    def __init__(self, logdir: str) -> None:
        self.writer = SummaryWriter(logdir=logdir)

    def _log_stats(self, stats: Iterable[str], trainer_cache: dict[str, Any]) -> None:
        for stat in stats:
            if stat not in trainer_cache:
                return
            if "val_" in stat:
                s = trainer_cache["t"]
            else:
                s = (
                    trainer_cache["t"] * trainer_cache["train_steps"]
                    + trainer_cache["step"]
                )
            self.writer.add_scalar(f"{stat}/stat", trainer_cache[stat], s)

    def on_step_end(self, trainer_cache: dict[str, Any]) -> None:
        # get stats that contain scores but are not val scores
        scores = [s for s in trainer_cache.keys() if "score" in s and "val_" not in s]
        self._log_stats(["loss", "lr"] + scores, trainer_cache)

    def on_epoch_end(self, trainer_cache: dict[str, Any]) -> None:
        # get all val stats
        stats = [s for s in trainer_cache.keys() if "val_" in s]
        self._log_stats(stats, trainer_cache)

    def on_run_end(self, trainer_cache: dict[str, Any]) -> None:
        self.writer.close()
