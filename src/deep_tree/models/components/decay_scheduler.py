"""CosineAnnealingWarmRestarts with decaying restart LR"""

from typing import Optional, Union, List
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import Optimizer


class DecayCosineAnnealingWarmRestarts(CosineAnnealingWarmRestarts):
    """
    CosineAnnealingWarmRestarts with decaying restart LR.

    After each cycle, max_lr is multiplied by decay_factor
    so restarts don’t jump back to the original full LR.
    """

    def __init__(
            self,
            optimizer: Optimizer,
            T_0: int,
            T_mult: int | float = 1,
            eta_min: float = 0.0,
            last_epoch: int = -1,
            decay_factor: float = 0.5,
            decay_from_initial: bool = True
    ) -> None:
        self.decay_factor = decay_factor
        self.decay_from_initial = decay_from_initial

        # flag to skip first step, so the decaying does not start immediately
        self._has_started = False

        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)
        self.base_lrs_initial: List[float] = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch: Optional[Union[int, float]] = None) -> None:
        """
        Perform one scheduler step and decay LR at restarts.
        """
        # Check if a restart just happened (skip the very first step)
        restart = self._has_started and self.T_cur == 0

        # Call the parent step
        super().step(epoch)

        # Apply decay if a restart occurred
        if restart:
            if self.decay_from_initial:
                # Decay relative to original LR
                cycle_num = self._get_cycle_number()
                new_base_lrs = [lr0 * self.decay_factor ** cycle_num
                                for lr0 in self.base_lrs_initial]
            else:
                # Decay relative to last cycle
                new_base_lrs = [lr * self.decay_factor for lr in self.base_lrs]

            self.base_lrs = new_base_lrs
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
                group["lr"] = base_lr

        self._has_started = True

    def _get_cycle_number(self) -> int:
        """
        Compute the number of completed cycles based on last_epoch and T_0/T_mult.
        """
        if self.T_mult == 1:
            return self.last_epoch // self.T_0
        n = 0
        T_curr = self.T_0           # pylint: disable=C0103
        epoch = self.last_epoch
        while epoch >= T_curr:
            epoch -= T_curr
            T_curr *= self.T_mult   # pylint: disable=C0103
            n += 1
        return n
