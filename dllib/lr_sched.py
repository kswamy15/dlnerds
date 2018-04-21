from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Optimizer

class myCosineAnnealingLR(CosineAnnealingLR):
    def __init__(self, optimizer, T_max, eta_min=0, cycle_len=None, apply_batch = True, last_epoch=-1):
        self.cycle_len = cycle_len
        super(myCosineAnnealingLR, self).__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if not (self.cycle_len is None) and self.cycle_len == self.last_epoch:
            self.last_epoch = 0
        super(myCosineAnnealingLR, self).get_lr()