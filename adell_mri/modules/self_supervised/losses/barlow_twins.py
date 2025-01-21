import torch


class BarlowTwinsLoss(torch.nn.Module):
    """
    The Barlow Twins loss. It works contrastively and attempts to maximise
    the feature correlation between different views of the same instance and
    minimises the correlation to other instances.
    """

    def __init__(self, moving: bool = False, lam=0.2):
        """
        Args:
            moving (bool, optional): whether the average should be calculated
                as a moving average. Defaults to False.
            lam (float, optional): weight for minimisation of correlation with
                other instances. Defaults to 0.2.
        """
        super().__init__()
        self.moving = moving
        self.lam = lam

        self.count = 0.0
        self.sum = None
        self.sum_of_squares = None
        self.average = None
        self.std = None

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        if self.moving is False and self.sum is None:
            o = torch.divide(
                x - torch.mean(x, 0, keepdim=True),
                torch.std(x, 0, keepdim=True),
            )
        else:
            o = torch.divide(x - self.average, self.std)
        return o

    def pearson_corr(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = x.flatten(start_dim=1), y.flatten(start_dim=1)
        x, y = self.standardize(x), self.standardize(y)
        x, y = x.unsqueeze(1), y.unsqueeze(0)
        n = torch.sum(x * y, axis=-1)
        d = torch.multiply(torch.norm(x, 2, -1), torch.norm(y, 2, -1))
        return n / d

    def calculate_loss(self, x, y, update=True):
        if update is True:
            n = x.shape[0]
            f = x.shape[1]
            if self.sum is None:
                self.sum = torch.zeros([1, f], device=x.device)
                self.sum_of_squares = torch.zeros([1, f], device=x.device)
            self.sum = torch.add(self.sum, torch.sum(x + y, 0, keepdim=True))
            self.sum_of_squares = torch.add(
                self.sum_of_squares,
                torch.sum(torch.square(x) + torch.square(y), 0, keepdim=True),
            )
            self.count += 2 * n
        return self.barlow_twins_loss(x, y)

    def barlow_twins_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        diag_idx = torch.arange(0, x.shape[0])
        n = x.shape[0]
        C = self.pearson_corr(x, y)
        inv_term = torch.diagonal(1 - C)
        red_term = torch.square(C)
        red_term[diag_idx, diag_idx] = 0
        loss = torch.add(inv_term.sum() / n, red_term.sum() / n * self.lam)
        return loss

    def calculate_average_std(self):
        self.average = self.sum / self.count
        self.std = self.sum_of_squares - torch.square(self.sum) / self.count

    def reset(self):
        self.count = 0.0
        self.sum[()] = 0
        self.sum_of_squares[()] = 0

    def forward(self, X1: torch.Tensor, X2: torch.Tensor, update: bool = True):
        loss = self.calculate_loss(X1, X2, update)
        return loss.sum()
