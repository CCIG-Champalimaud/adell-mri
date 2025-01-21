"""
Implementation of the generic form of the DINO loss. Some of this code 
(particularly concerning distributed training and SK centering) was adapted
from [1], the official implementation of the DinoV2 method. To the best of my
knowledge, this is on its own based on the original implementation available in
[2].

[1] https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/loss/dino_clstoken_loss.py
[2] https://github.com/facebookresearch/dino/blob/main/main_dino.py
"""

import torch
import torch.distributed as dist


class DinoLoss(torch.nn.Module):
    """
    DINO loss.

    Formally equivalent to a cross-entropy between two tensors. The second
    (typically corresponding to the teacher tensor) can be centered according
    to either EMA-updated centers (teacher_score_method = "center") or
    through Sinkhorn-Knopp centering (teacher_score_method = "sk"). The class
    has been implemented/adapted such that it is possible to perform updates
    in a distributed fashion.
    """

    def __init__(
        self,
        temperatures: float | tuple[float, float],
        n_features: int,
        center_m: float = 0.9,
        teacher_score_method: str = "center",
        sk_iterations: int = 3,
    ):
        super().__init__()
        self.temperatures = temperatures
        self.n_features = n_features
        self.center_m = center_m
        self.teacher_score_method = teacher_score_method
        self.sk_iterations = sk_iterations

        assert self.teacher_score_method in ["center", "sk"]

        if isinstance(self.temperatures, float):
            self.temperatures = [self.temperatures, self.temperatures]

        self.t1 = self.temperatures[0]
        self.t2 = self.temperatures[1]

        self.register_buffer("centers", torch.zeros([self.n_features]))
        self.updated = False
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    @property
    def world_size(self):
        return dist.get_world_size() if dist.is_initialized() else 1

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = self.world_size

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.centers = self.centers * self.center_m + _t * (1 - self.center_m)

            self.updated = True

    @torch.no_grad()
    def update_centers(self, x: torch.Tensor):
        self.reduce_center_update(x)

    @torch.no_grad()
    def reduce_center_update(self, x: torch.Tensor):
        self.updated = False
        if len(x.shape) > 2:
            x = x.flatten(end_dim=-2)
        self.len_teacher_output = len(x)
        self.async_batch_center = torch.sum(x, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    def get_scores_teacher(self, x: torch.Tensor):
        if self.teacher_score_method == "center":
            self.apply_center_update()
            return torch.softmax(x - self.centers / self.t2, dim=-1)
        elif self.teacher_score_method == "sk":
            return self.sinkhorn_knopp_teacher(x)

    def get_log_scores_student(self, x: torch.Tensor):
        return torch.log_softmax(x / self.t1, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, x: torch.Tensor):
        x = x.float()
        # .t() because Q is K-by-B for consistency with notations with paper
        x = x.flatten(end_dim=-2)
        Q = torch.exp(x / self.t2).t()
        B = Q.shape[1] * self.world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if self.world_size > 1:
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for _ in range(self.sk_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if self.world_size > 1:
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        loss = (
            torch.sum(
                torch.multiply(
                    self.get_scores_teacher(b).flatten(end_dim=-2),
                    self.get_log_scores_student(a).flatten(end_dim=-2),
                ),
                dim=-1,
            )
            .negative()
            .mean()
        )
        return loss
