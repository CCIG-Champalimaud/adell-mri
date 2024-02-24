import torch
import torchmetrics as tmc
from tqdm import trange


def cat_if_necessary(
    tensor: torch.Tensor | list[torch.Tensor],
) -> torch.Tensor:
    """
    Concatenates a list into a torch.Tensor if the input is a list.

    Args:
        tensor (torch.Tensor | list[torch.Tensor]): tensor or list of tensors.

    Returns:
        torch.Tensor: concatenated tensor.
    """
    if isinstance(tensor, list):
        tensor = torch.cat(tensor, 0)
    return tensor


def bootstrap_metric(
    metric: tmc.Metric,
    samples: int = None,
    sample_size: float = 0.5,
    interval: float = 0.95,
    significance: float = 0.05,
    generator: torch.Generator = None,
    seed: int = 42,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(metric, "preds") is False or hasattr(metric, "target") is False:
        raise NotImplementedError(
            "Bootstrapping only possible if preds and target are defined in \
                metric"
        )
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    lower = (1 - interval) / 2
    upper = interval + lower
    preds = cat_if_necessary(metric.preds)
    target = cat_if_necessary(metric.target)
    n = preds.shape[0]
    if isinstance(sample_size, float):
        sample_size = int(sample_size * n)
    if samples is None:
        # uses the heuristic proposed in [1]
        # [1] https://www.tandfonline.com/doi/abs/10.1080/07474930008800459
        samples = int(20 / significance - 1)

    all_samples = []
    with trange(samples) as pbar:
        pbar.set_description("Calculating bootstrap")
        for _ in pbar:
            sample_idxs = torch.randperm(n, generator=generator)[:sample_size]
            preds_sample = preds[sample_idxs]
            target_sample = target[sample_idxs]
            metric.reset()
            metric.update(preds_sample, target_sample)
            metric_value = metric.compute()
            if len(metric_value.shape) == 0:
                metric_value = metric_value.unsqueeze(0)
            all_samples.append(metric_value)

    all_samples = torch.cat(all_samples, 0)
    mean = all_samples.mean(0)
    quantiles = torch.quantile(
        all_samples, torch.as_tensor([lower, upper]).to(all_samples), dim=0
    )
    if len(mean.shape) == 0:
        mean = mean.unsqueeze(-1)
        quantiles = quantiles.unsqueeze(-1)
    return mean, quantiles
