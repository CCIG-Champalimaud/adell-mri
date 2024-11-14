from typing import Callable

import torch
from tqdm import tqdm

from generative.inferers import DiffusionInferer


class DiffusionInfererSkipSteps(DiffusionInferer):
    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        skip_steps: int = 0,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            skip_steps: skips the first skip_steps steps
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: conditioning for network input.
            mode: conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        if not scheduler:
            scheduler = self.scheduler
        image = input_noise
        if verbose:
            progress_bar = tqdm(scheduler.timesteps[skip_steps:], leave=True)
        else:
            progress_bar = iter(scheduler.timesteps[skip_steps:])
        intermediates = []
        for t in progress_bar:
            # 1. predict noise model_output
            if mode == "concat":
                model_input = torch.cat([image, conditioning], dim=1)
                model_output = diffusion_model(
                    model_input,
                    timesteps=torch.Tensor((t,)).to(input_noise.device),
                    context=None,
                )
            else:
                model_output = diffusion_model(
                    image,
                    timesteps=torch.Tensor((t,)).to(input_noise.device),
                    context=conditioning,
                )

            # 2. compute previous image: x_t -> x_t-1
            image, _ = scheduler.step(model_output, t, image)
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)
        if save_intermediates:
            return image, intermediates
        else:
            return image

    @torch.no_grad()
    def sample_iter(
        self,
        input_noise: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        skip_steps: int = 0,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        tqdm_fn: Callable = tqdm,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            skip_steps: skips the first skip_steps steps
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        if not scheduler:
            scheduler = self.scheduler
        image = input_noise
        if verbose:
            progress_bar = tqdm_fn(scheduler.timesteps[skip_steps:])
        else:
            progress_bar = iter(scheduler.timesteps[skip_steps:])
        for t in progress_bar:
            # 1. predict noise model_output
            if mode == "concat":
                model_input = torch.cat([image, conditioning], dim=1)
                model_output = diffusion_model(
                    model_input,
                    timesteps=torch.Tensor((t,)).to(input_noise.device),
                    context=None,
                )
            else:
                model_output = diffusion_model(
                    image,
                    timesteps=torch.Tensor((t,)).to(input_noise.device),
                    context=conditioning,
                )

            # 2. compute previous image: x_t -> x_t-1
            image, _ = scheduler.step(model_output, t, image)
            yield image
