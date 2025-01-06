from typing import Callable, Iterator

import torch
from tqdm import tqdm

from adell_mri.modules.gan.losses import cat_if_none
from generative.inferers import DiffusionInferer


class DiffusionInfererSkipSteps(DiffusionInferer):
    @torch.inference_mode()
    def sample(
        self,
        input_noise: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        skip_steps: int = 0,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        unconditioning: torch.Tensor | None = None,
        guidance_strength: float | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise (torch.Tensor, optional): random noise, of the same
                shape as the desired sample.
            diffusion_model (Callable): model to sample from.
            scheduler (Callable | None, optional): diffusion scheduler. If none
                provided will use the class attribute scheduler. Defaults to
                None.
            skip_steps (int, optional): skips the first skip_steps steps.
                Defaults to 0.
            save_intermediates (bool | None, optional): whether to return
                intermediates along the sampling change. Defaults to False.
            intermediate_steps (int | None, optional): if save_intermediates is
                True, saves every n steps. Defaults to 100.
            conditioning (torch.Tensor | None, optional): conditioning for
                network input. Defaults to None.
            unconditioning (torch.Tensor | None, optional): unconditioning
                context for network input (used only in classifier-free
                guidance). Defaults to None.
            guidance_strength (float | None, optional): strength of
                classifier-free guidance. Defaults to None.
            mode (str, optional): conditioning mode for the network. Defaults to
                "crossattn".
            verbose (bool, optional): if true, prints the progression bar of the
                sampling process.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]: sampled
                image or sampled image and intermediates.
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
        if mode == "concat":
            for t in progress_bar:
                # 1. predict noise model_output
                model_input = torch.cat([image, conditioning], dim=1)
                model_output = diffusion_model(
                    model_input,
                    timesteps=torch.tensor((t,), device=input_noise.device),
                    context=None,
                )
        else:
            if unconditioning is not None:
                conditioning = torch.cat([conditioning, unconditioning], dim=0)
            for t in progress_bar:
                if guidance_strength is None:
                    model_output = diffusion_model(
                        image,
                        timesteps=torch.tensor((t,), device=input_noise.device),
                        context=conditioning,
                    )
                elif unconditioning is not None:
                    b = image.shape[0]
                    model_input = torch.cat([image, image], dim=0)
                    model_output = diffusion_model(
                        model_input,
                        timesteps=torch.tensor((t,), device=input_noise.device),
                        context=conditioning,
                    )
                    model_output = torch.subtract(
                        (1.0 + guidance_strength) * model_output[:b],
                        guidance_strength * model_output[b:],
                    )
                else:
                    model_output = diffusion_model(
                        image,
                        timesteps=torch.tensor((t,), device=input_noise.device),
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

    @torch.inference_mode()
    def sample_iter(
        self,
        input_noise: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        skip_steps: int = 0,
        conditioning: torch.Tensor | None = None,
        unconditioning: torch.Tensor | None = None,
        guidance_strength: float | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        tqdm_fn: Callable = tqdm,
    ) -> Iterator[torch.Tensor]:
        """
                Args:
            input_noise (torch.Tensor, optional): random noise, of the same
                shape as the desired sample.
            diffusion_model (Callable): model to sample from.
            scheduler (Callable | None, optional): diffusion scheduler. If none
                provided will use the class attribute scheduler. Defaults to
                None.
            skip_steps (int, optional): skips the first skip_steps steps.
                Defaults to 0.
            save_intermediates (bool | None, optional): whether to return
                intermediates along the sampling change. Defaults to False.
            intermediate_steps (int | None, optional): if save_intermediates is
                True, saves every n steps. Defaults to 100.
            conditioning (torch.Tensor | None, optional): conditioning for
                network input. Defaults to None.
            unconditioning (torch.Tensor | None, optional): unconditioning
                context for network input (used only in classifier-free
                guidance). Defaults to None.
            guidance_strength (float | None, optional): strength of
                classifier-free guidance. Defaults to None.
            mode (str, optional): conditioning mode for the network. Defaults to
                "crossattn".
            verbose (bool, optional): if true, prints the progression bar of the
                sampling process.
            tqdm_fn (Callable, optional): function to use for the progress bar.
                Defaults to tqdm.

        Yields:
            torch.Tensor: sampled image at all steps.
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
        if mode == "concat":
            for t in progress_bar:
                # 1. predict noise model_output
                model_input = torch.cat([image, conditioning], dim=1)
                model_output = diffusion_model(
                    model_input,
                    timesteps=torch.tensor((t,), device=input_noise.device),
                    context=None,
                )
        else:
            if unconditioning is not None:
                conditioning = torch.cat([conditioning, unconditioning], dim=0)
            for t in progress_bar:
                if guidance_strength is None:
                    model_output = diffusion_model(
                        image,
                        timesteps=torch.tensor((t,), device=input_noise.device),
                        context=conditioning,
                    )
                elif unconditioning is not None:
                    b = image.shape[0]
                    image = torch.cat([image, image], dim=0)
                    model_output = diffusion_model(
                        image,
                        timesteps=torch.tensor((t,), device=input_noise.device),
                        context=conditioning,
                    )
                    model_output = torch.subtract(
                        (1.0 + guidance_strength) * model_output[:b],
                        guidance_strength * model_output[b:],
                    )
                else:
                    model_output = diffusion_model(
                        image,
                        timesteps=torch.tensor((t,), device=input_noise.device),
                        context=conditioning,
                    )

                # 2. compute previous image: x_t -> x_t-1
                image, _ = scheduler.step(model_output, t, image)
                yield image
