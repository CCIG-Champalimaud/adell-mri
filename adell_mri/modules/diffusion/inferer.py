from typing import Callable, Iterator

import torch
from generative.inferers import DiffusionInferer
from tqdm import tqdm


class DiffusionInfererSkipSteps(DiffusionInferer):
    def __call__(
        self,
        inputs: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
        concat_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for a supervised training iteration. If
        ``concat_condition`` is provided, the conditioning is concatenated
        along the channel dimension of the noisy input (concat conditioning
        mode). It can be combined with ``condition`` so that both
        channel-concatenated and cross-attention conditioning are used
        simultaneously.

        Args:
            inputs (torch.Tensor): input image to which noise is added.
            diffusion_model (Callable): diffusion model.
            noise (torch.Tensor): random noise, of the same shape as the
                input.
            timesteps (torch.Tensor): random timesteps.
            condition (torch.Tensor | None, optional): conditioning context
                for the network. Defaults to None.
            concat_condition (torch.Tensor | None, optional): conditioning
                tensor concatenated to the noisy input along the channel
                dimension. Defaults to None.

        Returns:
            torch.Tensor: predicted noise.
        """
        noisy_image = self.scheduler.add_noise(
            original_samples=inputs, noise=noise, timesteps=timesteps
        )
        if concat_condition is not None:
            model_input = torch.cat([noisy_image, concat_condition], dim=1)
            prediction = diffusion_model(
                x=model_input, timesteps=timesteps, context=condition
            )
        else:
            prediction = diffusion_model(
                x=noisy_image, timesteps=timesteps, context=condition
            )
        return prediction

    def _predict(
        self,
        diffusion_model: Callable[..., torch.Tensor],
        image: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: torch.Tensor | None,
        concat_condition: torch.Tensor | None,
        unconditioning: torch.Tensor | None = None,
        guidance_strength: float | None = None,
    ) -> torch.Tensor:
        """
        Predicts the noise at a given timestep, combining cross-attention
        context (``conditioning``) and channel-concatenated conditioning
        (``concat_condition``) simultaneously. If ``unconditioning`` and
        ``guidance_strength`` are provided, applies classifier-free guidance
        over the cross-attention context (the concatenated conditioning is kept
        unchanged for both branches).

        Args:
            diffusion_model (Callable): diffusion model.
            image (torch.Tensor): current noisy image.
            timestep (torch.Tensor): current timestep.
            conditioning (torch.Tensor | None): cross-attention context.
            concat_condition (torch.Tensor | None): conditioning tensor
                concatenated along the channel dimension.
            unconditioning (torch.Tensor | None, optional): unconditioned
                context for classifier-free guidance. Defaults to None.
            guidance_strength (float | None, optional): classifier-free
                guidance strength. Defaults to None.

        Returns:
            torch.Tensor: predicted noise.
        """
        use_guidance = (
            guidance_strength is not None
            and conditioning is not None
            and unconditioning is not None
        )
        if use_guidance:
            b = image.shape[0]
            model_input = torch.cat([image, image], dim=0)
            if concat_condition is not None:
                model_input = torch.cat(
                    [
                        model_input,
                        torch.cat([concat_condition, concat_condition], dim=0),
                    ],
                    dim=1,
                )
            context = torch.cat([conditioning, unconditioning], dim=0)
            model_output = diffusion_model(
                x=model_input,
                timesteps=timestep,
                context=context,
            )
            return torch.subtract(
                (1.0 + guidance_strength) * model_output[:b],
                guidance_strength * model_output[b:],
            )
        if concat_condition is not None:
            image = torch.cat([image, concat_condition], dim=1)
        return diffusion_model(
            x=image,
            timesteps=timestep,
            context=conditioning,
        )

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
        concat_condition: torch.Tensor | None = None,
        unconditioning: torch.Tensor | None = None,
        guidance_strength: float | None = None,
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
                network input (cross-attention context). Defaults to None.
            concat_condition (torch.Tensor | None, optional): conditioning
                tensor concatenated along the channel dimension of the noisy
                input. Can be combined with ``conditioning`` (mixed
                conditioning). Defaults to None.
            unconditioning (torch.Tensor | None, optional): unconditioning
                context for network input (used only in classifier-free
                guidance). Defaults to None.
            guidance_strength (float | None, optional): strength of
                classifier-free guidance. Defaults to None.
            verbose (bool, optional): if true, prints the progression bar of the
                sampling process.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]: sampled
                image or sampled image and intermediates.
        """
        if not scheduler:
            scheduler = self.scheduler
        image = input_noise
        if verbose:
            progress_bar = tqdm(
                scheduler.timesteps[skip_steps:],
                leave=False,
                ncols=80,
                mininterval=1,
            )
        else:
            progress_bar = iter(scheduler.timesteps[skip_steps:])
        intermediates = []
        for t in progress_bar:
            # 1. predict noise model_output
            model_output = self._predict(
                diffusion_model,
                image,
                torch.tensor((t,), device=image.device),
                conditioning=conditioning,
                concat_condition=concat_condition,
                unconditioning=unconditioning,
                guidance_strength=guidance_strength,
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
        concat_condition: torch.Tensor | None = None,
        unconditioning: torch.Tensor | None = None,
        guidance_strength: float | None = None,
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
            conditioning (torch.Tensor | None, optional): conditioning for
                network input. Defaults to None.
            concat_condition (torch.Tensor | None, optional): conditioning
                tensor concatenated along the channel dimension of the noisy
                input. Can be combined with ``conditioning`` (mixed
                conditioning). Defaults to None.
            unconditioning (torch.Tensor | None, optional): unconditioning
                context for network input (used only in classifier-free
                guidance). Defaults to None.
            guidance_strength (float | None, optional): strength of
                classifier-free guidance. Defaults to None.
            verbose (bool, optional): if true, prints the progression bar of the
                sampling process.
            tqdm_fn (Callable, optional): function to use for the progress bar.
                Defaults to tqdm.

        Yields:
            torch.Tensor: sampled image at all steps.
        """
        if not scheduler:
            scheduler = self.scheduler
        image = input_noise
        if verbose:
            progress_bar = tqdm_fn(scheduler.timesteps[skip_steps:])
        else:
            progress_bar = iter(scheduler.timesteps[skip_steps:])
        for t in progress_bar:
            # 1. predict noise model_output
            model_output = self._predict(
                diffusion_model,
                image,
                torch.tensor((t,), device=input_noise.device),
                conditioning=conditioning,
                concat_condition=concat_condition,
                unconditioning=unconditioning,
                guidance_strength=guidance_strength,
            )
            # 2. compute previous image: x_t -> x_t-1
            image, _ = scheduler.step(model_output, t, image)
            yield image
