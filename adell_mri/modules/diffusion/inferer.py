from typing import Callable, Iterator

import torch
from generative.inferers import DiffusionInferer
from rich.progress import track
from tqdm import tqdm

from adell_mri.utils.logging import get_progress


class _SamplingBuffers:
    """
    Pre-allocated tensors reused across the iterations of the sampling
    loop (timestep placeholder, network input with channel-concatenated
    conditioning, classifier-free guidance context and output). Constant
    entries (conditioning slices) are filled once on construction; only
    the entries which depend on the current iterate are refreshed at
    each step.
    """

    def __init__(
        self,
        image: torch.Tensor,
        conditioning: torch.Tensor | None,
        unconditioning: torch.Tensor | None,
        concat_condition: torch.Tensor | None,
        use_guidance: bool,
    ):
        self.batch_size = image.shape[0]
        self.image_channels = image.shape[1]
        self.timestep = torch.zeros(1, dtype=torch.long, device=image.device)
        n = self.batch_size * (2 if use_guidance else 1)
        if concat_condition is None and not use_guidance:
            self.model_input = None
            self.context = None
            self.prediction = None
            return
        in_channels = self.image_channels + (
            concat_condition.shape[1] if concat_condition is not None else 0
        )
        self.model_input = torch.empty(
            (n, in_channels, *image.shape[2:]),
            dtype=image.dtype,
            device=image.device,
        )
        if concat_condition is not None:
            offset = self.image_channels
            if use_guidance:
                self.model_input[: self.batch_size, offset:].copy_(
                    concat_condition
                )
                self.model_input[self.batch_size :, offset:].copy_(
                    concat_condition
                )
            else:
                self.model_input[:, offset:].copy_(concat_condition)
        if use_guidance:
            self.context = torch.empty(
                (n, *conditioning.shape[1:]),
                dtype=conditioning.dtype,
                device=conditioning.device,
            )
            self.context[: self.batch_size].copy_(conditioning)
            self.context[self.batch_size :].copy_(unconditioning)
        else:
            self.context = None
        self.prediction = None

    def get_model_input(self, image: torch.Tensor) -> torch.Tensor:
        """
        Returns the pre-allocated network input updated with the current
        noisy image (or the image itself if no buffers are needed).

        Args:
            image (torch.Tensor): current noisy image.

        Returns:
            torch.Tensor: network input.
        """
        if self.model_input is None:
            return image
        b = self.batch_size
        c = self.image_channels
        if self.context is not None:
            self.model_input[:b, :c].copy_(image)
            self.model_input[b:, :c].copy_(image)
        else:
            self.model_input[:, :c].copy_(image)
        return self.model_input

    def combine_guidance(
        self, model_output: torch.Tensor, guidance_strength: float
    ) -> torch.Tensor:
        """
        Combines conditional and unconditional predictions using
        classifier-free guidance without intermediate allocations:

        ``(1 + guidance_strength) * cond - guidance_strength * uncond``

        Args:
            model_output (torch.Tensor): doubled-batch network output
                (conditional followed by unconditional predictions).
            guidance_strength (float): classifier-free guidance strength.

        Returns:
            torch.Tensor: guided prediction.
        """
        b = self.batch_size
        conditional = model_output[:b]
        unconditional = model_output[b:]
        if (
            self.prediction is None
            or self.prediction.shape != conditional.shape
            or self.prediction.dtype != conditional.dtype
        ):
            self.prediction = torch.empty_like(conditional)
        prediction = self.prediction
        if torch.is_grad_enabled() and (
            conditional.requires_grad or unconditional.requires_grad
        ):
            return torch.subtract(
                (1.0 + guidance_strength) * conditional,
                guidance_strength * unconditional,
            )
        torch.subtract(conditional, unconditional, out=prediction)
        prediction.mul_(guidance_strength)
        prediction.add_(conditional)
        return prediction


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
        buffers: _SamplingBuffers | None = None,
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
            buffers (_SamplingBuffers | None, optional): pre-allocated
                buffers reused across sampling steps. If not provided,
                required intermediates are allocated on the fly. Defaults
                to None.

        Returns:
            torch.Tensor: predicted noise.
        """
        use_guidance = (
            guidance_strength is not None
            and conditioning is not None
            and unconditioning is not None
        )
        if use_guidance:
            if buffers is None:
                buffers = _SamplingBuffers(
                    image=image,
                    conditioning=conditioning,
                    unconditioning=unconditioning,
                    concat_condition=concat_condition,
                    use_guidance=True,
                )
            model_output = diffusion_model(
                x=buffers.get_model_input(image),
                timesteps=timestep,
                context=buffers.context,
            )
            return buffers.combine_guidance(model_output, guidance_strength)
        if concat_condition is not None:
            if buffers is None or buffers.context is not None:
                buffers = _SamplingBuffers(
                    image=image,
                    conditioning=None,
                    unconditioning=None,
                    concat_condition=concat_condition,
                    use_guidance=False,
                )
            model_input = buffers.get_model_input(image)
        else:
            model_input = image
        return diffusion_model(
            x=model_input,
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
        use_guidance = (
            guidance_strength is not None
            and conditioning is not None
            and unconditioning is not None
        )
        buffers = _SamplingBuffers(
            image=image,
            conditioning=conditioning,
            unconditioning=unconditioning,
            concat_condition=concat_condition,
            use_guidance=use_guidance,
        )
        timestep = buffers.timestep
        progress = get_progress(transient=True, disable=not verbose)
        intermediates = []
        with progress:
            for t in progress.track(
                scheduler.timesteps[skip_steps:],
                description="Generating image...",
            ):
                # 1. predict noise model_output
                timestep.fill_(t)
                model_output = self._predict(
                    diffusion_model,
                    image,
                    timestep,
                    conditioning=conditioning,
                    concat_condition=concat_condition,
                    unconditioning=unconditioning,
                    guidance_strength=guidance_strength,
                    buffers=buffers,
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
        use_guidance = (
            guidance_strength is not None
            and conditioning is not None
            and unconditioning is not None
        )
        buffers = _SamplingBuffers(
            image=image,
            conditioning=conditioning,
            unconditioning=unconditioning,
            concat_condition=concat_condition,
            use_guidance=use_guidance,
        )
        timestep = buffers.timestep
        if verbose:
            progress_bar = tqdm_fn(scheduler.timesteps[skip_steps:])
        else:
            progress_bar = iter(scheduler.timesteps[skip_steps:])
        for t in progress_bar:
            # 1. predict noise model_output
            timestep.fill_(t)
            model_output = self._predict(
                diffusion_model,
                image,
                timestep,
                conditioning=conditioning,
                concat_condition=concat_condition,
                unconditioning=unconditioning,
                guidance_strength=guidance_strength,
                buffers=buffers,
            )
            # 2. compute previous image: x_t -> x_t-1
            image, _ = scheduler.step(model_output, t, image)
            yield image
