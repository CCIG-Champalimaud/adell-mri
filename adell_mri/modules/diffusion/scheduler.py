"""
DDPM scheduler with a patched ``step`` function.

The upstream implementation in
``generative.networks.schedulers.DDPMScheduler`` instantiates the noise
tensor on the CPU and then moves it to the target device
(``torch.randn(...).to(device)``), which dominates sampling time for
small models. This module subclasses the upstream scheduler and samples
the noise directly on the target device instead.
"""

import torch
from generative.networks.schedulers.ddpm import DDPMPredictionType
from generative.networks.schedulers.ddpm import DDPMScheduler as _DDPMScheduler


class DDPMScheduler(_DDPMScheduler):
    """
    Drop-in replacement for
    :class:`generative.networks.schedulers.DDPMScheduler` which avoids
    CPU-to-device transfers inside ``step`` by generating noise directly
    on the target device.
    """

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output: direct output from learned diffusion model.
            timestep: current discrete timestep in the diffusion chain.
            sample: current instance of sample being created by diffusion process.
            generator: random number generator.

        Returns:
            pred_prev_sample: Predicted previous sample
        """
        if model_output.shape[1] == sample.shape[
            1
        ] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(
                model_output, sample.shape[1], dim=1
            )
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[timestep - 1] if timestep > 0 else self.one
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.prediction_type == DDPMPredictionType.EPSILON:
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        elif self.prediction_type == DDPMPredictionType.SAMPLE:
            pred_original_sample = model_output
        elif self.prediction_type == DDPMPredictionType.V_PREDICTION:
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output

        # 3. Clip "predicted x_0"
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (
            alpha_prod_t_prev ** (0.5) * self.betas[timestep]
        ) / beta_prod_t
        current_sample_coeff = (
            self.alphas[timestep] ** (0.5) * beta_prod_t_prev / beta_prod_t
        )

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * sample
        )

        # 6. Add noise (sampled directly on the target device)
        variance = 0
        if timestep > 0:
            generator_device = (
                generator.device if generator is not None else None
            )
            if (
                generator_device is not None
                and generator_device.type != model_output.device.type
            ):
                # fall back to a transfer when the provided generator lives
                # on a different device type than the model output
                noise = torch.randn(
                    model_output.size(),
                    dtype=model_output.dtype,
                    layout=model_output.layout,
                    device=generator_device,
                    generator=generator,
                ).to(model_output.device)
            else:
                noise = torch.randn(
                    model_output.size(),
                    dtype=model_output.dtype,
                    layout=model_output.layout,
                    device=model_output.device,
                    generator=generator,
                )
            variance = (
                self._get_variance(
                    timestep, predicted_variance=predicted_variance
                )
                ** 0.5
            ) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample, pred_original_sample
