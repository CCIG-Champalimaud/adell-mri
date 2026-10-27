"""
Hard-coded diffusion recipes for voxel-space DDPM and latent diffusion.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DiffusionRecipe:
    """
    Diffusion settings that are tied to a modelling choice.

    Args:
        name (str): recipe name.
        diffusion_steps (int): number of diffusion steps.
        scheduler_config (dict[str, Any]): keyword arguments for
            :class:`adell_mri.modules.diffusion.scheduler.DDPMScheduler`.
        min_snr_gamma (float | None, optional): Min-SNR clipping value. None
            disables Min-SNR weighting.
        latent_std_decay (float | None, optional): decay for the latent
            standard deviation EMA. Only used for latent diffusion.
    """

    name: str
    diffusion_steps: int
    scheduler_config: dict[str, Any] = field(default_factory=dict)
    min_snr_gamma: float | None = None
    latent_std_decay: float | None = None

    def resolve(
        self,
        config: dict[str, Any] | None = None,
        diffusion_steps: int | None = None,
    ) -> tuple[dict[str, Any], int, float | None, float | None]:
        """
        Resolves recipe defaults with optional configuration overrides.

        Args:
            config (dict[str, Any] | None, optional): configuration dictionary
                which may contain ``scheduler``, ``diffusion_steps``,
                ``min_snr_gamma`` and ``latent_std_decay`` overrides. Defaults
                to None.
            diffusion_steps (int | None, optional): command-line override for
                the number of diffusion steps. Defaults to None.

        Returns:
            tuple[dict[str, Any], int, float | None, float | None]: resolved
                scheduler configuration, diffusion steps, Min-SNR gamma and
                latent standard deviation decay.
        """
        config = config or {}
        scheduler_config = {
            **self.scheduler_config,
            **config.get("scheduler", {}),
        }
        steps = (
            diffusion_steps
            or config.get("diffusion_steps")
            or self.diffusion_steps
        )
        min_snr_gamma = config.get("min_snr_gamma", self.min_snr_gamma)
        latent_std_decay = config.get("latent_std_decay", self.latent_std_decay)
        return scheduler_config, steps, min_snr_gamma, latent_std_decay

    def to_metadata(
        self,
        scheduler_config: dict[str, Any] | None = None,
        diffusion_steps: int | None = None,
    ) -> dict[str, Any]:
        """
        Returns the recipe settings that should be stored in a checkpoint.

        Args:
            scheduler_config (dict[str, Any] | None, optional): resolved
                scheduler configuration. Defaults to None (recipe default).
            diffusion_steps (int | None, optional): resolved number of
                diffusion steps. Defaults to None (recipe default).

        Returns:
            dict[str, Any]: recipe name, scheduler settings and diffusion
                steps.
        """
        return {
            "recipe": self.name,
            "scheduler_config": dict(scheduler_config or self.scheduler_config),
            "diffusion_steps": diffusion_steps or self.diffusion_steps,
        }


DDPM_RECIPE = DiffusionRecipe(
    name="ddpm",
    diffusion_steps=1000,
    scheduler_config={
        "schedule": "linear_beta",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "prediction_type": "epsilon",
        "clip_sample": True,
    },
    min_snr_gamma=None,
)

LDM_RECIPE = DiffusionRecipe(
    name="ldm",
    diffusion_steps=1000,
    scheduler_config={
        "schedule": "scaled_linear_beta",
        "beta_start": 0.0005,
        "beta_end": 0.0195,
        "prediction_type": "v_prediction",
        "clip_sample": False,
    },
    min_snr_gamma=5.0,
    latent_std_decay=0.99,
)

RECIPES = {
    DDPM_RECIPE.name: DDPM_RECIPE,
    LDM_RECIPE.name: LDM_RECIPE,
}


def get_recipe(name: str) -> DiffusionRecipe:
    """
    Returns a diffusion recipe by name.

    Args:
        name (str): recipe name (``ddpm`` or ``ldm``).

    Returns:
        DiffusionRecipe: the requested recipe.

    Raises:
        ValueError: if the recipe is unknown.
    """
    if name not in RECIPES:
        raise ValueError(
            f"Unknown diffusion recipe '{name}'. Valid recipes are "
            f"{sorted(RECIPES)}"
        )
    return RECIPES[name]
