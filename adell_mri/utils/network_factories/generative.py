"""
Generative network factories.
"""

from typing import Any, Callable

from lightning.pytorch import LightningModule

from adell_mri.modules.diffusion.embedder import Embedder
from adell_mri.modules.gan.pl import GANPL
from adell_mri.utils.network_factories.common import compile_if_necessary
from adell_mri.utils.optimizer_factory import OPTIMIZER_EPS_DEFAULT


@compile_if_necessary
def get_generative_network(
    network_config: dict[str, Any],
    scheduler_config: dict[str, Any],
    categorical_specification: list[list[str] | int],
    numerical_specification: int,
    uncondition_proba: float,
    train_loader_call: Callable,
    max_epochs: int,
    warmup_steps: int,
    start_decay: int,
    diffusion_steps: int,
    concat_condition_key: str = None,
    optimizer_eps: float = OPTIMIZER_EPS_DEFAULT,
) -> LightningModule:
    """
    Returns a generative network.

    Args:
        network_config (dict[str, Any]): the configuration of the network.
        scheduler_config (dict[str, Any]): the configuration of the scheduler.
        categorical_specification (list[list[str] | int]): the categorical specification.
        numerical_specification (int): the numerical specification.
        uncondition_proba (float): the uncondition probability.
        train_loader_call (Callable): the training data loader.
        max_epochs (int): the maximum number of epochs.
        warmup_steps (int): the number of warmup steps.
        start_decay (int): the number of steps to start decay.
        diffusion_steps (int): the number of diffusion steps.

    Returns:
        A :class:`LightningModule` generative network.
    """
    try:
        import monai.networks.schedulers  # noqa: F401

        from adell_mri.modules.diffusion.inferer import (
            DiffusionInfererSkipSteps,
        )
        from adell_mri.modules.diffusion.pl import DiffusionUNetPL
        from adell_mri.modules.diffusion.scheduler import DDPMScheduler
    except ImportError:
        raise ImportError(
            "Please install the generative package to diffusion models"
            "(go to https://github.com/Project-MONAI/GenerativeModels for "
            "instructions)"
        )

    scheduler = DDPMScheduler(
        num_train_timesteps=diffusion_steps, **scheduler_config
    )
    inferer = DiffusionInfererSkipSteps(scheduler)
    if any(
        [
            categorical_specification is not None,
            numerical_specification is not None,
        ]
    ):
        embedder = Embedder(
            categorical_specification,
            numerical_specification,
            embedding_size=network_config["cross_attention_dim"],
        )
    else:
        embedder = None

    boilerplate_args = {
        "training_dataloader_call": train_loader_call,
        "image_key": "image",
        "cat_condition_key": None,
        "num_condition_key": None,
        "concat_condition_key": concat_condition_key,
        "n_epochs": max_epochs,
        "warmup_steps": warmup_steps,
        "start_decay": start_decay,
        "uncondition_proba": uncondition_proba,
        "optimizer_eps": optimizer_eps,
    }

    if categorical_specification is not None:
        boilerplate_args["cat_condition_key"] = "cat"
    if numerical_specification is not None:
        boilerplate_args["num_condition_key"] = "num"

    network = DiffusionUNetPL(
        inferer=inferer,
        scheduler=scheduler,
        embedder=embedder,
        **boilerplate_args,
        **network_config,
    )

    return network


@compile_if_necessary
def get_gan_network(
    network_config: dict[str, Any],
    generator_config: dict[str, Any],
    discriminator_config: dict[str, Any],
    training_dataloader_call: Callable,
    input_image_key: str,
    categorical_specification: list[list[str] | int] | None,
    numerical_specification: int | None,
    numerical_moments: tuple[list[float], list[float]] | None,
    max_epochs: int,
    steps_per_epoch: int,
    pct_start: int,
    optimizer_eps: float = OPTIMIZER_EPS_DEFAULT,
) -> LightningModule:
    """
    Returns a GAN network.

    Args:
        network_config (dict[str, Any]): the configuration of the network.
        generator_config (dict[str, Any]): the configuration of the generator.
        discriminator_config (dict[str, Any]): the configuration of the discriminator.
        training_dataloader_call (Callable): the training data loader.
        input_image_key (str): the key corresponding to the input image.
        categorical_specification (list[list[str] | int] | None): the categorical specification.
        numerical_specification (int | None): the numerical specification.
        numerical_moments (tuple[list[float], list[float]] | None): the numerical moments.
        max_epochs (int): the maximum number of epochs.
        steps_per_epoch (int): the number of steps per epoch.
        pct_start (int): the percentage of steps for warm-up.

    Returns:
        A :class:`LightningModule` GAN network.
    """
    try:
        from adell_mri.modules.gan.discriminator import Discriminator
        from adell_mri.modules.gan.unet_generator import UNetGenerator
    except ImportError:
        raise ImportError(
            "Please install the generative package to use gan models"
            "(go to https://github.com/Project-MONAI/GenerativeModels for "
            "instructions)"
        )
    boilerplate_args = {
        "real_image_key": "image",
        "classification_target_key": None,
        "regression_target_key": None,
        "epochs": max_epochs,
        "steps_per_epoch": steps_per_epoch,
        "pct_start": pct_start,
        "training_dataloader_call": training_dataloader_call,
        "class_target_specification": categorical_specification,
        "reg_target_specification": numerical_specification,
        "numerical_moments": numerical_moments,
        "optimizer_eps": optimizer_eps,
    }

    for key in [
        "lambda_gp",
        "lambda_feature_matching",
        "lambda_feature_map_matching",
        "lambda_identity",
        "n_critic",
        "momentum_beta1",
        "momentum_beta2",
        "learning_rate",
        "batch_size",
        "patch_size",
    ]:
        if key in network_config:
            boilerplate_args[key] = network_config[key]

    if categorical_specification is not None:
        boilerplate_args["classification_target_key"] = "cat"
        discriminator_config["additional_classification_targets"] = [
            x if isinstance(x, int) else len(x)
            for x in categorical_specification
        ]
    if numerical_specification is not None:
        num_spec = numerical_specification
        boilerplate_args["regression_target_key"] = "num"
        discriminator_config["additional_regression_targets"] = num_spec
    if input_image_key is not None:
        boilerplate_args["input_image_key"] = input_image_key

    boilerplate_args["generator"] = UNetGenerator(**generator_config)
    boilerplate_args["discriminator"] = Discriminator(**discriminator_config)

    if network_config.get("cycle_consistency", False) is True:
        if network_config.get("cycle_symmetry", False) is True:
            boilerplate_args = {
                **boilerplate_args,
                "generator_cycle": boilerplate_args["generator"],
                "discriminator_cycle": boilerplate_args["discriminator"],
                "cycle_consistency": True,
                "cycle_symmetry": True,
            }
        else:
            cycle_gen_conf = {k: generator_config[k] for k in generator_config}
            cycle_gen_conf["in_channels"] = generator_config["out_channels"]
            cycle_gen_conf["out_channels"] = generator_config["in_channels"]
            boilerplate_args = {
                **boilerplate_args,
                "generator_cycle": UNetGenerator(**cycle_gen_conf),
                "discriminator_cycle": Discriminator(**discriminator_config),
                "cycle_consistency": True,
            }

    network = GANPL(**boilerplate_args)

    return network
