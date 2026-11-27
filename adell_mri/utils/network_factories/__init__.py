"""
PyTorch Lightning network factories.

This package contains factory functions for classification, segmentation,
self-supervised and generative networks. All public functions are re-exported
here for backwards compatibility.
"""

from adell_mri.utils.network_factories.classification import (
    get_classification_network,
    get_deconfounded_classification_network,
)
from adell_mri.utils.network_factories.generative import (
    get_gan_network,
    get_generative_network,
)
from adell_mri.utils.network_factories.segmentation import (
    get_segmentation_network,
)
from adell_mri.utils.network_factories.ssl import (
    get_ema,
    get_ssl_network,
    get_ssl_network_no_pl,
    get_stop_gradients,
)

__all__ = [
    "get_classification_network",
    "get_deconfounded_classification_network",
    "get_ema",
    "get_gan_network",
    "get_generative_network",
    "get_segmentation_network",
    "get_ssl_network",
    "get_ssl_network_no_pl",
    "get_stop_gradients",
]
