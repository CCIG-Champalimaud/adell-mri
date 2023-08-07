"""
Diffusion classes to train diffusion models. 

Based on:
    https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/ddpm.py
    https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/ddpm_conditional.py
"""

import torch
from tqdm import tqdm

from typing import Union,Tuple,List

class Diffusion:
    """
    Diffusion process for diffusion models.

    For a given timestep t, we can diffuse an image X using ``noise_images``:

        $\sqrt{\hat{\alpha}} * X + \sqrt{1 - \hat{alpha}} * \epsilon$,

    where $\hat{\alpha}$ is the cummulative product of a linear space between
    beta_start and beta_end at timestep $t$, and $\epsilon$ is a noise image.
    In other words, the diffused image at timestep t is the weighted sum of that
    same image and a noise vector of the same size.

    To recover an image, a noise image is fed to a given model in ``sample``.
    This works by using a ``model`` to predict \epsilon and reverting the above
    equation a total of $n$ timesteps.

    This also has support for input conditioning (``x`` in ``sample``) and 
    class conditioning (``classification`` in ``sample``).
    """
    def __init__(self,
                 noise_steps: int=1000,
                 beta_start: float=1e-4,
                 beta_end: float=0.02,
                 img_size: Union[Tuple[int,int],Tuple[int,int,int]]=(256,256),
                 device: str="cuda"):
        """
        Args:
            noise_steps (int, optional): number of steps in diffusion process. 
                Defaults to 1000.
            beta_start (float, optional): initial amount of noise. Defaults to
                1e-4.
            beta_end (float, optional): final amount of noise. Defaults to 
                0.02.
            img_size (Union[Tuple[int,int],Tuple[int,int,int]], optional): size
                of image. Defaults to (256,256).
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size

        self.n_dim = len(img_size)

        self.check_arguments()

        self.beta = self.prepare_noise_schedule()
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def check_arguments(self):
        """
        Checks argument types and conditions. Self explanatory.
        """
        if isinstance(self.noise_steps,int) == False:
            raise TypeError("noise_steps must be int")
        if isinstance(self.beta_start,float) == False:
            raise TypeError("beta_start must be float")
        if isinstance(self.beta_end,float) == False:
            raise TypeError("beta_end must be float")
        if isinstance(self.img_size,(tuple,list)) == False:
            raise TypeError("img_size must be tuple")
        else:
            if len(self.img_size) not in [2,3]:
                raise TypeError("img_size must be tuple with size 2 or 3")
            if any([isinstance(x,int) == False for x in self.img_size]) is True:
                raise TypeError("img_size must be tuple of int")

    def prepare_noise_schedule(self)->torch.Tensor:
        """
        Returns a noise schedule between ``beta_start`` and ``beta_end``.

        Returns:
            torch.Tensor: tensor with size [self.noise_steps] going linearly
                from ``beta_start`` to ``beta_end``.
        """
        return torch.linspace(start=self.beta_start, 
                              end=self.beta_end, 
                              steps=self.noise_steps)

    def get_shape(self, x: torch.Tensor=None)->List[int]:
        """
        Convenience function to return correct shape for alpha vectors.

        Args:
            x (torch.Tensor, optional): input image to infer number of 
                dimensions. Defaults to None (uses ``n_dim``).

        Returns:
            List[int]: vector with the new shape for alpha vectors.
        """
        if x is None:
            n_dim = self.n_dim + 2
        else:
            n_dim = len(x.shape)
        new_shape = (-1,*[1 for _ in range(1,n_dim)])
        return new_shape

    def noise_images(self, x: torch.Tensor, t: int)->torch.Tensor:
        """
        Introduces noise to images x at timestep t.

        Args:
            x (torch.Tensor): input images.
            t (int): timestep.

        Returns:
            torch.Tensor: noised (diffused) images.
        """
        sh = self.get_shape(x)
        self.alpha_hat = self.alpha_hat.to(x.device)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).reshape(sh)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).reshape(sh)
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n:int)->torch.Tensor:
        """
        Returns random timesteps between and n.

        Args:
            n (int): number of timesteps.

        Returns:
            torch.Tensor: tensor with random timesteps.
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, 
               model: torch.nn.Module, 
               n: int=1,
               n_channels: int=1,
               x: torch.Tensor=None,
               classification: torch.Tensor=None,
               classification_scale: float=3.0,
               start_from: int=0)->torch.Tensor:
        """
        Samples an image from a given diffusion model.

        Args:
            model (torch.nn.Module): diffusion model predicting noise in image.
            n (int, optional): number of samples (batch size in image or class 
                conditioning overrides this). Defaults to 1.
            n_channels (int, optional): number of channels (channel number in
                image conditioning overrides this). Defaults to 1.
            x (torch.Tensor, optional): input image for conditioning. Defaults 
                to None.
            classification (torch.Tensor, optional): classification 
                conditioning. Should be an int tensor with shape [batch_size]. 
                Defaults to None.
            classification_scale (float, optional): classification scale for
                classification guidance (unconditioned and conditioned outputs
                are linearly interpolated using this value as a weight for the
                conditioned ouptuts). Defaults to 3.0.
            start_from (int, optional): starts the time sampling from this value.
                This allows using partially diffused images as input for input
                conditioning (helpful in artefact detection). Defaults to 0.

        Returns:
            torch.Tensor: image sampled from diffusion process.
        """

        if classification is not None:
            n = classification.shape[0]
        if x is not None:
            n = x.shape[:2]
        else:
            # fetch a model parameter to retrieve parameter
            device = next(model.parameters()).device
            x = torch.randn((n, 
                             n_channels, 
                             *self.img_size)).to(device)
        model.eval()
        with torch.no_grad():
            self.get_shape(x)
            t_range = reversed(range(start_from + 1, 
                                     self.noise_steps + start_from))
            for i in tqdm(t_range, position=0):
                t = (torch.ones(n) * i).long().to(x.device)
                predicted_noise = model(x, t, classification)
                alpha = self.alpha[t].reshape(-1,1,1,1)
                alpha_hat = self.alpha_hat[t].reshape(-1,1,1,1)
                beta = self.beta[t].reshape(-1,1,1,1)
                if classification_scale > 0:
                    nonconditional_predicted_noise = model(x, t)
                    predicted_noise = torch.lerp(
                        input=nonconditional_predicted_noise,
                        end=predicted_noise,
                        weight=classification_scale)
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                lhs = torch.multiply(
                    1 / torch.sqrt(alpha),
                    x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
                rhs = torch.sqrt(beta) * noise
                x = torch.add(lhs,rhs)
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
