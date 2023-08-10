"""
Diffusion classes to train diffusion models. 

Based on:
    https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/ddpm.py
    https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/ddpm_conditional.py
    https://arxiv.org/abs/2305.03486
    https://arxiv.org/pdf/2301.10972.pdf
    https://huggingface.co/blog/annotated-diffusion
"""

import numpy as np
import torch
from tqdm.auto import tqdm

from typing import Union,Tuple,List

def cosine_beta_schedule(timesteps: int, 
                         beta_start: float=0.0001,
                         beta_end: float=0.02,
                         s: float=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps: int, 
                         beta_start: float=0.0001, 
                         beta_end: float=0.02,
                         s: float=None):
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps: int, 
                            beta_start: float=0.0001, 
                            beta_end: float=0.02,
                            s: float=None):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps: int, 
                          beta_start: float=0.0001, 
                          beta_end: float=0.02,
                          s: float=None):
    betas = torch.linspace(-3, 3, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

class Diffusion:
    """
    Diffusion process for diffusion models.

    For a given timestep t, we can diffuse an image X using ``noise_images``:

        $\sqrt{\hat{\alpha}} * X + \sqrt{1 - \hat{alpha}} * \epsilon$,

    where $\hat{\alpha}$ is the cummulative product of alpha at timestep $t$, 
    and $\epsilon$ is a noise image. In other words, the diffused image at 
    timestep t is the weighted sum of that same image and a noise vector of the
    same size.

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
                 scheduler: str="cosine",
                 step_key: str="ddpm",
                 device: str="cuda",
                 track_progress: bool=False):
        """
        Args:
            noise_steps (int, optional): number of steps in diffusion process. 
                Defaults to 1000.
            beta_start (float, optional): initial amount of noise (only for 
                linear schedule). Defaults to 1e-4.
            beta_end (float, optional): final amount of noise (only for 
                linear schedule). Defaults to 0.02.
            img_size (Union[Tuple[int,int],Tuple[int,int,int]], optional): size
                of image. Defaults to (256,256).
            scheduler (float, optional): alpha scheduler. Can be "linear", 
                "quadratic", "sigmoid" or "cosine". Defaults to "cosine".
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.scheduler = scheduler
        self.step_key = step_key
        self.track_progress = track_progress

        self.n_dim = len(img_size)

        self.beta = self.get_noise_schedule()
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha,0)
        self.check_arguments()

    def get_noise_schedule(self):
        return self.schedulers[self.scheduler](self.noise_steps,
                                               self.beta_start,
                                               self.beta_end)

    @property
    def schedulers(self):
        return {
            "linear":linear_beta_schedule,
            "quadratic":quadratic_beta_schedule,
            "sigmoid":sigmoid_beta_schedule,
            "cosine":cosine_beta_schedule}

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

    def noise_images(self, 
                     x: torch.Tensor, 
                     epsilon: torch.Tensor, 
                     t: int)->torch.Tensor:
        """
        Introduces noise to images x at timestep t.

        Args:
            x (torch.Tensor): input images.
            t (int): timestep.

        Returns:
            torch.Tensor: noised (diffused) images.
        """
        sh = self.get_shape(x)
        t = t.to(x.device)
        self.alpha_bar = self.alpha_bar.to(x.device)
        alpha_bar = self.alpha_bar[t]
        sqrt_alpha = torch.sqrt(alpha_bar).reshape(sh)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar).reshape(sh)
        return sqrt_alpha * x + sqrt_one_minus_alpha * epsilon, epsilon

    def __call__(self, 
                 x: torch.Tensor, 
                 model: torch.nn.Module, 
                 epsilon: torch.Tensor, 
                 t: torch.Tensor,
                 classification: torch.Tensor=None):
        noised_image, epsilon = self.noise_images(x,epsilon=epsilon,t=t)
        prediction = model(noised_image,t/self.noise_steps,classification)
        return prediction

    def sample_timesteps(self, n:int)->torch.Tensor:
        """
        Returns random timesteps between and n.

        Args:
            n (int): number of timesteps.

        Returns:
            torch.Tensor: tensor with random timesteps.
        """
        return torch.randint(1,self.noise_steps,(n,))

    def ddpm_reverse_step(self,
                          x: torch.Tensor, 
                          epsilon: torch.Tensor,
                          t: int,
                          eta: float=1.0):
        sh = self.get_shape(x)
        alpha_bar = self.alpha_bar[t].reshape(sh)
        if t > 0:
            alpha_bar_prev = self.alpha_bar[t-1]
        else:
            alpha_bar_prev = torch.ones_like(alpha_bar)
        alpha_bar_prev = alpha_bar_prev.reshape(sh)
        beta_bar = 1. - alpha_bar
        x_prev = (x - torch.sqrt(beta_bar) * epsilon) / torch.sqrt(alpha_bar)
        x_prev = torch.clamp(x_prev,-1,1)
        coef_t_prev = (torch.sqrt(alpha_bar_prev) * self.beta[t]) / beta_bar
        coef_t = torch.sqrt(self.alpha[t]) * (1. - alpha_bar_prev) / beta_bar
        x = coef_t_prev * x_prev + coef_t * x
        if t > 0 and eta > 0:
            var = (1 - alpha_bar_prev) / (1 - alpha_bar) * self.beta[t]
            x = x + torch.randn_like(x) * torch.sqrt(var) * eta
        return x

    def ddim_inverse_step(self,
                          x: torch.Tensor, 
                          epsilon: torch.Tensor, 
                          t: int,
                          eta: float=1.0):
        sh = self.get_shape(x)
        alpha_bar = self.alpha_bar[t]
        if t > 0:
            alpha_bar_prev = self.alpha_bar[t-1]
        else:
            alpha_bar_prev = torch.ones_like(alpha_bar)
    
        alpha_bar = alpha_bar.reshape(sh)
        alpha_bar_prev = alpha_bar_prev.reshape(sh)
        var = torch.multiply(
            torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)),
            torch.sqrt(1 - alpha_bar / alpha_bar_prev)) * eta
        x_predict_x0 = torch.divide(
            x - torch.sqrt(1 - alpha_bar) * epsilon,
            torch.sqrt(alpha_bar))
        direction_to_xt = torch.sqrt(1 - alpha_bar_prev - var) * epsilon
        random_noise = torch.randn_like(epsilon) * var
        output = torch.add(
            torch.sqrt(alpha_bar_prev) * x_predict_x0,
            direction_to_xt + random_noise)
        return output

    def alpha_deblending_step(self,
                              x: torch.Tensor, 
                              epsilon: torch.Tensor, 
                              t: int,
                              eta: float=None):
        return x + (self.alpha_bar[t] - self.alpha_bar[t-1]) * epsilon

    def step(self,
             x: torch.Tensor,
             epsilon: torch.Tensor,
             t: int,
             eta: float=1.0):
        if self.step_key == "ddpm":
            return self.ddpm_reverse_step(x,epsilon=epsilon,t=t,eta=eta)
        elif self.step_key == "ddim":
            return self.ddim_reverse_step(x,epsilon=epsilon,t=t,eta=eta)
        elif self.step_key == "alpha_deblending":
            return self.alpha_deblending_step(x,epsilon=epsilon,t=t,eta=None)

    def sample(self, 
               model: torch.nn.Module, 
               n: int=1,
               n_channels: int=1,
               x: torch.Tensor=None,
               classification: torch.Tensor=None,
               classification_scale: float=3.0,
               start_from: int=None)->torch.Tensor:
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
                conditioning (helpful in artefact detection). Defaults to None.

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
        self.alpha = self.alpha.to(x.device)
        self.beta = self.beta.to(x.device)
        self.alpha_bar = self.alpha_bar.to(x.device)
        model.eval()
        print("Generating...")
        with torch.no_grad():
            final_t = self.noise_steps if start_from is None else start_from
            t_range = reversed(range(1,final_t))
            if self.track_progress is True:
                t_range = tqdm(t_range,total=final_t,
                               desc="Generating...",position=0,
                               leave=True)
            for i in t_range:
                t = torch.as_tensor([i],device=x.device)
                predicted_noise = model(x, t / self.noise_steps, 
                                        classification)
                if classification_scale > 0:
                    nonconditional_predicted_noise = model(
                        x, t / self.noise_steps)
                    predicted_noise = torch.lerp(
                        input=nonconditional_predicted_noise,
                        end=predicted_noise,
                        weight=classification_scale)
                x = self.step(x=x,epsilon=predicted_noise,t=t)
        model.train()
        x = x.clamp(-1, 1)
        return x
