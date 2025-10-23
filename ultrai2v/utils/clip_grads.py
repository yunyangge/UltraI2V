import os
import torch
import logging
import warnings
import types
from typing import Optional
from torch.nn.utils import get_total_norm, clip_grad_norm_, clip_grads_with_norm_

class AdaptiveGradClipper:
    def __init__(self, clip_grad_ema_decay: float = 0.99, init_max_grad_norm: float = 1.0):
        self.clip_grad_ema_decay = clip_grad_ema_decay

        self.moving_avg_max_grad_norm = -float('inf')
        self.moving_avg_max_grad_norm_var = 0.0
        self.grad_norm_before_clip = 0.0
        self.grad_norm_after_clip = 0.0
        self.max_grad_norm = init_max_grad_norm
        self.max_grad_norm_var = 0.0

        self.save_postfix = 'adaptive_grad_clip_states.pt'

    def state_dict(self) -> dict:
        return {
            "moving_avg_max_grad_norm": self.moving_avg_max_grad_norm,
            "moving_avg_max_grad_norm_var": self.moving_avg_max_grad_norm_var,
            "grad_norm_before_clip": self.grad_norm_before_clip,
            "grad_norm_after_clip": self.grad_norm_after_clip,
            "max_grad_norm": self.max_grad_norm,
            "max_grad_norm_var": self.max_grad_norm_var,
        }
    
    def load_state_dict(self, state_dict: dict):
        self.moving_avg_max_grad_norm = state_dict["moving_avg_max_grad_norm"]
        self.moving_avg_max_grad_norm_var = state_dict["moving_avg_max_grad_norm_var"]
        self.grad_norm_before_clip = state_dict["grad_norm_before_clip"]
        self.grad_norm_after_clip = state_dict["grad_norm_after_clip"]
        self.max_grad_norm = state_dict["max_grad_norm"]
        self.max_grad_norm_var = state_dict["max_grad_norm_var"]

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logs_info = f"""
            {'=' * 20}Adaptive Grad Clipper States Loaded{'=' * 20}
            moving_avg_max_grad_norm: {self.moving_avg_max_grad_norm}
            moving_avg_max_grad_norm_var: {self.moving_avg_max_grad_norm_var}
            grad_norm_before_clip: {self.grad_norm_before_clip}
            grad_norm_after_clip: {self.grad_norm_after_clip}
            max_grad_norm: {self.max_grad_norm}
            max_grad_norm_var: {self.max_grad_norm_var}
            {'=' * 20}{'=' * len('Adaptive Grad Clipper States Loaded')}{'=' * 20}
            """
            logging.info(logs_info)

    def load(self, output_dir: str):
        load_path = os.path.join(output_dir, self.save_postfix)
        if not os.path.exists(load_path):
            warnings.warn(f"No adaptive grad clipper state file found at {load_path}. Use initial values.")
            return
        state_dict = torch.load(load_path, map_location='cpu')
        self.load_state_dict(state_dict)

    def save(self, output_dir: str):
        torch.save(self.state_dict(), os.path.join(output_dir, self.save_postfix))

    def get_max_grad_norm(self, moving_avg_max_grad_norm, moving_avg_max_grad_norm_var) -> float:
        return moving_avg_max_grad_norm + 3.0 * (moving_avg_max_grad_norm_var ** 0.5)

    def get_max_grad_norm_var(self, moving_avg_max_grad_norm, grad_norm_before_clip) -> float:
        return (moving_avg_max_grad_norm - grad_norm_before_clip) ** 2

    # in first iteration, initialize moving averages
    def init_weights(self, grad_norm_before_clip: float):
        self.moving_avg_max_grad_norm = min(3 * grad_norm_before_clip, self.max_grad_norm)
        self.moving_avg_max_grad_norm_var = 0.0
        self.max_grad_norm = self.get_max_grad_norm(self.moving_avg_max_grad_norm, self.moving_avg_max_grad_norm_var)
        self.max_grad_norm_var = 0.0
        self.grad_norm_before_clip = grad_norm_before_clip
        self.grad_norm_after_clip = grad_norm_before_clip

    def normal_clip(
        self, 
        parameters,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = True,
        foreach: Optional[bool] = None,
    ):
        return clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite, foreach)

    def adaptive_clip(
        self, 
        parameters,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = True,
        foreach: Optional[bool] = None,
    ):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        else:
            is_generator = isinstance(parameters, types.GeneratorType)
            # prevent generators from being exhausted
            parameters = list(parameters)
            if is_generator and len(parameters) == 0:
                warnings.warn(
                    "`parameters` is an empty generator, no gradient clipping will occur.",
                    stacklevel=3,
                )
        grads = [p.grad for p in parameters if p.grad is not None]
        grad_norm_before_clip = get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
        grad_norm_before_clip_float = grad_norm_before_clip.item()
        # first iteration, initialize moving averages
        if self.moving_avg_max_grad_norm < -1:
            self.init_weights(grad_norm_before_clip_float)
            return grad_norm_before_clip
        
        max_grad_norm = self.get_max_grad_norm(self.moving_avg_max_grad_norm, self.moving_avg_max_grad_norm_var)
        self.grad_norm_before_clip = grad_norm_before_clip_float
        self.max_grad_norm = max_grad_norm
        # update moving averages only if grad norm is less than max grad norm (meaning grads are normal)
        if grad_norm_before_clip_float <= max_grad_norm:
            self.moving_avg_max_grad_norm = self.clip_grad_ema_decay * self.moving_avg_max_grad_norm + (1 - self.clip_grad_ema_decay) * grad_norm_before_clip_float
            self.max_grad_norm_var = (self.moving_avg_max_grad_norm - grad_norm_before_clip_float) ** 2
            self.moving_avg_max_grad_norm_var = self.clip_grad_ema_decay * self.moving_avg_max_grad_norm_var + (1 - self.clip_grad_ema_decay) * self.max_grad_norm_var
            self.grad_norm_after_clip = self.grad_norm_before_clip
        else:
            clip_grads_with_norm_(parameters, self.max_grad_norm, grad_norm_before_clip, foreach)
            self.grad_norm_after_clip = get_total_norm(grads, norm_type, error_if_nonfinite, foreach).item()
        return torch.tensor(self.grad_norm_after_clip)