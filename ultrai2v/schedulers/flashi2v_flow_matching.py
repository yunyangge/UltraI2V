
import torch

from .flow_matching import FlowMatchingScheduler

class FlashI2VFlowMatchingScheduler(FlowMatchingScheduler):

    def interpolation(
        self,
        latents,
        prior_dist,
        sigmas,
        start_frame_latents
    ):
        latents_dtype = latents.dtype
        latents = latents.float()
        prior_dist = prior_dist.float()
        sigmas = sigmas.float()
        start_frame_latents = start_frame_latents.float()

        interpolated_latents = sigmas * prior_dist + (1.0 - sigmas) * latents - start_frame_latents

        interpolated_latents = interpolated_latents.to(latents_dtype)

        return interpolated_latents


    def get_latent_model_input(self, latents, **kwargs):
        start_frame_latents = kwargs.get("start_frame_latents", None)
        fourier_features = kwargs.get("fourier_features", None)
        if start_frame_latents is None or fourier_features is None:
            raise ValueError("In flashi2v flow, start_frame_latents and fourier_features must be specified!")
        latent_model_input_cond = latent_model_input_uncond = latents - start_frame_latents
        latent_model_input_cond = torch.cat([latent_model_input_cond, fourier_features], dim=1)
        latent_model_input_uncond = torch.cat([latent_model_input_uncond, fourier_features], dim=1)

        return latent_model_input_cond, latent_model_input_uncond

    def q_sample(
        self, 
        latents,
        start_frame_latents,
        sigmas=None,
        prior_dist=None,
        **kwargs
    ):
        b, c, t, h, w = latents.shape
        if prior_dist is None:
            prior_dist = torch.randn_like(latents, dtype=latents.dtype, device=latents.device)
        image_seq_len = h * w // 4 if self.use_dynamic_shifting else None
        sigmas = self._set_sigmas(sigmas=sigmas, batch_size=b, image_seq_len=image_seq_len, device=latents.device)
        timesteps = sigmas.clone() * 1000
        while sigmas.ndim < latents.ndim:
            sigmas = sigmas.unsqueeze(-1)
        # print(f"rank = {torch.distributed.get_rank()}, sigmas = {sigmas.squeeze().cpu().numpy()}")
        interpolated_latents = self.interpolation(latents, prior_dist, sigmas, start_frame_latents)

        return dict(x_t=interpolated_latents, prior_dist=prior_dist, sigmas=sigmas, timesteps=timesteps)


flow_scheduler = {
    "flashi2v_flow_matching": FlashI2VFlowMatchingScheduler,
}