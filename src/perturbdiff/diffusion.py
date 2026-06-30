from __future__ import annotations

import torch
from torch import nn

from .config import DiffusionConfig


def _extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    """DDPM forward process and deterministic DDIM sampler."""

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        if config.schedule != "linear":
            raise ValueError(f"Unsupported diffusion schedule: {config.schedule}")
        self.prediction_type = getattr(config, "prediction_type", "eps")
        if self.prediction_type not in {"eps", "x0"}:
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")
        betas = torch.cat(
            [
                torch.zeros(1),
                torch.linspace(config.beta_start, config.beta_end, config.timesteps),
            ]
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.config = config
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

    @property
    def timesteps(self) -> int:
        return self.config.timesteps

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(1, self.timesteps + 1, (batch_size,), device=device)

    def q_sample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bar = _extract(self.sqrt_alphas_cumprod, timesteps, x0.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x0.shape)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise, noise

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha_bar = _extract(self.sqrt_alphas_cumprod, timesteps, x_t.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape)
        return (x_t - sqrt_one_minus * eps) / sqrt_alpha_bar.clamp_min(1e-12)

    def predict_eps_from_x0(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha_bar = _extract(self.sqrt_alphas_cumprod, timesteps, x_t.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape)
        return (x_t - sqrt_alpha_bar * x0) / sqrt_one_minus.clamp_min(1e-12)

    def prediction_to_x0_and_eps(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.prediction_type == "eps":
            eps = prediction
            x0 = self.predict_x0_from_eps(x_t, timesteps, eps)
            return x0, eps
        x0 = prediction
        eps = self.predict_eps_from_x0(x_t, timesteps, x0)
        return x0, eps

    @torch.no_grad()
    def ddim_sample(
        self,
        denoiser: nn.Module,
        shape: tuple[int, ...],
        z_cond: torch.Tensor,
        *,
        steps: int = 100,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if steps < 1:
            raise ValueError("steps must be at least 1")
        device = z_cond.device
        batch_size = z_cond.shape[0]
        x = noise if noise is not None else torch.randn(shape, device=device)
        if x.shape[0] != batch_size:
            raise ValueError("noise/sample batch size must match z_cond")

        time_seq = torch.linspace(
            self.timesteps,
            0,
            steps + 1,
            device=device,
        ).round().long()
        time_seq = torch.unique_consecutive(time_seq)
        if time_seq[-1].item() != 0:
            time_seq = torch.cat([time_seq, torch.zeros(1, device=device, dtype=torch.long)])

        for idx in range(len(time_seq) - 1):
            t_value = time_seq[idx]
            prev_t_value = time_seq[idx + 1]
            t = torch.full((batch_size,), int(t_value.item()), device=device, dtype=torch.long)
            prev_t = torch.full(
                (batch_size,),
                int(prev_t_value.item()),
                device=device,
                dtype=torch.long,
            )
            prediction = denoiser(x, t, z_cond)
            x0, eps = self.prediction_to_x0_and_eps(x, t, prediction)
            prev_alpha_bar = _extract(self.alphas_cumprod, prev_t, x.shape)
            x = torch.sqrt(prev_alpha_bar) * x0 + torch.sqrt(1.0 - prev_alpha_bar) * eps
        return x
