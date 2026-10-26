from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import MaskConditionedDenoiser


def cosine_beta_schedule(steps: int, offset: float = 0.008) -> torch.Tensor:
    points = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    cumulative = torch.cos(
        ((points / steps) + offset) / (1 + offset) * math.pi / 2
    ).pow(2)
    cumulative = cumulative / cumulative[0]
    betas = 1 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(1e-5, 0.999).float()


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    supervised: torch.Tensor,
) -> torch.Tensor:
    """Positive denoising objective evaluated only on artificial targets."""
    if supervised.any():
        return F.mse_loss(prediction[supervised], target[supervised])
    return prediction.sum() * 0


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    supervised: torch.Tensor,
) -> torch.Tensor:
    """Positive categorical NLL evaluated only on revealed training targets."""
    if supervised.any():
        return F.cross_entropy(logits[supervised], target[supervised])
    return logits.sum() * 0


class MixedDiffusion(nn.Module):
    """Conditional Gaussian and absorbing diffusion for mixed tables."""

    def __init__(
        self,
        denoiser: MaskConditionedDenoiser,
        steps: int = 200,
        categorical_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if steps < 2:
            raise ValueError("Diffusion requires at least two steps.")
        if categorical_loss_weight < 0:
            raise ValueError("categorical_loss_weight cannot be negative.")
        self.denoiser = denoiser
        self.steps = steps
        self.categorical_loss_weight = categorical_loss_weight

        betas = cosine_beta_schedule(steps)
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", torch.cat([torch.zeros(1), betas]))
        self.register_buffer("alphas", torch.cat([torch.ones(1), alphas]))
        self.register_buffer("alpha_bars", torch.cat([torch.ones(1), cumulative]))

    def config(self) -> dict[str, float | int]:
        return {
            "steps": self.steps,
            "categorical_loss_weight": self.categorical_loss_weight,
        }

    def training_loss(
        self,
        numerical: torch.Tensor,
        categorical: torch.Tensor,
        numerical_available: torch.Tensor,
        categorical_available: torch.Tensor,
        numerical_targets: torch.Tensor,
        categorical_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if (numerical_targets & ~numerical_available).any():
            raise ValueError("Numerical targets must be observed training cells.")
        if (categorical_targets & ~categorical_available).any():
            raise ValueError("Categorical targets must be observed training cells.")
        batch = numerical.shape[0]
        device = numerical.device
        time_index = torch.randint(1, self.steps + 1, (batch,), device=device)
        alpha_bar = self.alpha_bars[time_index].unsqueeze(1)

        condition_mask = torch.cat(
            [
                numerical_targets | ~numerical_available,
                categorical_targets | ~categorical_available,
            ],
            dim=1,
        )

        noise = torch.randn_like(numerical)
        noised_numerical = alpha_bar.sqrt() * numerical + (1 - alpha_bar).sqrt() * noise
        numerical_input = torch.where(numerical_targets, noised_numerical, numerical)
        unavailable_numerical = ~numerical_available
        numerical_input = torch.where(
            unavailable_numerical, torch.randn_like(numerical), numerical_input
        )

        categorical_input = categorical.clone()
        for index, size in enumerate(self.denoiser.categorical_sizes):
            move = torch.rand(batch, device=device) < (1 - alpha_bar[:, 0])
            move &= categorical_targets[:, index]
            categorical_input[move, index] = size
            categorical_input[~categorical_available[:, index], index] = size

        predicted_noise, categorical_logits = self.denoiser(
            numerical_input,
            categorical_input,
            time_index.float() / self.steps,
            condition_mask,
        )

        numerical_loss = masked_mse(predicted_noise, noise, numerical_targets)

        categorical_losses: list[torch.Tensor] = []
        for index, logits in enumerate(categorical_logits):
            supervised = categorical_targets[:, index] & (
                categorical_input[:, index] == self.denoiser.categorical_sizes[index]
            )
            if supervised.any():
                categorical_losses.append(
                    masked_cross_entropy(
                        logits,
                        categorical[:, index],
                        supervised,
                    )
                )
        if categorical_losses:
            categorical_loss = torch.stack(categorical_losses).mean()
        else:
            categorical_loss = sum(
                (logits.sum() * 0 for logits in categorical_logits),
                numerical_loss.new_zeros(()),
            )

        total = numerical_loss + self.categorical_loss_weight * categorical_loss
        return {
            "loss": total,
            "numerical_loss": numerical_loss,
            "categorical_loss": categorical_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        numerical: torch.Tensor,
        categorical: torch.Tensor,
        numerical_target_mask: torch.Tensor,
        categorical_target_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        device = numerical.device
        batch = numerical.shape[0]
        condition_mask = torch.cat(
            [numerical_target_mask, categorical_target_mask], dim=1
        )
        context_numerical = numerical.clone()
        context_categorical = categorical.clone()

        current_numerical = torch.where(
            numerical_target_mask, torch.randn_like(numerical), context_numerical
        )
        current_categorical = context_categorical.clone()
        for index, size in enumerate(self.denoiser.categorical_sizes):
            current_categorical[categorical_target_mask[:, index], index] = size

        for step in range(self.steps, 0, -1):
            time = torch.full(
                (batch,), step / self.steps, device=device, dtype=torch.float32
            )
            predicted_noise, logits = self.denoiser(
                current_numerical, current_categorical, time, condition_mask
            )

            beta = self.betas[step]
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            alpha_bar_previous = self.alpha_bars[step - 1]
            predicted_x0 = (
                current_numerical - (1 - alpha_bar).sqrt() * predicted_noise
            ) / alpha_bar.sqrt().clamp_min(1e-8)
            coefficient_x0 = beta * alpha_bar_previous.sqrt() / (1 - alpha_bar)
            coefficient_xt = (1 - alpha_bar_previous) * alpha.sqrt() / (1 - alpha_bar)
            mean = coefficient_x0 * predicted_x0 + coefficient_xt * current_numerical
            if step > 1:
                variance = beta * (1 - alpha_bar_previous) / (1 - alpha_bar)
                next_numerical = mean + variance.sqrt() * torch.randn_like(mean)
            else:
                next_numerical = mean
            current_numerical = torch.where(
                numerical_target_mask, next_numerical, context_numerical
            )

            unmask_probability = (alpha_bar_previous * beta / (1 - alpha_bar)).clamp(
                0, 1
            )
            for index, column_logits in enumerate(logits):
                size = self.denoiser.categorical_sizes[index]
                is_masked = current_categorical[:, index] == size
                candidates = is_masked & categorical_target_mask[:, index]
                if not candidates.any():
                    continue
                if step == 1:
                    reveal = candidates
                else:
                    reveal = candidates & (
                        torch.rand(batch, device=device) < unmask_probability
                    )
                if reveal.any():
                    probabilities = F.softmax(
                        column_logits[reveal] / temperature, dim=-1
                    )
                    current_categorical[reveal, index] = torch.multinomial(
                        probabilities, 1
                    ).squeeze(1)
            current_categorical = torch.where(
                categorical_target_mask,
                current_categorical,
                context_categorical,
            )

        return current_numerical, current_categorical
