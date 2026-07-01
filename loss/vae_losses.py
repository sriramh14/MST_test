import torch
import torch.nn.functional as F


def reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return F.l1_loss(
        reconstruction,
        target,
        reduction="mean",
    )


def kl_divergence_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    kl = -0.5 * (
        1.0
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    return kl.mean()
