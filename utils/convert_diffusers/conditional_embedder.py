import torch
import torch.nn as nn
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config


class ConditionalEmbedder(ModelMixin, ConfigMixin):
    """
    Patchifies VAE-encoded conditions (source video or reference image)
    into the DiT hidden dimension space via a Conv3d layer.
    """

    @register_to_config
    def __init__(
        self,
        in_dim: int = 48,
        dim: int = 3072,
        patch_size: list = [1, 2, 2],
        zero_init: bool = True,
        ref_pad_first: bool = False,
    ):
        super().__init__()
        kernel_size = tuple(patch_size)
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=kernel_size, stride=kernel_size
        )
        self.ref_pad_first = ref_pad_first
        if zero_init:
            nn.init.zeros_(self.patch_embedding.weight)
            nn.init.zeros_(self.patch_embedding.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embedding(x)
