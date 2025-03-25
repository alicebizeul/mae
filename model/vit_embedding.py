# coding=utf-8
# Copyright 2022 Facebook AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch ViT MAE (masked autoencoder) model."""

import collections.abc
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from ...utils import (
    torch_int,
)


class ViTMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = ViTMAEPatchEmbeddings(config)
        self.eigenvalues = config.eigenvalues
        self.info_patches = self.patch_embeddings.info_patches
        self.num_patches = self.patch_embeddings.num_patches
        self.patch_size = self.patch_embeddings.patch_size
        self.position_embeddings = torch.nn.Conv1d(1, config.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(
        #     self.position_embeddings.shape[-1], int(self.patch_embeddings.num_patches**0.5), add_cls_token=True
        # )
        # self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

    # Copied from transformers.models.vit.modeling_vit.ViTEmbeddings.interpolate_pos_encoding
    # def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
    #     """
    #     This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
    #     images. This method is also adapted to support torch.jit tracing.

    #     Adapted from:
    #     - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
    #     - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
    #     """

    #     num_patches = embeddings.shape[1] - 1
    #     num_positions = self.position_embeddings.shape[1] - 1

    #     # always interpolate when tracing to ensure the exported model works for dynamic input shapes
    #     if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
    #         return self.position_embeddings

    #     class_pos_embed = self.position_embeddings[:, :1]
    #     patch_pos_embed = self.position_embeddings[:, 1:]

    #     dim = embeddings.shape[-1]

    #     new_height = height // self.patch_size
    #     new_width = width // self.patch_size

    #     sqrt_num_positions = torch_int(num_positions**0.5)
    #     patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
    #     patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

    #     patch_pos_embed = nn.functional.interpolate(
    #         patch_pos_embed,
    #         size=(new_height, new_width),
    #         mode="bicubic",
    #         align_corners=False,
    #     )

    #     patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    #     return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def random_masking(self, sequence, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def forward(self, pixel_values, noise=None, interpolate_pos_encoding: bool = False):
        # batch_size, num_channels, height, width = pixel_values.shape
        batch_size, num_pcs = pixel_values.shape

        # here we need to create the 1D vector with padding at the right spots, then we need to understand the target
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        if interpolate_pos_encoding:
            # position_embeddings = self.interpolate_pos_encoding(embeddings, height, width)
            print("Problem with positional embeddings")
        else:
            position_embeddings = self.position_embeddings

        # add position embeddings w/o cls token
        embeddings = embeddings + position_embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

        # append cls token
        cls_token = self.cls_token + position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore


class ViTMAEPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)

        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        info_patches = 1/num_patches 
        patch_size = np.argmin(np.abs(np.cumsum(config.eigenvalues) - (1-info_patches)))

        self.image_size = image_size
        self.patch_size = patch_size
        self.info_patches = info_patches
        self.num_channels = num_channels
        self.num_patches = num_patches
        print("This is the info per patch",info_patches)

        self.projection = nn.Conv1d(1, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values, interpolate_pos_encoding: bool = False):
        batch_size, nb_pcs = pixel_values.shape
        # batch_size, num_channels, height, width = pixel_values.shape
        if not interpolate_pos_encoding and (nb_pcs != self.num_channels*self.image_size[0]*self.image_size[1]):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        x = self.projection(pixel_values).transpose(1, 2) # not sure about transpose
        return x

