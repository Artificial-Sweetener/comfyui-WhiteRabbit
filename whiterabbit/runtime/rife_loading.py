# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Load and cache RIFE models through Comfy's model management."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

import torch
from torch import nn

from ..domain.rife import RifeModelSpec
from .rife_architecture import (
    LegacyRife47,
    RifeInferenceModel,
    create_core_rife,
    remap_core_state_dict,
    required_core_alignment,
    required_legacy_alignment,
)
from .rife_models import RifeModelResolver


@dataclass(frozen=True)
class LoadedRifeModel:
    """A model, its Comfy patcher, and inference device metadata."""

    inference_model: RifeInferenceModel
    patcher: Any
    device: torch.device
    dtype: torch.dtype
    spec: RifeModelSpec


class RifeModelLoader:
    """Load each cataloged checkpoint once and let Comfy manage residency."""

    def __init__(self, resolver: RifeModelResolver | None = None) -> None:
        """Create the loader with an injectable model resolver."""

        self._resolver = resolver or RifeModelResolver()
        self._cache: dict[str, LoadedRifeModel] = {}

    def load(
        self,
        filename: str,
        frame_shape: tuple[int, ...] | None = None,
        scale_factor: float = 1.0,
    ) -> LoadedRifeModel:
        """Resolve, construct, and load a supported RIFE model onto its run device."""

        cached = self._cache.get(filename)
        if cached is None:
            cached = self._construct(filename)
            self._cache[filename] = cached
        management: Any = import_module("comfy.model_management")
        activation_memory = 0
        if frame_shape is not None and len(frame_shape) >= 3:
            height, width = frame_shape[-3:-1]
            alignment = (
                required_core_alignment(scale_factor)
                if cached.spec.architecture == "core"
                else required_legacy_alignment(scale_factor)
            )
            padded_height = ((height - 1) // alignment + 1) * alignment
            padded_width = ((width - 1) // alignment + 1) * alignment
            internal_scale = max(1.0, scale_factor)
            activation_memory = int(
                300
                * padded_height
                * padded_width
                * internal_scale**2
                * cached.dtype.itemsize
            )
        management.load_models_gpu([cached.patcher], memory_required=activation_memory)
        return cached

    def _construct(self, filename: str) -> LoadedRifeModel:
        """Construct an inference module and CoreModelPatcher from one checkpoint."""

        path, spec = self._resolver.resolve(filename)
        comfy_utils: Any = import_module("comfy.utils")
        raw: Any = comfy_utils.load_torch_file(str(path), safe_load=True)
        state_dict = cast(dict[str, torch.Tensor], raw)
        if spec.architecture == "legacy47":
            module: nn.Module = LegacyRife47()
            module.load_state_dict(state_dict, strict=True)
            inference = cast(RifeInferenceModel, module)
        else:
            module = create_core_rife(remap_core_state_dict(state_dict))
            inference = cast(RifeInferenceModel, module)

        management: Any = import_module("comfy.model_management")
        device = cast(torch.device, management.get_torch_device())
        dtype = torch.float16 if management.should_use_fp16(device) else torch.float32
        module.eval().to(dtype=dtype)
        patcher_module: Any = import_module("comfy.model_patcher")
        patcher = patcher_module.CoreModelPatcher(
            module,
            load_device=device,
            offload_device=management.unet_offload_device(),
        )
        return LoadedRifeModel(inference, patcher, device, dtype, spec)


__all__ = ["LoadedRifeModel", "RifeModelLoader"]
