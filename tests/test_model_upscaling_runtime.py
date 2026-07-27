# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Execution tests for Comfy-managed model upscaling."""

from __future__ import annotations

from types import ModuleType
from typing import cast

import pytest
import torch
from torch import nn

import whiterabbit.runtime.model_upscaling as upscaling_module
from whiterabbit.runtime.model_upscaling import ModelUpscaler


class _FakeUpscaleModel:
    """Identity model descriptor with the same surface as Comfy's wrapper."""

    def __init__(self) -> None:
        self.model: nn.Module = nn.Identity()
        self.scale = 1.0
        self.devices: list[str] = []

    def to(self, device: torch.device | str) -> _FakeUpscaleModel:
        self.devices.append(str(device))
        self.model.to(device)
        return self

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.model(images))


class _FakeManagement(ModuleType):
    """Record memory coordination while providing a CPU Comfy boundary."""

    def __init__(self) -> None:
        super().__init__("comfy.model_management")
        self.memory_requests: list[int] = []

    @staticmethod
    def get_torch_device() -> torch.device:
        return torch.device("cpu")

    @staticmethod
    def module_size(model: nn.Module) -> int:
        del model
        return 0

    def free_memory(self, amount: int, device: torch.device) -> None:
        assert device.type == "cpu"
        self.memory_requests.append(amount)

    @staticmethod
    def raise_non_oom(error: Exception) -> None:
        raise error


class _FakeProgress:
    """Accept the same construction as Comfy's progress bar."""

    def __init__(self, total: int) -> None:
        assert total > 0


class _FakeUtils(ModuleType):
    """Execute tiles directly while counting chunk-level calls."""

    def __init__(self) -> None:
        super().__init__("comfy.utils")
        self.calls = 0
        self.tile_sizes: list[int] = []
        self.ProgressBar = _FakeProgress

    @staticmethod
    def get_tiled_scale_steps(
        width: int,
        height: int,
        *,
        tile_x: int,
        tile_y: int,
        overlap: int,
    ) -> int:
        del width, height, tile_x, tile_y, overlap
        return 1

    def tiled_scale(
        self,
        source: torch.Tensor,
        model: _FakeUpscaleModel,
        *,
        tile_x: int,
        tile_y: int,
        overlap: int,
        upscale_amount: float,
        pbar: _FakeProgress,
    ) -> torch.Tensor:
        del tile_y, overlap, upscale_amount, pbar
        self.calls += 1
        self.tile_sizes.append(tile_x)
        return model(source)


def test_model_upscaler_chunks_batches_and_returns_model_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime coordinates memory, chunks work, and always offloads the model."""

    management = _FakeManagement()
    comfy_utils = _FakeUtils()

    def resolve_module(name: str) -> ModuleType:
        return management if name == "comfy.model_management" else comfy_utils

    monkeypatch.setattr(upscaling_module, "import_module", resolve_module)
    model = _FakeUpscaleModel()
    images = torch.rand((3, 8, 8, 3), generator=torch.Generator().manual_seed(12))

    output = ModelUpscaler().upscale(
        model,
        images,
        maximum_batch_size=2,
        tile_size=256,
        channels_last=False,
        precision="fp32",
    )

    torch.testing.assert_close(output, images)
    assert comfy_utils.calls == 2
    assert management.memory_requests
    assert model.devices[-1] == "cpu"


def test_model_upscaler_normalizes_too_small_manual_tiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manual tiles below Comfy's safe minimum still execute at the minimum size."""

    management = _FakeManagement()
    comfy_utils = _FakeUtils()

    def resolve_module(name: str) -> ModuleType:
        return management if name == "comfy.model_management" else comfy_utils

    monkeypatch.setattr(upscaling_module, "import_module", resolve_module)
    image = torch.rand((1, 8, 8, 3), generator=torch.Generator().manual_seed(13))

    output = ModelUpscaler().upscale(
        _FakeUpscaleModel(),
        image,
        maximum_batch_size=1,
        tile_size=64,
        channels_last=False,
        precision="fp32",
    )

    torch.testing.assert_close(output, image)
    assert comfy_utils.tile_sizes == [128]
