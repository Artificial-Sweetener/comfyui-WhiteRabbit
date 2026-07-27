# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""Vectorized scoring runtime for automatic video-loop end cropping."""

from __future__ import annotations

import math
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from importlib import import_module
from statistics import median
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as functional

from ..domain.loop_autocrop import (
    DIAGNOSTICS_HEADER,
    CandidateMetrics,
    LoopAutocropOptions,
)
from ..shared.tensor_validation import validate_image_batch


class ProgressBar(Protocol):
    """Subset of Comfy's progress bar used for crop candidates."""

    def update(self, value: int) -> None:
        """Advance progress by a number of candidates."""


@dataclass(frozen=True)
class PrecomputedMetrics:
    """Adjacent and seam metrics shared by every crop candidate."""

    adjacent_distance: torch.Tensor
    adjacent_similarity: torch.Tensor
    adjacent_flow: list[float]
    luma_means: torch.Tensor
    seam_distance: list[torch.Tensor]
    seam_similarity: list[torch.Tensor]
    seam_exposure: list[torch.Tensor]
    seam_flow: list[torch.Tensor]
    frames_nchw: torch.Tensor


class LoopAutocropRuntime:
    """Choose a natural loop by scoring every allowed crop from the clip end."""

    _gaussian_windows: dict[tuple[int, int, float, str, str], torch.Tensor] = {}

    def find(
        self,
        clip_frames: torch.Tensor,
        options: LoopAutocropOptions,
    ) -> tuple[torch.Tensor, int, int, float, str]:
        """Return the best cropped clip and complete CSV scoring diagnostics."""

        shape = validate_image_batch(clip_frames, name="clip_frames")
        if shape.batch_size < 2:
            return clip_frames, 0, shape.batch_size, 0.0, DIAGNOSTICS_HEADER
        evaluation = (
            (clip_frames * 255).round().clamp(0, 255) / 255
            if options.score_in_8bit
            else clip_frames
        )
        device = torch.device(
            "cuda"
            if options.accelerate_with_gpu and torch.cuda.is_available()
            else "cpu"
        )
        scales = parse_scales(options.ssim_downsample_scales)
        with torch.no_grad(), self._precision_context(device, options):
            precomputed = self._precompute(evaluation.to(device), options, scales)

        candidate_count = max(0, options.maximum_end_crop) + 1
        progress = self._progress_bar(candidate_count)
        rows: list[CandidateMetrics] = []
        best_crop = 0
        best_score = float("inf")
        for end_crop in range(candidate_count):
            kept_frames = shape.batch_size - end_crop
            if kept_frames < 2:
                progress.update(1)
                continue
            metrics = self._score_candidate(
                end_crop,
                kept_frames,
                precomputed,
                options,
            )
            rows.append(metrics)
            if metrics.score < best_score:
                best_score = metrics.score
                best_crop = end_crop
            progress.update(1)

        final_length = max(2, shape.batch_size - best_crop)
        diagnostics = DIAGNOSTICS_HEADER
        if rows:
            diagnostics += "\n" + "\n".join(row.to_csv() for row in rows)
        return (
            clip_frames[:final_length],
            best_crop,
            final_length,
            best_score,
            diagnostics,
        )

    def _precompute(
        self,
        frames: torch.Tensor,
        options: LoopAutocropOptions,
        scales: list[int],
    ) -> PrecomputedMetrics:
        """Compute adjacent and first-frame-aligned seam metric tables."""

        nchw = frames.permute(0, 3, 1, 2).contiguous()
        adjacent_left = nchw[:-1]
        adjacent_right = nchw[1:]
        adjacent_distance = self._distance(
            adjacent_left, adjacent_right, options.distance_metric
        )
        adjacent_similarity = (
            self._multiscale_ssim(adjacent_left, adjacent_right, scales)
            if options.use_ssim_similarity
            else torch.empty(0, device=frames.device)
        )
        luma_means = self._luma(nchw).mean(dim=(1, 2, 3))
        adjacent_flow = (
            [
                self._flow_magnitude(
                    frames[index : index + 1], frames[index + 1 : index + 2]
                )
                for index in range(frames.shape[0] - 1)
            ]
            if options.use_flow_guard
            else [0.0] * (frames.shape[0] - 1)
        )
        window = max(1, min(options.seam_window_frames, frames.shape[0] - 1))
        seam_distance: list[torch.Tensor] = []
        seam_similarity: list[torch.Tensor] = []
        seam_exposure: list[torch.Tensor] = []
        seam_flow: list[torch.Tensor] = []
        for index in range(window):
            first = nchw[index : index + 1].expand_as(nchw)
            seam_distance.append(self._distance(nchw, first, options.distance_metric))
            seam_similarity.append(
                self._multiscale_ssim(nchw, first, scales)
                if options.use_ssim_similarity
                else torch.empty(0, device=frames.device)
            )
            seam_exposure.append((luma_means - luma_means[index]).abs())
            seam_flow.append(
                torch.tensor(
                    [
                        self._flow_magnitude(
                            frames[frame_index : frame_index + 1],
                            frames[index : index + 1],
                        )
                        for frame_index in range(frames.shape[0])
                    ],
                    device=frames.device,
                )
                if options.use_flow_guard
                else torch.empty(0, device=frames.device)
            )
        return PrecomputedMetrics(
            adjacent_distance,
            adjacent_similarity,
            adjacent_flow,
            luma_means,
            seam_distance,
            seam_similarity,
            seam_exposure,
            seam_flow,
            nchw,
        )

    def _score_candidate(
        self,
        end_crop: int,
        kept_frames: int,
        metrics: PrecomputedMetrics,
        options: LoopAutocropOptions,
    ) -> CandidateMetrics:
        """Score a candidate crop against real adjacent motion targets."""

        last_index = kept_frames - 1
        window = max(
            1,
            min(
                options.seam_window_frames,
                last_index + 1,
                metrics.frames_nchw.shape[0] - 1,
            ),
        )
        target_distance = self._selected_target(
            metrics.adjacent_distance,
            kept_frames,
            options,
        )
        target_similarity = (
            self._selected_target(
                metrics.adjacent_similarity,
                kept_frames,
                options,
            )
            if options.use_ssim_similarity
            else 0.0
        )
        target_exposure = self._exposure_target(
            metrics.luma_means, kept_frames, options
        )
        target_flow = (
            float(median(metrics.adjacent_flow[: kept_frames - 1]))
            if options.use_flow_guard and kept_frames >= 3 and metrics.adjacent_flow
            else 0.0
        )
        indexes = torch.tensor(
            [last_index - (window - 1 - row) for row in range(window)],
            device=metrics.frames_nchw.device,
            dtype=torch.long,
        )
        seam_distance = self._seam_average(metrics.seam_distance, indexes, window)
        seam_similarity = (
            self._seam_average(metrics.seam_similarity, indexes, window)
            if options.use_ssim_similarity
            else 0.0
        )
        seam_exposure = (
            self._seam_average(metrics.seam_exposure, indexes, window)
            if options.use_exposure_guard
            else 0.0
        )
        seam_flow = (
            self._seam_average(metrics.seam_flow, indexes, window)
            if options.use_flow_guard
            else 0.0
        )
        epsilon = 1e-12
        step_cost = abs(seam_distance - target_distance) / (target_distance + epsilon)
        similarity_cost = (
            abs(seam_similarity - target_similarity)
            / (abs(target_similarity) + epsilon)
            if options.use_ssim_similarity
            else 0.0
        )
        exposure_cost = (
            abs(seam_exposure - target_exposure) / (target_exposure + epsilon)
            if options.use_exposure_guard and target_exposure > 0
            else 0.0
        )
        flow_cost = (
            abs(seam_flow - target_flow) / (target_flow + epsilon)
            if options.use_flow_guard and target_flow > 0
            else 0.0
        )
        score = (
            options.weight_step_size * step_cost
            + options.weight_similarity * similarity_cost
            + options.weight_exposure * exposure_cost
            + options.weight_flow * flow_cost
        )
        return CandidateMetrics(
            end_crop,
            score,
            seam_distance,
            target_distance,
            seam_similarity,
            target_similarity,
            seam_exposure,
            target_exposure,
            seam_flow,
            target_flow,
        )

    @staticmethod
    def _selected_target(
        adjacent: torch.Tensor,
        kept_frames: int,
        options: LoopAutocropOptions,
    ) -> float:
        """Select and median-combine enabled adjacent metric targets."""

        if adjacent.numel() == 0:
            return 0.0
        last_index = kept_frames - 1
        selected: list[torch.Tensor] = []
        if options.include_first_step:
            selected.append(adjacent[0])
        if options.include_last_step:
            selected.append(adjacent[last_index - 1])
        if options.include_global_median_step and kept_frames >= 3:
            selected.append(adjacent[: kept_frames - 1].median())
        if not selected:
            selected.append(adjacent[0])
        result = selected[0] if len(selected) == 1 else torch.stack(selected).median()
        return float(result.item())

    @staticmethod
    def _exposure_target(
        luma_means: torch.Tensor,
        kept_frames: int,
        options: LoopAutocropOptions,
    ) -> float:
        """Return the characterized first/last/global exposure target."""

        if not options.use_exposure_guard:
            return 0.0
        last_index = kept_frames - 1
        selected = [
            (luma_means[0] - luma_means[1]).abs(),
            (luma_means[last_index] - luma_means[last_index - 1]).abs(),
        ]
        if options.include_global_median_step and kept_frames >= 3:
            selected.append(
                (luma_means[: kept_frames - 1] - luma_means[1:kept_frames])
                .abs()
                .median()
            )
        return float(torch.stack(selected).median().item())

    @staticmethod
    def _seam_average(
        tables: list[torch.Tensor],
        indexes: torch.Tensor,
        window: int,
    ) -> float:
        """Average aligned seam metrics across a configured frame window."""

        values = torch.stack(
            [
                tables[row].index_select(0, indexes[row : row + 1]).squeeze(0)
                for row in range(window)
            ]
        )
        return float(values.mean().item())

    @staticmethod
    def _distance(left: torch.Tensor, right: torch.Tensor, kind: str) -> torch.Tensor:
        """Return per-pair L1 or MSE image distance."""

        difference = left - right
        if kind == "MSE":
            return difference.square().mean(dim=(1, 2, 3))
        return difference.abs().mean(dim=(1, 2, 3))

    @staticmethod
    def _luma(images: torch.Tensor) -> torch.Tensor:
        """Return Rec. 709 luma for NCHW gray, RGB, or RGBA batches."""

        if images.shape[1] == 1:
            return images[:, :1]
        return (
            0.2126 * images[:, 0:1] + 0.7152 * images[:, 1:2] + 0.0722 * images[:, 2:3]
        )

    def _multiscale_ssim(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        scales: list[int],
    ) -> torch.Tensor:
        """Average SSIM across configured area-downsample scales."""

        values = [
            self._ssim(self._downsample(left, scale), self._downsample(right, scale))
            for scale in scales
        ]
        return torch.stack(values).mean(dim=0)

    def _ssim(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return Gaussian-window structural similarity for batched pairs."""

        channels = left.shape[1]
        window = self._gaussian_window(channels, left.device, left.dtype)
        left_mean = functional.conv2d(left, window, padding=3, groups=channels)
        right_mean = functional.conv2d(right, window, padding=3, groups=channels)
        left_variance = (
            functional.conv2d(left.square(), window, padding=3, groups=channels)
            - left_mean.square()
        )
        right_variance = (
            functional.conv2d(right.square(), window, padding=3, groups=channels)
            - right_mean.square()
        )
        covariance = (
            functional.conv2d(left * right, window, padding=3, groups=channels)
            - left_mean * right_mean
        )
        numerator = (2 * left_mean * right_mean + 0.01**2) * (2 * covariance + 0.03**2)
        denominator = (left_mean.square() + right_mean.square() + 0.01**2) * (
            left_variance + right_variance + 0.03**2
        ) + 1e-12
        return (numerator / denominator).mean(dim=(1, 2, 3))

    def _gaussian_window(
        self,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a cached 7x7 Gaussian window for grouped SSIM convolution."""

        key = (channels, 7, 1.5, str(device), str(dtype))
        cached = self._gaussian_windows.get(key)
        if cached is not None:
            return cached
        axis = torch.arange(7, dtype=dtype, device=device) - 3
        gaussian = torch.exp(-0.5 * (axis / 1.5).square())
        kernel = gaussian / gaussian.sum()
        window = (
            (kernel.unsqueeze(1) @ kernel.unsqueeze(0))
            .expand(channels, 1, 7, 7)
            .contiguous()
        )
        self._gaussian_windows[key] = window
        return window

    @staticmethod
    def _downsample(images: torch.Tensor, scale: int) -> torch.Tensor:
        """Area-downsample a batch for multiscale SSIM."""

        if scale == 1:
            return images
        height, width = images.shape[-2:]
        return functional.interpolate(
            images,
            size=(max(1, height // scale), max(1, width // scale)),
            mode="area",
        )

    @staticmethod
    def _flow_magnitude(
        left: torch.Tensor,
        right: torch.Tensor,
        maximum_side: int = 256,
    ) -> float:
        """Return mean Farneback flow magnitude, or zero without OpenCV."""

        try:
            cv2: Any = import_module("cv2")
            numpy: Any = import_module("numpy")
        except ImportError:
            return _translation_magnitude(left, right, maximum_side)
        left_array: Any = (
            (left.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")[0]
        )
        right_array: Any = (
            (right.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")[0]
        )

        def gray(array: Any) -> Any:
            if array.ndim == 2:
                return array
            channels = array.shape[-1]
            if channels == 1:
                return array[..., 0]
            if channels == 3:
                return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
            if channels == 4:
                return cv2.cvtColor(array, cv2.COLOR_RGBA2GRAY)
            return array.mean(axis=-1).astype(array.dtype)

        left_gray = gray(left_array)
        right_gray = gray(right_array)
        height, width = left_gray.shape
        scale = max(1.0, max(height, width) / maximum_side)
        if scale > 1:
            resized = (round(width / scale), round(height / scale))
            left_gray = cv2.resize(left_gray, resized, interpolation=cv2.INTER_AREA)
            right_gray = cv2.resize(right_gray, resized, interpolation=cv2.INTER_AREA)
        try:
            flow: Any = cv2.calcOpticalFlowFarneback(
                left_gray, right_gray, None, 0.5, 3, 21, 3, 5, 1.1, 0
            )
            magnitude: Any = numpy.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            return float(magnitude.mean())
        except Exception:
            return _translation_magnitude(left, right, maximum_side)

    @staticmethod
    def _precision_context(
        device: torch.device,
        options: LoopAutocropOptions,
    ) -> AbstractContextManager[None]:
        """Use mixed precision only for requested CUDA scoring."""

        if device.type != "cuda" or not options.use_mixed_precision:
            return nullcontext()
        return torch.autocast(device_type="cuda")

    @staticmethod
    def _progress_bar(total: int) -> ProgressBar:
        """Construct a Comfy progress bar through a typed protocol."""

        comfy_utils = import_module("comfy.utils")
        return cast(ProgressBar, comfy_utils.ProgressBar(total))


def parse_scales(value: str) -> list[int]:
    """Parse unique positive SSIM downsample scales with a safe default."""

    scales: list[int] = []
    for item in value.split(","):
        try:
            scale = int(item.strip())
        except ValueError:
            continue
        if scale >= 1 and scale not in scales:
            scales.append(scale)
    return scales or [1]


def _translation_magnitude(
    left: torch.Tensor,
    right: torch.Tensor,
    maximum_side: int,
) -> float:
    """Estimate global motion by phase correlation when OpenCV is unavailable."""

    def grayscale(images: torch.Tensor) -> torch.Tensor:
        values = images.detach().to(device="cpu", dtype=torch.float32).movedim(-1, 1)
        if values.shape[1] == 1:
            return values
        return (
            0.2126 * values[:, 0:1] + 0.7152 * values[:, 1:2] + 0.0722 * values[:, 2:3]
        )

    left_gray = grayscale(left)
    right_gray = grayscale(right)
    height, width = left_gray.shape[-2:]
    resize_factor = max(1.0, max(height, width) / max(1, maximum_side))
    if resize_factor > 1:
        size = (round(height / resize_factor), round(width / resize_factor))
        left_gray = functional.interpolate(left_gray, size=size, mode="area")
        right_gray = functional.interpolate(right_gray, size=size, mode="area")
    left_gray -= left_gray.mean()
    right_gray -= right_gray.mean()
    cross_power = torch.fft.rfft2(left_gray) * torch.fft.rfft2(right_gray).conj()
    amplitude = cross_power.abs()
    if float(amplitude.max()) <= 1e-12:
        return 0.0
    correlation = torch.fft.irfft2(
        cross_power / amplitude.clamp_min(1e-12),
        s=left_gray.shape[-2:],
    ).abs()
    peak = int(correlation.reshape(-1).argmax())
    peak_y, peak_x = divmod(peak, correlation.shape[-1])
    if peak_y > correlation.shape[-2] // 2:
        peak_y -= correlation.shape[-2]
    if peak_x > correlation.shape[-1] // 2:
        peak_x -= correlation.shape[-1]
    return math.hypot(peak_x, peak_y) * resize_factor


__all__ = ["LoopAutocropRuntime", "PrecomputedMetrics", "parse_scales"]
