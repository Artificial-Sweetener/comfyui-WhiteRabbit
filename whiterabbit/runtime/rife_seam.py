# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

"""RIFE-based timing calibration for synthesized loop seam frames."""

from __future__ import annotations

import torch

from ..shared.tensor_validation import validate_image_batch
from .rife_interpolation import RifeInterpolationEngine


class RifeSeamTimingAnalyzer:
    """Solve blend positions whose visual steps match real source motion."""

    def __init__(self, interpolation: RifeInterpolationEngine | None = None) -> None:
        """Create the analyzer with an injectable interpolation engine."""

        self._interpolation = interpolation or RifeInterpolationEngine()

    def analyze(
        self,
        model_name: str,
        scale_factor: float,
        ensemble: bool,
        full_clip: torch.Tensor,
        multiplier: int,
        use_first_two: bool,
        use_last_two: bool,
        use_global_median: bool,
        metric: str,
        iterations: int,
        minimum_timing: float,
        maximum_timing: float,
        auto_maximum: bool = False,
        timing_cap: float = 0.995,
    ) -> tuple[str, int]:
        """Return a sorted CSV timing list and the normalized multiplier."""

        multiplier = max(0, int(multiplier))
        if multiplier == 0:
            return "", 0
        shape = validate_image_batch(full_clip, name="full_clip")
        if shape.batch_size < 2:
            raise ValueError("full_clip must contain at least 2 frames.")
        distances = [
            self._distance(
                full_clip[index : index + 1],
                full_clip[index + 1 : index + 2],
                metric,
            )
            for index in range(shape.batch_size - 1)
        ]
        selected: list[torch.Tensor] = []
        if use_first_two:
            selected.append(distances[0])
        if use_last_two:
            selected.append(distances[-1])
        if use_global_median:
            if shape.batch_size < 3:
                raise ValueError(
                    "use_global_median requires full_clip with >= 3 frames."
                )
            selected.append(torch.stack(distances).median())
        if not selected:
            raise ValueError(
                "Enable at least one of: use_first_two, use_last_two, "
                "use_global_median."
            )
        target = (
            selected[0] if len(selected) == 1 else torch.stack(selected).median()
        ).item()
        minimum = max(0.0, min(1.0, minimum_timing))
        maximum = max(0.0, min(1.0, maximum_timing))
        if not minimum < maximum:
            raise ValueError("Require 0 <= t_min < t_max <= 1.0")
        if auto_maximum:
            maximum = max(
                maximum,
                min(max(minimum + 1e-6, timing_cap), 0.9999),
            )
        last = full_clip[-1:]
        first = full_clip[:1]

        def synthesize(timing: float) -> torch.Tensor:
            return self._interpolation.synthesize(
                model_name,
                last,
                first,
                max(minimum, min(maximum, timing)),
                scale_factor,
                ensemble,
            )

        low, high = minimum, maximum
        for _ in range(iterations):
            middle = (low + high) / 2
            if self._distance(synthesize(middle), first, metric).item() > target:
                low = middle
            else:
                high = middle
        latest = (low + high) / 2
        previous = synthesize(latest)
        timings = [latest]
        for _ in range(multiplier - 1):
            low, high = minimum, timings[-1]
            for _ in range(iterations):
                middle = (low + high) / 2
                if self._distance(synthesize(middle), previous, metric).item() > target:
                    low = middle
                else:
                    high = middle
            timing = (low + high) / 2
            timings.append(timing)
            previous = synthesize(timing)
        return ", ".join(f"{timing:.6f}" for timing in sorted(timings)), multiplier

    @staticmethod
    def _distance(
        left: torch.Tensor,
        right: torch.Tensor,
        metric: str,
    ) -> torch.Tensor:
        """Return scalar L1 or MSE frame distance."""

        difference = left - right
        return difference.abs().mean() if metric == "L1" else difference.square().mean()


__all__ = ["RifeSeamTimingAnalyzer"]
