# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener

import os
import random
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import comfy.utils as comfy_utils
import folder_paths
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchlanc import lanczos_resize


def _chunk_spans(n: int, cap: int) -> List[Tuple[int, int]]:
    if cap <= 0 or cap >= n:
        return [(0, n)]
    out = []
    i = 0
    while i < n:
        j = min(n, i + cap)
        out.append((i, j))
        i = j
    return out


def _bhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    return x.movedim(-1, -3)


def _nchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    return x.movedim(-3, -1)


def _ensure_rgba_nchw(wm: torch.Tensor) -> torch.Tensor:
    """
    wm: (1,H,W,C) in [0,1] → return (4,H,W) float
    C may be 1,3,4; synthesize alpha=1 if missing.
    """
    if wm.dim() != 4 or wm.shape[0] != 1:
        raise ValueError(
            "watermark must be a single IMAGE tensor of shape (1,H,W,C) in [0,1]."
        )
    _, h, w, c = wm.shape
    x = _bhwc_to_nchw(wm[0]).float().clamp_(0, 1)  # (C,H,W)
    if c == 4:
        return x
    if c == 3:
        a = torch.ones(1, h, w, device=x.device, dtype=x.dtype)
        return torch.cat([x, a], dim=0)
    if c == 1:
        rgb = x.repeat(3, 1, 1)
        a = torch.ones(1, h, w, device=x.device, dtype=x.dtype)
        return torch.cat([rgb, a], dim=0)
    raise ValueError(f"Unsupported watermark channel count C={c}. Expected 1, 3 or 4.")


def _load_rgba_from_path(path: str, device: torch.device) -> torch.Tensor:
    """
    Load an image from disk as RGBA in [0,1] and return (4,H,W) on the target device.
    No rotation or other processing happens here.
    """
    try:
        with Image.open(path) as im:
            im = im.convert("RGBA")
            arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,4) in [0,1]
    except Exception as e:
        raise ValueError(f"Failed to load watermark image from '{path}': {e}")
    t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)  # (H,W,4)
    return t.permute(2, 0, 1).contiguous()  # (4,H,W)


def _rotate_bicubic_expand(x: torch.Tensor, degrees: float) -> torch.Tensor:
    """
    x: (N,C,H,W). Rotate around center with bicubic sampling and EXPAND canvas
    (PIL-like `expand=True`). Parts outside input are zero/transparent.
    """
    deg = float(degrees) % 360.0
    if deg == 0.0:
        return x

    N, C, H, W = x.shape
    rad = deg * 3.141592653589793 / 180.0
    cosr = float(torch.cos(torch.tensor(rad)))
    sinr = float(torch.sin(torch.tensor(rad)))

    # Expanded output size (axis-aligned bounding box of the rotated rectangle)
    new_w = int((abs(W * cosr) + abs(H * sinr)) + 0.9999)
    new_h = int((abs(H * cosr) + abs(W * sinr)) + 0.9999)
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    # Centers in pixel coords
    cx_in = (W - 1) * 0.5
    cy_in = (H - 1) * 0.5
    cx_out = (new_w - 1) * 0.5
    cy_out = (new_h - 1) * 0.5

    # Output grid in pixel coords
    ys = torch.linspace(0, new_h - 1, new_h, device=x.device, dtype=x.dtype)
    xs = torch.linspace(0, new_w - 1, new_w, device=x.device, dtype=x.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    # Inverse rotation: output → input (rotate about centers)
    rx = gx - cx_out
    ry = gy - cy_out
    x_in = cosr * rx + sinr * ry + cx_in
    y_in = -sinr * rx + cosr * ry + cy_in

    # Normalize to [-1,1] for align_corners=False
    x_norm = (x_in + 0.5) / W * 2.0 - 1.0
    y_norm = (y_in + 0.5) / H * 2.0 - 1.0
    grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0).repeat(N, 1, 1, 1)

    # Sample
    try:
        return F.grid_sample(
            x, grid, mode="bicubic", padding_mode="zeros", align_corners=False
        )
    except Exception:
        return F.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )


def _position_xy(
    position: str,
    base_w: int,
    base_h: int,
    wm_w: int,
    wm_h: int,
    pad_x: int,
    pad_y: int,
) -> Tuple[int, int]:
    pos = (position or "bottom-right").strip().lower()
    if pos == "center":
        return (base_w - wm_w) // 2, (base_h - wm_h) // 2

    x = (
        0
        if "left" in pos
        else (base_w - wm_w if "right" in pos else (base_w - wm_w) // 2)
    )
    y = (
        0
        if "top" in pos
        else (base_h - wm_h if "bottom" in pos else (base_h - wm_h) // 2)
    )

    if "left" in pos:
        x += int(pad_x)
    if "right" in pos:
        x -= int(pad_x)
    if "top" in pos:
        y += int(pad_y)
    if "bottom" in pos:
        y -= int(pad_y)
    return x, y


class _SmallLRU:
    def __init__(self, capacity: int = 6):
        self.capacity = int(max(1, capacity))
        self._m: "OrderedDict[Tuple, Tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()

    def get(self, key: Tuple):
        v = self._m.get(key)
        if v is not None:
            self._m.move_to_end(key)
        return v

    def put(self, key: Tuple, value):
        if key in self._m:
            self._m.move_to_end(key)
        self._m[key] = value
        if len(self._m) > self.capacity:
            self._m.popitem(last=False)


class BatchWatermarkSingle:
    """
    Single-position watermark for image batches.

    - Scale uses base image WIDTH × (scale/100)
    - Rotation always applies, with clipping (no expand)
    - Padding in pixels (ignored for center)
    - TorchLanc for watermark resize
    - Chunked batches + small LRU cache + optional torch.compile
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Mirror LoadImage: list files from the input directory, allow upload
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        files = folder_paths.filter_files_content_types(files, ["image"])

        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Images to watermark. Accepts (H,W,C) or (B,H,W,C) with values in [0–1]. Processed on GPU."
                    },
                ),
                "watermark": (
                    sorted(files),
                    {
                        "image_upload": True,
                        "tooltip": "Select or upload the watermark image (PNG recommended). The file’s transparency is preserved.",
                    },
                ),
                "position": (
                    ["bottom-right", "bottom-left", "top-right", "top-left", "center"],
                    {
                        "default": "bottom-right",
                        "tooltip": "Where to place the watermark. Padding is ignored when 'center' is selected. Rotation clips; no canvas expand.",
                    },
                ),
                "scale": (
                    "INT",
                    {
                        "default": 70,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Width-based scaling. Target watermark width = image width × (scale/100). Aspect ratio preserved.",
                    },
                ),
                "transparency": (
                    "INT",
                    {
                        "default": 100,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Alpha multiplier for the watermark: 100 = unchanged, 0 = fully transparent.",
                    },
                ),
                "rotation": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 359,
                        "step": 1,
                        "tooltip": "Rotate the watermark (degrees) with bicubic resampling. Canvas expands so nothing is clipped (PIL-style).",
                    },
                ),
                "padding_x": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Extra horizontal padding in pixels from the chosen edge (ignored when position='center').",
                    },
                ),
                "padding_y": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Extra vertical padding in pixels from the chosen edge (ignored when position='center').",
                    },
                ),
                "optical_padding": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Adjust placement by the watermark’s visual center so equal padding looks right (optical alignment). Affects corner positions; ignored when position='center'.",
                    },
                ),
                "optical_strength": (
                    "INT",
                    {
                        "default": 40,
                        "min": 0,
                        "max": 100,
                        "step": 5,
                        "tooltip": "How strongly to nudge toward visual centering (0–100). 0 = off. Higher values shift more for wide/rotated marks.",
                    },
                ),
                "max_batch_size": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Process images in chunks to control VRAM. 0 = process the whole batch at once.",
                    },
                ),
                "sinc_window": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Lanczos window size (a) used when resizing the watermark. Higher = sharper (but more ringing).",
                    },
                ),
                "precision": (
                    ["fp32", "fp16", "bf16"],
                    {
                        "default": "fp32",
                        "tooltip": "Resampling compute dtype. fp32 = safest quality; fp16/bf16 can be faster on many GPUs.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/post"
    DESCRIPTION = (
        "GPU accelerated watermark overlay. TorchLanc resize for quality and speed. Works for single images, but efficient for batches, too!"
    )

    def apply(
        self,
        image: torch.Tensor,
        watermark: str,
        position: str,
        scale: int,
        transparency: int,
        rotation: int,
        padding_x: int,
        padding_y: int,
        optical_padding: bool,
        optical_strength: int,
        max_batch_size: int,
        sinc_window: int,
        precision: str,
    ):

        if image is None or not isinstance(image, torch.Tensor):
            raise ValueError(
                "image must be a torch.Tensor with shape (H,W,C) or (B,H,W,C) in [0,1]."
            )
        if not isinstance(watermark, str) or not watermark:
            raise ValueError("Select a watermark image from the list (or upload one).")

        if not folder_paths.exists_annotated_filepath(watermark):
            raise ValueError(f"Invalid watermark file: {watermark}")
        watermark_path = folder_paths.get_annotated_filepath(watermark)

        # Refuse sequences (we must get a tensor just like Lanczos)
        if isinstance(image, (list, tuple)):
            raise TypeError(
                "Expected IMAGE tensor (H,W,C) or (B,H,W,C); got a sequence. Use 'Image Batch' to re-batch."
            )

        # Accept both single images (H,W,C) and batches (B,H,W,C); normalize to batch
        if image.dim() == 3:
            image = image.unsqueeze(0)  # -> (1,H,W,C)
        elif image.dim() != 4:
            raise ValueError(
                f"Unexpected IMAGE tensor rank {image.dim()}; expected 3 or 4 dims."
            )

        B, H, W, C = image.shape
        if C not in (1, 3, 4):
            raise ValueError(f"Unsupported channel count C={C}. Expected 1, 3 or 4.")

        # Common
        device = torch.device("cuda")
        scale = int(scale)
        transparency = max(0, min(100, int(transparency)))
        rotation = int(rotation) % 360
        pad_x = int(padding_x)
        pad_y = int(padding_y)
        optical_padding = bool(optical_padding)
        optical_strength = max(0, min(100, int(optical_strength)))

        # Prepare watermark once (load RGBA from disk to preserve original transparency)
        wm_rgba = _load_rgba_from_path(watermark_path, device)  # (4,hw,ww)
        wm_h0, wm_w0 = int(wm_rgba.shape[1]), int(wm_rgba.shape[2])

        # Progress
        pbar = comfy_utils.ProgressBar(B)

        out_chunks: List[torch.Tensor] = []

        # Compute final watermark once (all images in a Comfy batch share H×W)
        target_w = max(1, int(round(W * (scale / 100.0))))
        target_h = max(1, int(round(wm_h0 * target_w / max(1, wm_w0))))

        # Premultiply BEFORE resampling to avoid dark fringes
        pm0 = wm_rgba[:3, :, :] * wm_rgba[3:4, :, :]
        a0 = wm_rgba[3:4, :, :]
        wm_pm = torch.cat([pm0, a0], dim=0).unsqueeze(0)  # (1,4,hw,ww)

        wm_resized_pm = lanczos_resize(
            wm_pm,
            height=target_h,
            width=target_w,
            a=int(sinc_window),
            precision=str(precision),
            clamp=True,
            chunk_size=0,
        )[
            0
        ]  # (4,h,w)

        # Apply transparency uniformly to premultiplied color AND alpha
        if transparency != 100:
            t = float(transparency) / 100.0
            wm_resized_pm[:3, :, :].mul_(t)
            wm_resized_pm[3:4, :, :].mul_(t)

        # Rotate in premultiplied space (expand canvas)
        wm_final = _rotate_bicubic_expand(wm_resized_pm.unsqueeze(0), rotation)[
            0
        ]  # (4,h,w)
        pm_final, a_final = wm_final[:3, :, :], wm_final[3:4, :, :]  # (3,h,w), (1,h,w)

        # Position
        wm_h, wm_w = int(pm_final.shape[1]), int(pm_final.shape[2])
        x, y = _position_xy(position, W, H, wm_w, wm_h, pad_x, pad_y)

        # Optional optical padding (corner positions only)
        if optical_padding and position != "center":
            a = a_final[0]  # (h,w)
            denom = a.sum()
            if float(denom.item() if hasattr(denom, "item") else denom) > 1e-8:
                ys = torch.linspace(0, wm_h - 1, wm_h, device=a.device, dtype=a.dtype)
                xs = torch.linspace(0, wm_w - 1, wm_w, device=a.device, dtype=a.dtype)
                cy = (a.sum(dim=1) * ys).sum() / denom
                cx = (a.sum(dim=0) * xs).sum() / denom
                gx = (wm_w - 1) * 0.5
                gy = (wm_h - 1) * 0.5
                s = float(optical_strength) / 100.0
                dx = (gx - cx) * s  # positive when centroid is left of center
                dy = (gy - cy) * s  # positive when centroid is above center

                if "right" in position:
                    x += int(round(dx.item()))
                if "left" in position:
                    x -= int(round(dx.item()))
                if "bottom" in position:
                    y += int(round(dy.item()))
                if "top" in position:
                    y -= int(round(dy.item()))

        # Intersection with base image (clip)
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(W, x + wm_w)
        y1 = min(H, y + wm_h)

        if x1 <= x0 or y1 <= y0:
            out = image.to("cpu", non_blocking=False).float().clamp_(0, 1).contiguous()
            if not torch.is_tensor(out) or out.dim() != 4:
                raise TypeError(
                    f"Pass-through produced non-tensor or wrong rank: {type(out)} / {getattr(out,'shape',None)}"
                )
            return (out,)

        wx0 = x0 - x
        wy0 = y0 - y
        w_w = x1 - x0
        w_h = y1 - y0

        pm_crop = pm_final[:, wy0 : wy0 + w_h, wx0 : wx0 + w_w].contiguous()
        a_crop = a_final[:, wy0 : wy0 + w_h, wx0 : wx0 + w_w].contiguous()

        # Process in chunks
        for s, e in _chunk_spans(B, int(max_batch_size)):
            sub = (
                _bhwc_to_nchw(image[s:e])
                .to(device, non_blocking=True)
                .float()
                .clamp_(0, 1)
            )

            ov_pm = pm_crop.unsqueeze(0).expand(sub.shape[0], -1, -1, -1)
            ov_a = a_crop.unsqueeze(0).expand(sub.shape[0], -1, -1, -1)

            if C == 1:
                rgb = sub.repeat(1, 3, 1, 1)
                roi = rgb[:, :, y0:y1, x0:x1]
                roi_out = roi * (1.0 - ov_a) + ov_pm
                rgb[:, :, y0:y1, x0:x1] = roi_out
                # Convert back to 1ch (luma)
                y_luma = (
                    0.2126 * rgb[:, 0:1] + 0.7152 * rgb[:, 1:2] + 0.0722 * rgb[:, 2:3]
                ).clamp_(0, 1)
                sub = y_luma
            elif C == 3:
                roi = sub[:, :3, y0:y1, x0:x1]
                roi_out = roi * (1.0 - ov_a) + ov_pm
                sub[:, :3, y0:y1, x0:x1] = roi_out
            else:  # C == 4
                roi = sub[:, :3, y0:y1, x0:x1]
                roi_out = roi * (1.0 - ov_a) + ov_pm
                sub[:, :3, y0:y1, x0:x1] = roi_out

            out_chunks.append(
                _nchw_to_bhwc(sub).to("cpu", non_blocking=False).clamp_(0, 1)
            )
            pbar.update(e - s)

        out = torch.cat(out_chunks, dim=0)  # CPU BHWC chunks → CPU BHWC batch

        if out.dim() > 4:
            b_flat = 1
            for s in out.shape[:-3]:
                b_flat *= int(s)
            out = out.reshape(b_flat, *out.shape[-3:])
        if out.dim() == 3:
            out = out.unsqueeze(0)
        if (
            out.dim() == 4
            and out.shape[1] in (1, 3, 4)
            and out.shape[-1] not in (1, 3, 4)
        ):
            out = out.permute(0, 2, 3, 1).contiguous()
        if out.dim() != 4:
            raise ValueError(
                f"Unexpected IMAGE tensor shape {tuple(out.shape)}; expected (B,H,W,C)."
            )

        out = (
            out.to("cpu", non_blocking=False)
            .to(dtype=torch.float32)
            .clamp_(0, 1)
            .contiguous()
        )

        if not torch.is_tensor(out):
            raise TypeError(f"IMAGE output must be torch.Tensor, got: {type(out)}")

        return (out,)
