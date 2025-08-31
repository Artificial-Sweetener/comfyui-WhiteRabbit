# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>
import math
from typing import List, Optional, Tuple

import comfy.utils as comfy_utils
import torch
import torch.nn.functional as F
from comfy import model_management
from torchlanc import lanczos_resize


class UpscaleWithModelAdvanced:
    DESCRIPTION = """Based on Comfy's native "Upscale Image (using Model)", with controls exposed to tune for large batches, avoid slow
OOM fallbacks, and create opportunities to optimize for speed.

Defaults
- Behaves about the same as the original node.

Controls
- max_batch_size > 0: process images in chunks to keep VRAM steady and reduce fallback slowdowns.
- tile_size: choose a starting tile; original node defaults to 512. 0 = auto (falls back 512 → 256 → 128 on OOM).
- channels_last: try ON for a speedup on some systems.
- precision: lower (fp16/bf16) can be faster; may impact quality depending on the model.

More from me!: https://artificialsweetener.ai
"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": (
                    "UPSCALE_MODEL",
                    {"tooltip": "Pick your ESRGAN model (e.g. 2× / 4×)."},
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Images to upscale. Accepts a batch: frames×H×W×C with values in [0–1]."
                    },
                ),
            },
            "optional": {
                "max_batch_size": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "How many images to process at once. 0 = all at once. Set >0 if you hit OOM.",
                    },
                ),
                "tile_size": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2048,
                        "step": 32,
                        "tooltip": "How big each tile is. 0 = auto (starts at 512 and halves on OOM). Bigger is faster; smaller is safer.",
                    },
                ),
                "channels_last": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Try this ON for a small speed boost on some GPUs. If you see no gain, leave it OFF.",
                    },
                ),
                "precision": (
                    ["fp32", "fp16", "bf16"],
                    {
                        "default": "fp32",
                        "tooltip": "Math mode. fp32 = safest. fp16/bf16 can be faster on many GPUs, may impact image quality.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(
        self,
        upscale_model,
        image,
        max_batch_size=0,
        tile_size=0,
        channels_last=False,
        precision="fp32",
    ):
        def spans(n, cap):
            if cap <= 0 or cap >= n:
                return [(0, n)]
            out = []
            i = 0
            while i < n:
                j = min(n, i + cap)
                out.append((i, j))
                i = j
            return out

        device = model_management.get_torch_device()

        upscale_model.to(device)
        for p in upscale_model.model.parameters():
            if p.device != device:
                p.data = p.data.to(device)
                if p._grad is not None:
                    p._grad.data = p._grad.data.to(device)

        upscale_model.model.eval()

        scale = float(getattr(upscale_model, "scale", 4.0))
        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (
            (512 * 512 * 3) * image.element_size() * max(scale, 1.0) * 384.0
        )
        memory_required += image.nelement() * image.element_size()
        model_management.free_memory(memory_required, device)

        B, H, W, C = image.shape
        out_chunks = []

        for s, e in spans(B, int(max_batch_size)):
            sub = image[s:e].movedim(-1, -3).to(device, non_blocking=True)

            if channels_last and device.type == "cuda":
                sub = sub.to(memory_format=torch.channels_last)

            tile = 512 if tile_size in (0, None) else int(tile_size)
            overlap = 32

            oom = True
            while oom:
                try:
                    steps = sub.shape[0] * comfy_utils.get_tiled_scale_steps(
                        sub.shape[3],
                        sub.shape[2],
                        tile_x=tile,
                        tile_y=tile,
                        overlap=overlap,
                    )
                    pbar = comfy_utils.ProgressBar(steps)

                    if device.type == "cuda" and precision in ("fp16", "bf16"):
                        amp_dtype = (
                            torch.float16 if precision == "fp16" else torch.bfloat16
                        )
                        with torch.autocast(
                            device_type="cuda", dtype=amp_dtype
                        ), torch.inference_mode():
                            sr = comfy_utils.tiled_scale(
                                sub,
                                lambda a: upscale_model(a),
                                tile_x=tile,
                                tile_y=tile,
                                overlap=overlap,
                                upscale_amount=scale,
                                pbar=pbar,
                            )

                    else:
                        with torch.inference_mode():
                            sr = comfy_utils.tiled_scale(
                                sub,
                                lambda a: upscale_model(a),
                                tile_x=tile,
                                tile_y=tile,
                                overlap=overlap,
                                upscale_amount=scale,
                                pbar=pbar,
                            )

                    oom = False
                except model_management.OOM_EXCEPTION as e:
                    tile //= 2
                    if tile < 128:
                        raise e

            out_chunks.append(
                torch.clamp(sr.movedim(-3, -1), 0.0, 1.0).to("cpu", non_blocking=True)
            )

        upscale_model.to("cpu")
        return (torch.cat(out_chunks, dim=0),)


def _chunk_spans(n: int, max_bs: int) -> List[Tuple[int, int]]:
    if max_bs <= 0 or max_bs >= n:
        return [(0, n)]
    s, spans = 0, []
    while s < n:
        e = min(n, s + max_bs)
        spans.append((s, e))
        s = e
    return spans


def _floor_mul(x: int, k: int) -> int:
    if k <= 1:
        return max(1, x)
    return x - (x % k)


def _ceil_mul(x: int, k: int) -> int:
    if k <= 1:
        return max(1, x)
    r = x % k
    return x if r == 0 else x + (k - r)


def _fit_keep_aspect(sw: int, sh: int, tw: int, th: int) -> Tuple[int, int]:
    if tw <= 0 and th <= 0:
        return sw, sh
    if tw <= 0:
        r = th / sh
    elif th <= 0:
        r = tw / sw
    else:
        r = min(tw / sw, th / sh)
    return max(1, int(round(sw * r))), max(1, int(round(sh * r)))


def _fit_keep_ar_divisible(
    sw: int, sh: int, tw: int, th: int, d: int
) -> Tuple[int, int]:
    if d <= 1:
        return _fit_keep_aspect(sw, sh, tw, th)
    fw, fh = _fit_keep_aspect(sw, sh, tw, th)
    g = math.gcd(sw, sh)
    base_w = d * (sw // g)
    base_h = d * (sh // g)
    k = min(fw // base_w, fh // base_h)
    if k >= 1:
        return base_w * k, base_h * k
    return max(d, _floor_mul(fw, d)), max(d, _floor_mul(fh, d))


def _scale_then_crop_divisible(
    sw: int, sh: int, req_w: int, req_h: int, d: int
) -> Tuple[int, int, int, int]:
    """
    AR Scale + Divisible Crop:
      1) Scale once (keep AR), locking the SOURCE long side to floor(requested_long/d)*d (>0).
      2) Crop ONLY the short side to the largest multiple of d that is ≤ scaled short side and ≤ requested short side (>0).
    """
    d = max(1, int(d))
    req_w = max(1, int(req_w))
    req_h = max(1, int(req_h))
    req_w_div = _floor_mul(req_w, d)
    req_h_div = _floor_mul(req_h, d)

    src_long_is_h = sh >= sw

    if src_long_is_h:
        if req_h_div == 0:
            raise ValueError(
                f"AR Scale + Divisible Crop: requested height {req_h}px < divisible_by {d}."
            )
        scale = req_h_div / sh
        rh = req_h_div
        rw = max(1, int(round(sw * scale)))
        if rw < d:
            raise ValueError(
                f"AR Scale + Divisible Crop: scaled width {rw}px < divisible_by {d}."
            )
        if req_w_div == 0:
            raise ValueError(
                f"AR Scale + Divisible Crop: requested width {req_w}px < divisible_by {d}."
            )
        out_w = min(req_w_div, _floor_mul(rw, d))
        out_h = rh
    else:
        if req_w_div == 0:
            raise ValueError(
                f"AR Scale + Divisible Crop: requested width {req_w}px < divisible_by {d}."
            )
        scale = req_w_div / sw
        rw = req_w_div
        rh = max(1, int(round(sh * scale)))
        if rh < d:
            raise ValueError(
                f"AR Scale + Divisible Crop: scaled height {rh}px < divisible_by {d}."
            )
        if req_h_div == 0:
            raise ValueError(
                f"AR Scale + Divisible Crop: requested height {req_h}px < divisible_by {d}."
            )
        out_h = min(req_h_div, _floor_mul(rh, d))
        out_w = rw

    if (rw % d == 0) and (rh % d == 0):
        return rw, rh, rw, rh

    return rw, rh, out_w, out_h


def _cover_keep_aspect(sw: int, sh: int, tw: int, th: int) -> Tuple[int, int]:
    r = max(tw / sw, th / sh)
    return max(1, int((sw * r) + 0.999999)), max(1, int((sh * r) + 0.999999))


def _pad_sides(pos: str, pad_w: int, pad_h: int) -> Tuple[int, int, int, int]:
    lw = pad_w // 2
    rw = pad_w - lw
    th = pad_h // 2
    bh = pad_h - th
    if pos in ("top-left", "top", "top-right"):
        th, bh = 0, pad_h
    if pos in ("bottom-left", "bottom", "bottom-right"):
        th, bh = pad_h, 0
    if pos in ("top-left", "left", "bottom-left"):
        lw, rw = 0, pad_w
    if pos in ("top-right", "right", "bottom-right"):
        lw, rw = pad_w, 0
    return lw, rw, th, bh


def _crop_offsets(
    pos: str, in_w: int, in_h: int, out_w: int, out_h: int
) -> Tuple[int, int]:
    dx = max(0, in_w - out_w)
    dy = max(0, in_h - out_h)
    mapx = {
        "top-left": "left",
        "left": "left",
        "bottom-left": "left",
        "top": "center",
        "center": "center",
        "bottom": "center",
        "top-right": "right",
        "right": "right",
        "bottom-right": "right",
    }
    mapy = {
        "top-left": "top",
        "top": "top",
        "top-right": "top",
        "left": "center",
        "center": "center",
        "right": "center",
        "bottom-left": "bottom",
        "bottom": "bottom",
        "bottom-right": "bottom",
    }
    lx = {"left": 0, "center": dx // 2, "right": dx}.get(
        mapx.get(pos, "center"), dx // 2
    )
    ly = {"top": 0, "center": dy // 2, "bottom": dy}.get(
        mapy.get(pos, "center"), dy // 2
    )
    return lx, ly


def _parse_pad_color(
    s: str, c: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    s = (s or "").strip()
    if not s:
        return torch.zeros(c, device=device, dtype=dtype)
    try:
        parts = [int(p.strip()) for p in s.split(",")]
    except Exception:
        parts = [0, 0, 0]
    rgb = [int(max(0, min(255, v))) for v in (parts + [0, 0, 0])[:3]]
    v = torch.tensor(
        [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0], device=device, dtype=dtype
    )
    if c == 1:
        return v[:1]
    if c == 4:
        return torch.cat([v, torch.ones(1, device=device, dtype=dtype)])
    return v


def _nearest_interp(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    try:
        return F.interpolate(x, size=size, mode="nearest-exact")
    except Exception:
        return F.interpolate(x, size=size, mode="nearest")


def _divisible_box(w: int, h: int, d: int) -> Tuple[int, int]:
    if d <= 1:
        return int(w), int(h)
    return _floor_mul(int(w), d), _floor_mul(int(h), d)


def _normalize_mode(mode: str) -> str:
    key = (mode or "").strip().lower()
    table = {
        "keep ar": "keep_ar",
        "stretch": "stretch",
        "crop (cover + crop)": "crop",
        "pad (fit + pad)": "pad",
        "ar scale + divisible crop": "ar_scale_crop_divisible",
    }
    if key not in table:
        raise ValueError(
            "Unknown resize_mode. Use one of: "
            "'Keep AR', 'Stretch', 'Crop (Cover + Crop)', 'Pad (Fit + Pad)', 'AR Scale + Divisible Crop'."
        )
    return table[key]


class BatchResizeWithLanczos:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Input batch (B,H,W,C) in [0,1] float.\nProcessed on GPU."
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Target width (pixels).\n\n"
                        "Notes:\n"
                        "• Keep AR / Pad: maximum width for the fit\n"
                        "• Crop: final output width\n"
                        "• AR Scale + Divisible Crop: requested width before divisibility",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 576,
                        "min": 1,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Target height (pixels).\n\n"
                        "Notes:\n"
                        "• Keep AR / Pad: maximum height for the fit\n"
                        "• Crop: final output height\n"
                        "• AR Scale + Divisible Crop: requested height before divisibility",
                    },
                ),
                "resize_mode": (
                    [
                        "Keep AR",
                        "Stretch",
                        "Crop (Cover + Crop)",
                        "Pad (Fit + Pad)",
                        "AR Scale + Divisible Crop",
                    ],
                    {
                        "default": "Keep AR",
                        "tooltip": "Modes:\n"
                        "- Keep AR: Fit inside width×height (preserve aspect)\n"
                        "- Stretch: Force to width×height (may distort)\n"
                        "- Crop (Cover + Crop): Scale to cover, then crop to width×height\n"
                        "- Pad (Fit + Pad): Fit inside, then pad to width×height\n"
                        "- AR Scale + Divisible Crop: Scale by SOURCE long side to ≤ requested divisible; crop ONLY the short side to its divisible",
                    },
                ),
                "divisible_by": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Force output dimensions to multiples of N.\n\n"
                        "Details:\n"
                        "• Keep AR: Fit → then step down to the largest size ≤ requested that keeps AR AND makes both sides divisible\n"
                        "• AR Scale + Divisible Crop: Lock the scaled LONG side to its divisible target; crop ONLY the short side to its divisible\n"
                        "Set to 1 (or 0 in UI) to disable",
                    },
                ),
                "max_batch_size": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "0 = process whole batch\n>0 = chunk the batch to this size",
                    },
                ),
                "sinc_window": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Lanczos window size (a). Higher = sharper (more ringing).",
                    },
                ),
                "pad_color": (
                    "STRING",
                    {
                        "default": "0, 0, 0",
                        "tooltip": "Pad mode only. RGB as 'r, g, b' (0-255).",
                    },
                ),
                "crop_position": (
                    [
                        "center",
                        "top-left",
                        "top",
                        "top-right",
                        "left",
                        "right",
                        "bottom-left",
                        "bottom",
                        "bottom-right",
                    ],
                    {
                        "default": "center",
                        "tooltip": "Where to crop/pad from.\nChoose which edges are preserved for cropping, or where padding is added.",
                    },
                ),
                "precision": (
                    ["fp32", "fp16", "bf16"],
                    {"default": "fp32", "tooltip": "Resampling compute dtype."},
                ),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {
                        "tooltip": "Optional mask (B,H,W) in [0,1].\nResized with nearest.\nFollows the same crop/pad as the image."
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK")
    RETURN_NAMES = ("IMAGE", "width", "height", "mask")
    FUNCTION = "process"
    CATEGORY = "image/resize"
    DESCRIPTION = (
        "CUDA-accelerated, gamma-correct Lanczos resizer (TorchLanc).\n\n"
        "Modes:\n"
        "• Keep AR\n"
        "• Stretch\n"
        "• Crop (Cover + Crop)\n"
        "• Pad (Fit + Pad)\n"
        "• AR Scale + Divisible Crop\n\n"
        "Node functionality based on Resize nodes by Kijai\n\n"
        "More from me!: https://artificialsweetener.ai"
    )

    def process(
        self,
        image: torch.Tensor,
        width: int,
        height: int,
        resize_mode: str,
        divisible_by: int,
        max_batch_size: int,
        sinc_window: int,
        pad_color: str,
        crop_position: str,
        precision: str,
        mask: Optional[torch.Tensor] = None,
    ):
        if image is None or not isinstance(image, torch.Tensor):
            raise ValueError(
                "image must be a torch.Tensor of shape (B,H,W,C) in [0,1]."
            )

        B, H, W, C = image.shape
        if C not in (1, 3, 4):
            raise ValueError(f"Unsupported channel count C={C}. Expected 1, 3 or 4.")

        d = int(divisible_by) if int(divisible_by) > 1 else 1
        mode = _normalize_mode(resize_mode)

        device = torch.device("cuda")
        image = image.float().clamp_(0, 1)

        if mode == "stretch":
            tw, th = _divisible_box(width, height, d)
            rw, rh = tw, th
            out_w, out_h = tw, th

        elif mode == "keep_ar":
            rw, rh = _fit_keep_ar_divisible(W, H, int(width), int(height), d)
            out_w, out_h = rw, rh

        elif mode == "ar_scale_crop_divisible":
            if width <= 0 or height <= 0:
                raise ValueError(
                    "AR Scale + Divisible Crop requires non-zero width and height."
                )
            rw, rh, out_w, out_h = _scale_then_crop_divisible(
                W, H, int(width), int(height), d
            )

        elif mode == "crop":
            if width <= 0 or height <= 0:
                raise ValueError("Crop requires non-zero width and height.")
            tw, th = _divisible_box(width, height, d)
            rw, rh = _cover_keep_aspect(W, H, tw, th)
            out_w, out_h = tw, th

        elif mode == "pad":
            if width <= 0 or height <= 0:
                raise ValueError("Pad requires non-zero width and height.")
            tw, th = _divisible_box(width, height, d)
            rw, rh = _fit_keep_aspect(W, H, tw, th)
            out_w, out_h = tw, th

        else:
            raise ValueError(f"Unknown resize_mode: {resize_mode}")

        out_imgs: List[torch.Tensor] = []
        out_masks: List[torch.Tensor] = []

        crop_like = mode in ("crop", "ar_scale_crop_divisible")
        pad_like = mode == "pad"
        resize_to = (rh, rw) if (crop_like or pad_like) else (out_h, out_w)

        pbar = comfy_utils.ProgressBar(B)

        for s, e in _chunk_spans(B, int(max_batch_size)):
            x = image[s:e].movedim(-1, 1).to(device, non_blocking=True)

            y = lanczos_resize(
                x,
                height=resize_to[0],
                width=resize_to[1],
                a=int(sinc_window),
                precision=str(precision),
                clamp=True,
                chunk_size=0,
            )

            ox = oy = 0
            left = right = top = bottom = 0

            if crop_like:
                ox, oy = _crop_offsets(crop_position, rw, rh, out_w, out_h)
                y = y[:, :, oy : oy + out_h, ox : ox + out_w]
            elif pad_like:
                pad_w = max(0, out_w - rw)
                pad_h = max(0, out_h - rh)
                left, right, top, bottom = _pad_sides(crop_position, pad_w, pad_h)

                if d > 1:
                    base_w = rw + left + right
                    base_h = rh + top + bottom
                    right += _ceil_mul(base_w, d) - base_w
                    bottom += _ceil_mul(base_h, d) - base_h
                    out_w = rw + left + right
                    out_h = rh + top + bottom

                color = _parse_pad_color(pad_color, C, y.device, y.dtype).view(
                    1, C, 1, 1
                )
                canvas = color.expand(y.shape[0], -1, out_h, out_w).clone()
                canvas[:, :, top : top + rh, left : left + rw] = y
                y = canvas

            out_imgs.append(y.to("cpu", non_blocking=False).movedim(1, -1))

            if isinstance(mask, torch.Tensor):
                m = mask[s:e].unsqueeze(1).to(device, non_blocking=True)
                m_res = _nearest_interp(m, size=resize_to)
                if crop_like:
                    m_res = m_res[:, :, oy : oy + out_h, ox : ox + out_w]
                elif pad_like:
                    base = torch.zeros(
                        (m_res.shape[0], 1, out_h, out_w),
                        device=m_res.device,
                        dtype=m_res.dtype,
                    )
                    base[:, :, top : top + rh, left : left + rw] = m_res
                    m_res = base
                out_masks.append(m_res.squeeze(1).to("cpu", non_blocking=False))

            pbar.update(e - s)

        images_out = torch.cat(out_imgs, dim=0)
        mask_out = (
            torch.cat(out_masks, dim=0)
            if out_masks
            else torch.zeros((B, out_h, out_w), dtype=torch.float32)
        )

        return images_out, out_w, out_h, mask_out


NODE_CLASS_MAPPINGS = {"BatchResizeWithLanczos": BatchResizeWithLanczos}
NODE_DISPLAY_NAME_MAPPINGS = {"BatchResizeWithLanczos": "Batch Resize with Lanczos"}
