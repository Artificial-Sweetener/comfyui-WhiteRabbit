# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _to_lin(x):
    return torch.where(
        x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp(min=0) ** 2.4
    )


def _to_srgb(x):
    return torch.where(
        x <= 0.0031308, 12.92 * x, 1.055 * x.clamp(min=0) ** (1 / 2.4) - 0.055
    )


def _luma(x):
    return 0.2126 * x[..., 0:1] + 0.7152 * x[..., 1:2] + 0.0722 * x[..., 2:3]


def _sobel_mag(y):  # y: NHWC 1ch
    kx = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        .view(1, 1, 3, 3)
        .to(y.device)
    )
    ky = (
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        .view(1, 1, 3, 3)
        .to(y.device)
    )
    t = F.pad(y.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(t, kx)
    gy = F.conv2d(t, ky)
    return torch.sqrt(gx * gx + gy * gy).permute(0, 2, 3, 1).contiguous()


def _gauss1d(sigma, r):
    if r <= 0:
        return torch.tensor([1.0], dtype=torch.float32)
    xs = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(xs * xs) / (2 * sigma * sigma))
    return (k / k.sum()).contiguous()


def _blur_nhwc(x, sigma):
    if sigma <= 0:
        return x
    N, H, W, C = x.shape
    max_r = max(0, min(H, W) // 2 - 1)
    r = min(int(math.ceil(3.0 * sigma)), max_r)
    if r <= 0:
        return x
    k = _gauss1d(sigma, r)
    kH = k.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    kW = k.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    t = x.permute(0, 3, 1, 2).contiguous()
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode="reflect"), kH, groups=C)
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="reflect"), kW, groups=C)
    return t.permute(0, 2, 3, 1).contiguous()


def _avgpool_tiles(x1, tile):
    t = x1.permute(0, 3, 1, 2)
    o = F.avg_pool2d(t, kernel_size=tile, stride=tile)
    return o.permute(0, 2, 3, 1)


def _mad_tiles(x1, tile):
    t = x1.permute(0, 3, 1, 2)
    N, C, H, W = t.shape
    th, tw = H // tile, W // tile
    t = t[:, :, : th * tile, : tw * tile]
    patches = F.unfold(t, kernel_size=tile, stride=tile)  # (N, C*tile*tile, th*tw)
    patches = patches.transpose(1, 2).reshape(-1, tile * tile)  # (N*th*tw, K)
    med = patches.median(dim=1, keepdim=True).values
    mad = (patches - med).abs().median(dim=1).values.view(N, th, tw, 1)
    return mad


def _upsample_mask(mask_tile, H, W, mode="nearest"):
    t = mask_tile.permute(0, 3, 1, 2)
    t = F.interpolate(
        t,
        size=(H, W),
        mode=("bilinear" if mode == "bilinear" else "nearest"),
        align_corners=False if mode == "bilinear" else None,
    )
    return t.permute(0, 2, 3, 1)


def _dilate(mask01, r):
    if r <= 0:
        return mask01
    t = mask01.permute(0, 3, 1, 2)
    t = F.pad(t, (r, r, r, r), mode="replicate")
    t = F.max_pool2d(t, kernel_size=2 * r + 1, stride=1)
    return t.permute(0, 2, 3, 1)


def _resize_lanczos(img01, H, W):  # (1,Hr,Wr,C) float CPU -> (1,H,W,C) float CPU
    arr = (img01[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode="RGB").resize((W, H), resample=Image.LANCZOS)
    out = np.asarray(pil).astype(np.float32) / 255.0
    return torch.from_numpy(out).unsqueeze(0)


class PixelHold:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": (
                    "IMAGE",
                    {"tooltip": "Your clip (frames×H×W×C, values 0–1)."},
                ),
                "ref_source": (
                    ["external", "batch_index"],
                    {
                        "default": "external",
                        "tooltip": "Pick the reference: an external image or a frame from this clip.",
                    },
                ),
                "ref_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999999,
                        "tooltip": "If using a frame from this clip, which frame to use as the reference.",
                    },
                ),
                "reference": (
                    "IMAGE",
                    {
                        "default": None,
                        "tooltip": "Optional external reference (1×H×W×C). If sizes differ, it will be resized to match.",
                    },
                ),
                "linearize": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Work in linear color for steadier results on flat areas.",
                    },
                ),
                "auto_luma": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Auto sensitivity for brightness changes (adapts per frame).",
                    },
                ),
                "auto_k": (
                    "FLOAT",
                    {
                        "default": 2.5,
                        "min": 0.5,
                        "max": 6.0,
                        "step": 0.1,
                        "tooltip": "Auto strength. Higher = lock more to the reference (2–3 is typical).",
                    },
                ),
                "tau_luma": (
                    "FLOAT",
                    {
                        "default": 1.5 / 255.0,
                        "min": 0.0,
                        "max": 4.0 / 255.0,
                        "step": 0.0005,
                        "tooltip": "Manual brightness threshold when Auto is OFF. Lower = stricter (more locking).",
                    },
                ),
                "tau_grad": (
                    "FLOAT",
                    {
                        "default": 0.02,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "How much edge change to allow. Lower protects edges more.",
                    },
                ),
                "mode": (
                    ["tile", "pixel"],
                    {
                        "default": "tile",
                        "tooltip": "Tile: fast & robust. Pixel: finer but noisier.",
                    },
                ),
                "tile_size": (
                    "INT",
                    {
                        "default": 32,
                        "min": 8,
                        "max": 256,
                        "step": 8,
                        "tooltip": "Tile size when using Tile mode.",
                    },
                ),
                "score_mode": (
                    ["l1_tile", "mad_tile"],
                    {
                        "default": "l1_tile",
                        "tooltip": "How tiles measure change: mean abs diff (fast) or median abs dev (robust).",
                    },
                ),
                "edge_band": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Protect a belt around strong edges to avoid wobble/stretch.",
                    },
                ),
                "band_radius": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 64,
                        "tooltip": "Width of the protected belt (pixels).",
                    },
                ),
                "tau_edge_low": (
                    "FLOAT",
                    {
                        "default": 1.5 / 255.0,
                        "min": 0.0,
                        "max": 0.25,
                        "step": 0.0005,
                        "tooltip": "Treat as low-motion below this level (edge belt).",
                    },
                ),
                "tau_edge_high": (
                    "FLOAT",
                    {
                        "default": 6.0 / 255.0,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.0005,
                        "tooltip": "Treat as high-motion above this level (edge belt).",
                    },
                ),
                "apply": (
                    ["all", "lowfreq"],
                    {
                        "default": "all",
                        "tooltip": "Hold the whole image (All) or only its smooth part (Low-freq).",
                    },
                ),
                "dilate": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 16,
                        "tooltip": "Expand the mask (pixels).",
                    },
                ),
                "feather_sigma": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 16.0,
                        "step": 0.5,
                        "tooltip": "Soften mask edges (pixels).",
                    },
                ),
                "process_on": (
                    ["auto", "cpu", "gpu"],
                    {
                        "default": "auto",
                        "tooltip": "Choose CPU/GPU. Auto switches to GPU on very large frames.",
                    },
                ),
                "gpu_clear_every": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "tooltip": "If >0 and using GPU, free memory every N frames.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "mask_preview")
    FUNCTION = "apply_hold"
    CATEGORY = "video utils"
    DESCRIPTION = (
        "Locks parts of each frame to a chosen reference (external image or a frame from the clip) whenever changes are small—"
        "useful for stabilizing flat areas or backgrounds while leaving motion to pass through.\n\n"
        "More from me!: https://artificialsweetener.ai"
    )

    @torch.no_grad()
    def apply_hold(
        self,
        frames,
        ref_source="external",
        ref_index=0,
        reference=None,
        linearize=True,
        auto_luma=True,
        auto_k=2.5,
        tau_luma=1.5 / 255.0,
        tau_grad=0.02,
        mode="tile",
        tile_size=32,
        score_mode="l1_tile",
        edge_band=True,
        band_radius=4,
        tau_edge_low=1.5 / 255.0,
        tau_edge_high=6.0 / 255.0,
        apply="all",
        dilate=1,
        feather_sigma=2.0,
        process_on="auto",
        gpu_clear_every=0,
    ):
        x = frames if isinstance(frames, torch.Tensor) else torch.tensor(frames)
        B, H, W, C = x.shape

        if str(ref_source) == "external" and reference is not None:
            ref = (
                reference
                if isinstance(reference, torch.Tensor)
                else torch.tensor(reference)
            )
            if ref.shape[1] != H or ref.shape[2] != W:
                ref = _resize_lanczos(ref[:1].to("cpu"), H, W)
            ref = ref[:1].repeat(B, 1, 1, 1)
        else:
            idx = max(0, min(int(ref_index), B - 1))
            ref = x[idx : idx + 1].repeat(B, 1, 1, 1)

        x_lin = _to_lin(x) if linearize else x
        r_lin = _to_lin(ref) if linearize else ref
        want_gpu = (process_on == "gpu") or (
            process_on == "auto" and torch.cuda.is_available() and (H * W >= 6_000_000)
        )
        dev = torch.device("cuda") if want_gpu else torch.device("cpu")

        r_lin = r_lin.to(dev)
        y_r = _luma(r_lin)
        g_r = _sobel_mag(y_r)

        if apply == "lowfreq":
            LF_r = _blur_nhwc(r_lin.to("cpu"), 13.0)

        out_frames, mask_frames = [], []
        clear_ctr = 0

        for i in range(B):
            f = x_lin[i : i + 1].to(dev)
            y_f = _luma(f)
            g_f = _sobel_mag(y_f)

            dY = (y_f - y_r[i : i + 1]).abs()
            dG = (g_f - g_r[i : i + 1]).abs()

            if auto_luma:
                med = torch.median(dY.view(-1))
                sigma = 1.4826 * med.item()
                tau_luma_eff = max(0.0, min(4.0 / 255.0, float(auto_k) * float(sigma)))
            else:
                tau_luma_eff = float(tau_luma)

            if mode == "tile":
                sY = (
                    _mad_tiles(dY, tile_size)
                    if score_mode == "mad_tile"
                    else _avgpool_tiles(dY, tile_size)
                )
                sG = (
                    _mad_tiles(dG, tile_size)
                    if score_mode == "mad_tile"
                    else _avgpool_tiles(dG, tile_size)
                )
                mask = (sY < tau_luma_eff).to(torch.float32) * (
                    sG < float(tau_grad)
                ).to(torch.float32)
                mask = _upsample_mask(mask, H, W, mode="nearest")
            else:
                mask = (dY < tau_luma_eff).to(torch.float32) * (
                    dG < float(tau_grad)
                ).to(torch.float32)

            mask = _dilate(mask, int(dilate))
            if feather_sigma > 0:
                mask = (
                    _blur_nhwc(mask.to("cpu"), float(feather_sigma))
                    .to(dev)
                    .clamp_(0.0, 1.0)
                )

            if edge_band:
                D = (y_f - y_r[i : i + 1]).abs()
                high = (D > float(tau_edge_high)).to(torch.float32)
                low = (D < float(tau_edge_low)).to(torch.float32)
                band = _dilate(high, int(band_radius)) * low
                if feather_sigma > 0:
                    band = (
                        _blur_nhwc(band.to("cpu"), float(feather_sigma))
                        .to(dev)
                        .clamp_(0.0, 1.0)
                    )
                mask = (mask * (1.0 - band)).clamp_(0.0, 1.0)

            if apply == "all":
                composed_lin = mask * r_lin[i : i + 1] + (1.0 - mask) * f
                composed_lin = composed_lin.to("cpu")
            else:
                f_cpu = f.to("cpu")
                LF_f = _blur_nhwc(f_cpu, 13.0)
                HF_f = f_cpu - LF_f
                LF_mix = (
                    mask.to("cpu") * LF_r[i : i + 1] + (1.0 - mask.to("cpu")) * LF_f
                )
                composed_lin = (HF_f + LF_mix).clamp(0.0, 1.0)

            out = _to_srgb(composed_lin) if linearize else composed_lin
            mvis = mask.to("cpu").repeat(1, 1, 1, 3).clamp_(0.0, 1.0)

            out_frames.append(out.clamp(0, 1))
            mask_frames.append(mvis)

            if dev.type == "cuda" and int(gpu_clear_every) > 0:
                clear_ctr += 1
                if clear_ctr >= int(gpu_clear_every):
                    torch.cuda.empty_cache()
                    clear_ctr = 0

        y_out = torch.cat(out_frames, dim=0)
        mask_preview = torch.cat(mask_frames, dim=0)
        return (y_out, mask_preview)


class BlackSpotCleaner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": (
                    "IMAGE",
                    {"tooltip": "Your clip (frames×H×W×C, values 0–1)."},
                ),
                "linearize": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Work in linear color for cleaner detection.",
                    },
                ),
                "detector": (
                    ["blackhat", "local_floor"],
                    {
                        "default": "blackhat",
                        "tooltip": "blackhat: tiny dark specks • local_floor: larger soft blotches.",
                    },
                ),
                "radius": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 31,
                        "tooltip": "Approximate spot size (pixels). Increase for bigger blotches.",
                    },
                ),
                "tau_blackhat": (
                    "FLOAT",
                    {
                        "default": 4.0 / 255.0,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.0005,
                        "tooltip": "Base sensitivity (0–1). Lower = fix more, higher = fix less.",
                    },
                ),
                "auto_blackhat": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Auto-tune sensitivity from image noise (robust to lighting/texture).",
                    },
                ),
                "bh_k": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.5,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": "Auto strength multiplier. Higher = more aggressive fixes.",
                    },
                ),
                "temporal_gate": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Only fix if darker than neighboring frames (reduces false positives).",
                    },
                ),
                "temporal_radius": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 3,
                        "tooltip": "How many neighbor frames to compare on each side.",
                    },
                ),
                "grad_guard": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Skip fixes on strong edges/text to avoid halos.",
                    },
                ),
                "tau_grad_edge": (
                    "FLOAT",
                    {
                        "default": 0.07,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Edge strength where fixes are skipped (higher = skip more).",
                    },
                ),
                "dilate": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 8,
                        "tooltip": "Expand the fix mask (pixels).",
                    },
                ),
                "feather_sigma": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 16.0,
                        "step": 0.5,
                        "tooltip": "Soften mask edges (pixels).",
                    },
                ),
                "process_on": (
                    ["auto", "cpu", "gpu"],
                    {
                        "default": "auto",
                        "tooltip": "Choose CPU/GPU. Auto switches to GPU on very large frames.",
                    },
                ),
                "gpu_clear_every": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "tooltip": "If >0 and using GPU, free memory every N frames.",
                    },
                ),
            },
            "optional": {
                "reference": (
                    "IMAGE",
                    {
                        "tooltip": "Optional external reference floor (1×H×W×C). Resized if needed."
                    },
                ),
                "ref_source": (
                    ["none", "external", "batch_index"],
                    {
                        "default": "none",
                        "tooltip": "Choose a floor: none, an external image, or a frame index from this clip.",
                    },
                ),
                "ref_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999999,
                        "tooltip": "If using a frame index as the floor, which one to use.",
                    },
                ),
                "tau_down": (
                    "FLOAT",
                    {
                        "default": 2.0 / 255.0,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.0005,
                        "tooltip": "Only lift where the frame is at least this much darker than the floor.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "mask_preview")
    FUNCTION = "clean"
    CATEGORY = "video utils"
    DESCRIPTION = (
        "Removes tiny dark specks and soft blotches by gently lifting only the dark outliers—keeps edges and details safe with guards.\n\n"
        "More from me!: https://artificialsweetener.ai"
    )

    @torch.no_grad()
    def clean(
        self,
        frames,
        linearize=True,
        detector="blackhat",
        radius=5,
        tau_blackhat=4.0 / 255.0,
        auto_blackhat=True,
        bh_k=3.0,
        temporal_gate=True,
        temporal_radius=1,
        grad_guard=True,
        tau_grad_edge=0.07,
        dilate=1,
        feather_sigma=1.5,
        process_on="auto",
        gpu_clear_every=0,
        reference=None,
        ref_source="none",
        ref_index=0,
        tau_down=2.0 / 255.0,
    ):
        x = frames if isinstance(frames, torch.Tensor) else torch.tensor(frames)
        B, H, W, C = x.shape

        ref = None
        if str(ref_source) == "external" and reference is not None:
            ref = (
                reference
                if isinstance(reference, torch.Tensor)
                else torch.tensor(reference)
            )
            if ref.shape[1] != H or ref.shape[2] != W:
                ref = _resize_lanczos(ref[:1].to("cpu"), H, W)
            ref = ref[:1].repeat(B, 1, 1, 1)
        elif str(ref_source) == "batch_index":
            idx = max(0, min(int(ref_index), B - 1))
            ref = x[idx : idx + 1].repeat(B, 1, 1, 1)

        xx = _to_lin(x) if linearize else x
        rr = _to_lin(ref) if (ref is not None and linearize) else ref

        want_gpu = (process_on == "gpu") or (
            process_on == "auto" and torch.cuda.is_available() and (H * W >= 6_000_000)
        )
        dev = torch.device("cuda") if want_gpu else torch.device("cpu")

        y = _luma(xx).to(dev)
        g = _sobel_mag(y)
        if rr is not None:
            y_ref = _luma(rr).to(device=y.device, dtype=y.dtype)  # match y
            assert (
                y_ref.shape[0] == y.shape[0]
            ), f"y_ref B={y_ref.shape[0]} vs y B={y.shape[0]}"
            assert (
                y_ref.shape[1:3] == y.shape[1:3]
            ), f"spatial mismatch {y_ref.shape[1:3]} vs {y.shape[1:3]}"
            floor = (y_ref - y) > float(tau_down)

        r = int(radius)
        if detector == "blackhat":
            k = max(1, 2 * r + 1)
            k = min(k, 2 * min(H, W) - 1)
            t = y.permute(0, 3, 1, 2)
            d = F.max_pool2d(
                F.pad(t, (k // 2, k // 2, k // 2, k // 2), mode="replicate"),
                kernel_size=k,
                stride=1,
            )
            e = -F.max_pool2d(
                F.pad(-d, (k // 2, k // 2, k // 2, k // 2), mode="replicate"),
                kernel_size=k,
                stride=1,
            )
            y_close = e.permute(0, 2, 3, 1)
            score = (y_close - y).clamp_min(0)
        else:
            sigma = max(0.5, r / 2.0)
            Bsm = _blur_nhwc(y.to("cpu"), sigma).to(y.device)
            score = (Bsm - y).clamp_min(0)
        tau = float(tau_blackhat)
        if bool(auto_blackhat):
            region = (g < float(tau_grad_edge)).to(torch.float32)
            if region.sum() < 1:
                region = torch.ones_like(region)
            sel = score[region > 0.5].view(-1)
            if sel.numel() > 0:
                med = torch.median(sel)
                sigma_bh = 1.4826 * torch.median((sel - med).abs())
                tau = max(tau, float(bh_k) * float(sigma_bh))

        mask = (score > tau).to(torch.float32)

        if rr is not None:
            floor = (y_ref - y) > float(tau_down)
            mask = torch.maximum(mask, floor.to(torch.float32))

        if temporal_gate and B > 1:
            idxs = []
            for dt in range(1, int(temporal_radius) + 1):
                if dt < B:
                    idxs += [
                        torch.clamp(torch.arange(B) - dt, 0, B - 1),
                        torch.clamp(torch.arange(B) + dt, 0, B - 1),
                    ]
            neigh = torch.stack([y[i] for i in torch.stack(idxs, dim=0)], dim=0)
            y_med = torch.median(neigh, dim=0).values
            mask = mask * ((y_med - y) > tau).to(torch.float32)

        if grad_guard:
            guard = (g < float(tau_grad_edge)).to(torch.float32)
            mask = mask * guard

        mask = _dilate(mask, int(dilate))
        if feather_sigma > 0:
            mask = (
                _blur_nhwc(mask.to("cpu"), float(feather_sigma))
                .to(dev)
                .clamp_(0.0, 1.0)
            )

        delta = score * mask
        delta3 = delta.repeat(1, 1, 1, 3)
        out_lin = (xx.to(dev) + delta3).clamp(0.0, 1.0)
        if dev.type == "cuda":
            out_lin = out_lin.to("cpu")

        out = _to_srgb(out_lin) if linearize else out_lin
        mask_preview = mask.to("cpu").repeat(1, 1, 1, 3).clamp_(0.0, 1.0)

        if dev.type == "cuda" and int(gpu_clear_every) > 0:
            torch.cuda.empty_cache()

        return (out.clamp(0, 1), mask_preview)
