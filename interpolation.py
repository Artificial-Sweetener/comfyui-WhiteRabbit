# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>
#
# Portions adapted from "ComfyUI-Frame-Interpolation" (MIT License)
#   Copyright (c) 2023–2025 Fannovel16 and contributors
#   See LICENSES/MIT-ComfyUI-Frame-Interpolation.txt for the full text.

# Imports for: RIFE_FPS_Resample, RIFE_SeamTimingAnalyzer_Advanced, RIFE_VFI_Advanced
import sys
from pathlib import Path

# Point Python at the sibling base pack
BASE = Path(__file__).resolve().parent.parent / "comfyui-frame-interpolation"
VFI_MODELS = BASE / "vfi_models"

# Prefer external comfyui-frame-interpolation; fall back to vendored /vendor/ inside this repo
p_models = str(VFI_MODELS)  # external: .../custom_nodes/comfyui-frame-interpolation/vfi_models
p_base = str(BASE)          # external: .../custom_nodes/comfyui-frame-interpolation
p_vendor = str(Path(__file__).resolve().parent / "vendor")  # <-- fix: repo-local vendor/

if VFI_MODELS.is_dir() and p_models not in sys.path:
    sys.path.insert(0, p_models)   # exposes external 'rife'
if BASE.is_dir() and p_base not in sys.path:
    sys.path.insert(0, p_base)     # exposes external 'vfi_utils'
if p_vendor not in sys.path:
    sys.path.append(p_vendor)      # fallback: vendored 'rife' and 'vfi_utils'


import math
import re
import typing
from fractions import Fraction

# === Cached RIFE model loader (shared by all nodes) ===
from functools import lru_cache

import torch
import torch.nn.functional as F

# --- Comfy + stdlib used by the three nodes ---
from comfy.model_management import get_torch_device
from comfy.utils import ProgressBar as _ComfyProgressBar
from rife import CKPT_NAME_VER_DICT, MODEL_TYPE

# --- imports from the base pack ---
from vfi_utils import (
    InterpolationStateList,
    generic_frame_loop,
    load_file_from_github_release,
    postprocess_frames,
    preprocess_frames,
)


@lru_cache(maxsize=4)
def _get_rife_model(ckpt_name: str):
    import torch
    from comfy.model_management import get_torch_device
    from rife import CKPT_NAME_VER_DICT, MODEL_TYPE
    from rife.rife_arch import IFNet
    from vfi_utils import load_file_from_github_release

    model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
    arch_ver = CKPT_NAME_VER_DICT[ckpt_name]
    m = IFNet(arch_ver=arch_ver)
    sd = torch.load(model_path, map_location="cpu")
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m.to(get_torch_device()).to(memory_format=torch.channels_last)

# ---------- shared helpers (for all nodes) ----------
def _scale_list(scale_factor: float):
    s = float(scale_factor)
    return [8.0 / s, 4.0 / s, 2.0 / s, 1.0 / s]

def _prep_frames(frames: torch.Tensor) -> torch.Tensor:
    x = preprocess_frames(frames)
    if x.device.type == "cpu" and torch.cuda.is_available():
        x = x.pin_memory()
    return x

def _progress(total_ticks: int):
    return _ComfyProgressBar(int(max(1, total_ticks)))

def _count_synth(frames: torch.Tensor, multiplier: int) -> int:
    N = int(frames.shape[0])
    m = max(0, int(multiplier))
    return max(0, N - 1) * m

def _make_rife_callback():
    def _cb(frame_0, frame_1, timestep, model, scale_list, ensemble, t_mapper):
        if torch.is_tensor(timestep):
            t_scalar = float(timestep.reshape(-1)[0].item())
        else:
            t_scalar = float(timestep)
        t = t_mapper(t_scalar) if t_mapper is not None else t_scalar
        # fast_mode is a no-op for 4.7/4.9; pass False
        return model(frame_0, frame_1, t, scale_list, False, ensemble)
    return _cb

def _with_progress(cb, pbar):
    if pbar is None:
        return cb
    def _wrapped(*args, **kwargs):
        y = cb(*args, **kwargs)
        pbar.update(1)
        return y
    return _wrapped
# ----------------------------------------------------

class RIFE_VFI_Opt:
    DESCRIPTION = (
        "Interpolate a clip by a chosen multiple using RIFE 4.7/4.9 — inserts evenly spaced in-between frames "
        "between every pair (e.g., ×2 adds 1 frame per pair).\n\n"
        "• Built on the ComfyUI-Frame-Interpolation base pack (models & utilities).\n"
        "• Credits: based on and requires Fannovel16’s ComfyUI-Frame-Interpolation (and its contributors). "
        "Please visit the repo for full-stack credits and updates: https://github.com/Fannovel16/ComfyUI-Frame-Interpolation\n\n"
        "More from me!: https://artificialsweetener.ai"
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (
                    ["rife47.pth", "rife49.pth"],
                    {"default": "rife47.pth", "tooltip": "Choose the RIFE 4.7 or 4.9 model file from the base pack."}
                ),
                "frames": (
                    "IMAGE",
                    {"tooltip": "Input clip: one image per frame (NHWC)."}
                ),
                "multiplier": (
                    "INT",
                    {"default": 2, "min": 1, "tooltip": "How many frames to add between each pair. 2 → add 1, 4 → add 3."}
                ),
                "scale_factor": (
                    [0.25, 0.5, 1.0, 2.0, 4.0],
                    {"default": 1.0, "tooltip": "Quality/speed trade-off. 1.0 recommended. Lower = faster/softer; higher = sharper/slower."}
                ),
                "ensemble": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Blend forward & backward predictions to reduce artifacts (slower)."}
                ),
                "clear_cache_after_n_frames": (
                    "INT",
                    {"default": 10, "min": 0, "max": 1000, "tooltip": "Advanced: free GPU memory every N generated frames. 0 = never."}
                ),
            },
            "optional": {
                "optional_interpolation_states": (
                    "INTERPOLATION_STATES",
                    {"tooltip": "Skip interpolation for specific frame pairs (e.g., scene cuts). Does not change timing."}
                ),
            },
        }


    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "video utils"

    def vfi(
        self,
        ckpt_name: typing.AnyStr,
        frames: torch.Tensor,
        multiplier: int = 2,
        scale_factor: float = 1.0,
        ensemble: bool = True,
        clear_cache_after_n_frames: int = 10,
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs
    ):
        model = _get_rife_model(ckpt_name)

        frames = _prep_frames(frames)

        m_effective = int(multiplier)
        if m_effective <= 1:
            return (postprocess_frames(frames),)

        scale_list = _scale_list(scale_factor)
        total_synth = _count_synth(frames, m_effective)
        pbar = _progress(total_synth)
        cb = _make_rife_callback()
        cbp = _with_progress(cb, pbar)

        args = [model, scale_list, ensemble, None]  # t_mapper=None → identity

        with torch.inference_mode():
            out = postprocess_frames(
                generic_frame_loop(
                    type(self).__name__,
                    frames,
                    clear_cache_after_n_frames,
                    m_effective,
                    cbp,
                    *args,
                    interpolation_states=optional_interpolation_states,
                    dtype=torch.float32,
                )
            )
        return (out,)

class RIFE_VFI_Advanced:
    """
    Advanced RIFE node exposing timestep controls (by multiple, custom t-schedules).
    """
    DESCRIPTION = (
        "Custom timing for RIFE 4.7/4.9 — still “interpolate by multiple,” but you control where the in-betweens land "
        "(ease in/out, clamps, or your own t-list).\n\n"
        "• Built on the ComfyUI-Frame-Interpolation base pack (models & utilities).\n"
        "• Credits: based on and requires Fannovel16’s ComfyUI-Frame-Interpolation (and its contributors). "
        "Please visit the repo for full-stack credits and updates: https://github.com/Fannovel16/ComfyUI-Frame-Interpolation\n\n"
        "More from me!: https://artificialsweetener.ai"
    )
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (
                    ["rife47.pth", "rife49.pth"],
                    {"default": "rife47.pth", "tooltip": "Choose the RIFE 4.7 or 4.9 model file from the base pack."}
                ),
                "frames": (
                    "IMAGE",
                    {"tooltip": "Input clip: one image per frame (NHWC)."}
                ),
                "multiplier": (
                    "INT",
                    {"default": 2, "min": 0, "tooltip": "Number of in-betweens per pair. 0 = passthrough (no synthesis)."}
                ),
                "t_mode": (
                    ["linear", "gamma_in", "gamma_out", "gamma_in_out", "bounded_linear", "custom_list"],
                    {"default": "linear", "tooltip": "How to place the in-betweens over time: straight line, ease in/out, clamp range, or custom list."}
                ),
                "t_gamma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.05, "max": 10.0, "step": 0.05, "tooltip": "Strength of the ease curve for gamma modes."}
                ),
                "t_min": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Lower clamp for t (use with bounded_linear)."}
                ),
                "t_max": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Upper clamp for t (use with bounded_linear)."}
                ),
                "scale_factor": (
                    [0.25, 0.5, 1.0, 2.0, 4.0],
                    {"default": 1.0, "tooltip": "Quality/speed trade-off. 1.0 recommended. Lower = faster/softer; higher = sharper/slower."}
                ),
                "ensemble": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Blend forward & backward predictions to reduce artifacts (slower)."}
                ),
                "clear_cache_after_n_frames": (
                    "INT",
                    {"default": 10, "min": 0, "max": 1000, "tooltip": "Advanced: free GPU memory every N generated frames. 0 = never."}
                ),
            },
            "optional": {
                "custom_t_list_csv": (
                    "STRING",
                    {"default": "", "tooltip": "Exact t values (0–1) separated by commas, e.g. 0.18,0.41,0.66. Overrides the schedule."}
                ),
                "optional_interpolation_states": (
                    "INTERPOLATION_STATES",
                    {"tooltip": "Skip interpolation for specific frame pairs (e.g., scene cuts). Does not change timing."}
                ),
            }
        }


    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi_advanced"
    CATEGORY = "video utils"

    def _map_t(self, t_scalar, t_mode, t_gamma, t_min, t_max, custom_sorted, m_effective):
        t = max(0.0, min(1.0, float(t_scalar)))
        if t_mode == "custom_list" and custom_sorted:
            idx = int(round(t * (m_effective + 1))) - 1
            idx = max(0, min(len(custom_sorted) - 1, idx))
            return float(custom_sorted[idx])

        if t_mode == "gamma_in":
            t = t ** max(1e-6, t_gamma)
        elif t_mode == "gamma_out":
            t = 1.0 - (1.0 - t) ** max(1e-6, t_gamma)
        elif t_mode == "gamma_in_out":
            g = max(1e-6, t_gamma)
            if t < 0.5: t = 0.5 * (2.0 * t) ** g
            else:       t = 1.0 - 0.5 * (2.0 * (1.0 - t)) ** g
        return max(0.0, min(1.0, t_min + (t_max - t_min) * t))

    def vfi_advanced(
        self,
        ckpt_name,
        frames,
        clear_cache_after_n_frames = 10,
        multiplier = 2,
        ensemble = True,
        scale_factor = 1.0,
        t_mode = "linear",
        t_gamma = 1.0,
        t_min = 0.0,
        t_max = 1.0,
        custom_t_list_csv = "",
        optional_interpolation_states = None,
        **kwargs
    ):
        model = _get_rife_model(ckpt_name)
        device = next(model.parameters()).device
        dtype  = next(model.parameters()).dtype

        frames = _prep_frames(frames)

        custom_sorted = None
        if custom_t_list_csv and t_mode == "custom_list":
            toks = [x for x in re.split(r"[,\s]+", custom_t_list_csv.strip()) if x]
            custom_sorted = sorted([max(0.0, min(1.0, float(x))) for x in toks])

        m_effective = int(multiplier)
        if m_effective <= 0:
            return (postprocess_frames(frames),)

        def _adv_t_mapper(t_scalar: float) -> float:
            return self._map_t(t_scalar, t_mode, t_gamma, t_min, t_max, custom_sorted, m_effective)

        scale_list = _scale_list(scale_factor)
        total_synth = _count_synth(frames, m_effective)
        pbar = _progress(total_synth)
        cb = _make_rife_callback()
        cbp = _with_progress(cb, pbar)

        args = [model, scale_list, ensemble, _adv_t_mapper]

        with torch.inference_mode():
            out = postprocess_frames(
                generic_frame_loop(
                    type(self).__name__,
                    frames,
                    clear_cache_after_n_frames,
                    m_effective,
                    cbp,
                    *args,
                    interpolation_states=optional_interpolation_states,
                    dtype=torch.float32
                )
            )
        return (out,)

class RIFE_FPS_Resample:
    DESCRIPTION = (
        "Convert a clip from one FPS to another using RIFE 4.7/4.9. Non-integer changes synthesize in-betweens; "
        "exact integer downscales just decimate. Includes optional stabilizers to reduce flicker and protect edges.\n\n"
        "• Built on the ComfyUI-Frame-Interpolation base pack (models & utilities).\n"
        "• Credits: based on and requires Fannovel16’s ComfyUI-Frame-Interpolation (and its contributors). "
        "Please visit the repo for full-stack credits and updates: https://github.com/Fannovel16/ComfyUI-Frame-Interpolation\n\n"
        "More from me!: https://artificialsweetener.ai"
    )
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    ["rife47.pth", "rife49.pth"],
                    {"default": "rife47.pth", "tooltip": "Choose the RIFE 4.7 or 4.9 model file from the base pack."}
                ),
                "frames": (
                    "IMAGE",
                    {"tooltip": "Input clip: one image per frame (NHWC)."}
                ),
                "fps_in": (
                    "FLOAT",
                    {"default": 24.0, "min": 1e-6, "max": 1000.0, "step": 0.01, "tooltip": "The clip’s current frame rate."}
                ),
                "fps_out": (
                    "FLOAT",
                    {"default": 60.0, "min": 1e-6, "max": 2000.0, "step": 0.01, "tooltip": "The frame rate you want."}
                ),

                "scale_factor": (
                    [0.25, 0.5, 1.0, 2.0, 4.0],
                    {"default": 1.0, "tooltip": "Quality/speed trade-off. 1.0 recommended. Lower = faster/softer; higher = sharper/slower."}
                ),
                "ensemble": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Blend forward & backward predictions to reduce artifacts (slower)."}
                ),

                "linearize": ("BOOLEAN", {"default": False, "tooltip": "Process in linear color for more accurate brightness handling (slower)."}),
                "lf_guardrail": ("BOOLEAN", {"default": False, "tooltip": "Keep the overall brightness/gradients close to the originals to reduce flicker."}),
                "lf_sigma": ("FLOAT", {"default": 13.0, "min": 0.0, "max": 64.0, "step": 0.5, "tooltip": "How strong the low-frequency smoothing is. Higher = smoother changes."}),
                "source_pair_match": ("BOOLEAN", {"default": False, "tooltip": "Match exposure/contrast to the source pair to reduce flicker."}),
                "match_a_cap": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.2, "step": 0.001, "tooltip": "Limit for exposure scale (a)."}),
                "match_b_cap": ("FLOAT", {"default": 2.0/255.0, "min": 0.0, "max": 0.1, "step": 0.0005, "tooltip": "Limit for brightness offset (b)."}),
                "edge_band_lock": ("BOOLEAN", {"default": False, "tooltip": "Protect edges by blending from the nearest source frame in edge regions."}),
                "tau_low": ("FLOAT", {"default": 1.5/255.0, "min": 0.0, "max": 0.25, "step": 0.0005, "tooltip": "Edge detector: lower threshold (finds more edges if smaller)."}),
                "tau_high": ("FLOAT", {"default": 6.0/255.0, "min": 0.0, "max": 0.5, "step": 0.0005, "tooltip": "Edge detector: higher threshold (finds only strong edges if larger)."}),
                "band_radius": ("INT", {"default": 4, "min": 0, "max": 64, "tooltip": "Edge protection width in pixels."}),
                "band_soft_sigma": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 16.0, "step": 0.5, "tooltip": "Feather the edge mask. Higher = softer."}),

                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000, "tooltip": "Advanced: free GPU memory every N output frames. 0 = never."}),
            }
        }


    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resample"
    CATEGORY = "video utils"

    @staticmethod
    def _srgb_to_linear(x):
        import torch
        return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp(min=0) ** 2.4)

    @staticmethod
    def _linear_to_srgb(x):
        import torch
        return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp(min=0) ** (1.0 / 2.4) - 0.055)

    @staticmethod
    def _gaussian_kernel1d(sigma, radius):
        import torch
        if radius <= 0:
            k = torch.tensor([1.0], dtype=torch.float32)
            return k / k.sum()
        xs = torch.arange(-radius, radius + 1, dtype=torch.float32)
        k = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
        return k / k.sum()

    _GK_CACHE = {}  # (float(sigma), int(radius), int(C)) -> (kH, kW)

    @staticmethod
    def _gaussian_blur_nhwc(x, sigma):
        if sigma <= 0:
            return x

        N, H, W, C = x.shape
        max_radius = max(0, min(H, W) // 2 - 1)
        radius = min(int(math.ceil(3.0 * sigma)), max_radius)
        if radius <= 0:
            return x

        key = (float(sigma), int(radius), int(C))
        k = RIFE_FPS_Resample._GK_CACHE.get(key)
        if k is None:
            g = RIFE_FPS_Resample._gaussian_kernel1d(sigma, radius)
            kH = g.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
            kW = g.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
            RIFE_FPS_Resample._GK_CACHE[key] = (kH, kW)
            k = (kH, kW)
        kH, kW = k

        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        x1 = F.pad(x_nchw, (0, 0, radius, radius), mode="reflect")
        x1 = F.conv2d(x1, kH, groups=C)
        x2 = F.pad(x1, (radius, radius, 0, 0), mode="reflect")
        x2 = F.conv2d(x2, kW, groups=C)
        return x2.permute(0, 2, 3, 1).contiguous()


    @staticmethod
    def _luma_linear(x_lin):
        import torch
        r, g, b = x_lin[..., 0], x_lin[..., 1], x_lin[..., 2]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    @staticmethod
    def _exposure_match_ab(out_lin, tgt_lin, a_cap, b_cap):
        import torch
        y_o = RIFE_FPS_Resample._luma_linear(out_lin).flatten()
        y_t = RIFE_FPS_Resample._luma_linear(tgt_lin).flatten()
        p = torch.tensor([0.05, 0.5, 0.95], dtype=torch.float32)
        p5_o, p50_o, p95_o = torch.quantile(y_o, p)
        p5_t, p50_t, p95_t = torch.quantile(y_t, p)
        eps = 1e-6
        a = ((p95_t - p5_t) / (p95_o - p5_o + eps)).item()
        a = max(1.0 - a_cap, min(1.0 + a_cap, a))
        b = (p50_t - a * p50_o).item()
        b = max(-b_cap, min(b_cap, b))
        return a, b

    @staticmethod
    def _dilate_mask_nhwc(mask01, radius):
        if radius <= 0:
            return mask01
        x = mask01.permute(0, 3, 1, 2)
        x = F.pad(x, (radius, radius, radius, radius), mode="replicate")
        x = F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1)
        return x.permute(0, 2, 3, 1)

    @torch.inference_mode()
    def resample(
        self,
        ckpt_name,
        fps_in,
        fps_out,
        frames,
        scale_factor=1.0,
        ensemble=True,
        linearize=False,
        lf_guardrail=False,
        lf_sigma=13.0,
        source_pair_match=False,
        match_a_cap=0.02,
        match_b_cap=2.0/255.0,
        edge_band_lock=False,
        tau_low=1.5/255.0,
        tau_high=6.0/255.0,
        band_radius=4,
        band_soft_sigma=2.0,
        clear_cache_after_n_frames=10,
        **kwargs
    ):

        if fps_in <= 0 or fps_out <= 0:
            raise ValueError("fps_in and fps_out must be > 0")

        F_in = Fraction(str(float(fps_in)))
        F_out = Fraction(str(float(fps_out)))
        same_rate = (F_in == F_out)
        ratio = F_in / F_out
        integer_downscale = (ratio.denominator == 1 and ratio.numerator > 1)

        if same_rate:
            return (frames,)
        if integer_downscale:
            step = int(ratio.numerator)
            return (frames[::step].contiguous(),)

        model = _get_rife_model(ckpt_name)
        device = next(model.parameters()).device
        dtype  = next(model.parameters()).dtype

        x_nhwc = _prep_frames(frames)

        n_in = int(x_nhwc.shape[0])
        if n_in <= 1:
            return (postprocess_frames(x_nhwc[:1]),)

        K_max = int(Fraction(n_in - 1) * F_out / F_in)
        n_out = K_max + 1
        scale_list = _scale_list(scale_factor)

        def to_nchw_on_device(idx: int):
            f = x_nhwc[idx:idx + 1].permute(0, 3, 1, 2).contiguous()
            return f.to(device=device, dtype=dtype, non_blocking=True, memory_format=torch.channels_last)

        cache = {}

        def synth(i: int, t_local: float):
            if i not in cache:
                cache[i] = to_nchw_on_device(i)
            if (i + 1) not in cache:
                cache[i + 1] = to_nchw_on_device(i + 1)
            # fast_mode is a no-op for 4.7/4.9; pass False explicitly
            y = model(cache[i], cache[i + 1], float(t_local), scale_list, False, ensemble)
            return y.permute(0, 2, 3, 1).contiguous().detach().cpu().to(torch.float32)

        to_lin = self._srgb_to_linear if linearize else (lambda z: z)
        to_srgb = self._linear_to_srgb if linearize else (lambda z: z)
        blur = self._gaussian_blur_nhwc

        y0 = x_nhwc[0:1]
        H, W, C = int(y0.shape[1]), int(y0.shape[2]), int(y0.shape[3])
        y_buf = torch.empty((n_out, H, W, C), dtype=torch.float32, device='cpu')

        def _write(idx: int, frame_nhwc_cpu: torch.Tensor):
            y_buf[idx:idx+1].copy_(frame_nhwc_cpu)

        num = F_in.numerator * F_out.denominator
        den = F_in.denominator * F_out.numerator
        acc = 0
        frames_since_clear = 0

        pbar = _progress(int(n_out))

        for k in range(n_out):
            pbar.update(1)
            i = acc // den


            frac_num = acc % den

            if i >= (n_in - 1):
                _write(k, x_nhwc[n_in - 1:n_in])
            else:
                if frac_num == 0:
                    _write(k, x_nhwc[i:i + 1])
                else:
                    frac = float(frac_num) / den
                    y = synth(i, frac)

                    if linearize or lf_guardrail or source_pair_match or edge_band_lock:
                        A = x_nhwc[i:i + 1]
                        B = x_nhwc[i + 1:i + 2]
                        y_lin, A_lin, B_lin = to_lin(y), to_lin(A), to_lin(B)

                        if lf_guardrail and lf_sigma > 0:
                            HF = y_lin - blur(y_lin, lf_sigma)
                            LF_target = (1.0 - frac) * blur(A_lin, lf_sigma) + frac * blur(B_lin, lf_sigma)
                            y_lin = HF + LF_target

                        if source_pair_match:
                            tgt = (1.0 - frac) * A_lin + frac * B_lin
                            a, b = self._exposure_match_ab(y_lin, tgt, match_a_cap, match_b_cap)
                            y_lin = (a * y_lin + b).clamp(0.0, 1.0)

                        if edge_band_lock:
                            d = (self._luma_linear(A_lin) - self._luma_linear(B_lin)).abs().unsqueeze(-1)
                            high = (d > tau_high).float()
                            low = (d < tau_low).float()
                            band = self._dilate_mask_nhwc(high, band_radius) * low
                            if band_soft_sigma > 0:
                                band = blur(band, band_soft_sigma).clamp(0.0, 1.0)
                            near = A_lin if frac < 0.5 else B_lin
                            y_lin = band * near + (1.0 - band) * y_lin

                        y = to_srgb(y_lin).clamp(0.0, 1.0)

                    _write(k, y)

            frames_since_clear += 1
            if torch.cuda.is_available() and int(clear_cache_after_n_frames) > 0:
                if frames_since_clear >= int(clear_cache_after_n_frames):
                    torch.cuda.empty_cache()
                    frames_since_clear = 0

            acc += num

        return (postprocess_frames(y_buf),)


class RIFE_SeamTimingAnalyzer:
    """
    Batch-in, timings-out. Chooses t-schedule for the seam [last, first] using
    visual step sizes measured from REAL frames in the input batch.

    Inputs:
      - full_clip (IMAGE, NHWC [N,H,W,C], required): real frames (e.g., 1..10)
      - multiplier (INT): number of in-betweens to synthesize for the seam
      - use_first_two (BOOL): include dist(clip[0], clip[1]) as a target
      - use_last_two  (BOOL): include dist(clip[-2], clip[-1]) as a target
      - use_global_median (BOOL): include median of ALL adjacent deltas (N>=3)
      - ckpt_name / ensemble / scale_factor: RIFE runtime knobs (only used to test t)
      - calibrate_metric: "MSE" or "L1" visual distance
      - calibrate_iters: binary-search iterations
      - t_min / t_max: clamp t search range (keep t_max < 1.0 to avoid hugging frame 1)

    Outputs:
      - t_list_csv (STRING): e.g. "0.183112, 0.408217, 0.653991, 0.913541"
      - multiplier (INT): echo of input multiplier for wiring convenience
    """
    DESCRIPTION = (
        "Finds a smooth loop timing: measures motion in your clip and solves a set of t-values across the wrap "
        "[last→first] so the seam blends naturally.\n\n"
        "• Built on the ComfyUI-Frame-Interpolation base pack (models & utilities).\n"
        "• Credits: based on and requires Fannovel16’s ComfyUI-Frame-Interpolation (and its contributors). "
        "Please visit the repo for full-stack credits and updates: https://github.com/Fannovel16/ComfyUI-Frame-Interpolation\n\n"
        "More from me!: https://artificialsweetener.ai"
    )
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (
                    ["rife47.pth", "rife49.pth"],
                    {
                        "default": "rife47.pth",
                        "tooltip": "Choose the RIFE 4.7 or 4.9 model file (used only to test candidate timings)."
                    },
                ),
                "scale_factor": (
                    [0.25, 0.5, 1.0, 2.0, 4.0],
                    {
                        "default": 1.0,
                        "tooltip": "Quality/speed trade-off for the probe renders. 1.0 recommended."
                    },
                ),
                "ensemble": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Blend forward & backward predictions to reduce artifacts (slower)."
                    },
                ),

                "full_clip": (
                    "IMAGE",
                    {
                        "tooltip": "Your input clip (≥2 frames). We measure real motion here to decide the seam timing."
                    },
                ),
                "multiplier": (
                    "INT",
                    {
                        "default": 4, "min": 0,
                        "tooltip": "How many in-betweens you plan at the seam [last→first]. Set 0 to skip solving."
                    },
                ),

                "use_first_two": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Match the step size between the first two frames in your clip."
                    },
                ),
                "use_last_two": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Match the step size between the last two frames in your clip."
                    },
                ),
                "use_global_median": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use the median step across the whole clip (needs ≥3 frames). Helps ignore outliers."
                    },
                ),

                "calibrate_metric": (
                    ["MSE", "L1"],
                    {
                        "default": "MSE",
                        "tooltip": "How we compare frames while solving: MSE (more sensitive) or L1 (more forgiving)."
                    },
                ),
                "calibrate_iters": (
                    "INT",
                    {
                        "default": 12, "min": 4, "max": 24,
                        "tooltip": "Search depth per solve. Higher = slower, but a tighter match."
                    },
                ),

                "t_min": (
                    "FLOAT",
                    {
                        "default": 0.00, "min": 0.0, "max": 1.0, "step": 0.001,
                        "tooltip": "Lowest allowed t at the seam."
                    },
                ),
                "t_max": (
                    "FLOAT",
                    {
                        "default": 0.96, "min": 0.0, "max": 1.0, "step": 0.001,
                        "tooltip": "Highest allowed t at the seam. Keep <1.0 to avoid hugging the first frame."
                    },
                ),
            }
        }


    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("t_list_csv", "multiplier")
    FUNCTION = "analyze_wrapper"
    CATEGORY = "video utils"

    # --- helpers ---
    def _to_nchw(self, x):
        if x.ndim == 4 and x.shape[-1] in (1, 3, 4):
            return x.permute(0, 3, 1, 2).contiguous()
        return x

    def _dist(self, A, B, kind="MSE"):
        if kind == "L1":
            return (A - B).abs().mean()
        return ((A - B) ** 2).mean()

    def _adjacent_deltas(self, clip_nchw, kind="MSE"):
        N = clip_nchw.shape[0]
        ds = []
        for i in range(N - 1):
            ds.append(self._dist(clip_nchw[i:i+1], clip_nchw[i+1:i+2], kind))
        return ds  # list of scalar tensors

    def analyze_wrapper(self, **kwargs):
        """
        Safe entrypoint for Comfy mapping:
        - If 'multiplier' is omitted: default to 0 (passthrough).
        - If 'multiplier' is invalid/negative: normalize to 0 (passthrough).
        """
        if "multiplier" not in kwargs:
            kwargs["multiplier"] = 0

        try:
            m = int(kwargs["multiplier"])
        except (TypeError, ValueError):
            m = 0
        if m < 0:
            m = 0

        kwargs["multiplier"] = m
        return self.analyze(**kwargs)

    def analyze(self,
                full_clip,
                multiplier,
                use_first_two,
                use_last_two,
                use_global_median,
                ckpt_name,
                ensemble,
                scale_factor,
                calibrate_metric,
                calibrate_iters,
                t_min,
                t_max
            ):

        try:
            m = int(multiplier)
        except (TypeError, ValueError):
            m = 0
        if m < 0:
            m = 0
        if m == 0:
            return ("", 0)

        clip_nhwc = preprocess_frames(full_clip)
        if clip_nhwc.device.type == "cpu" and torch.cuda.is_available():
            clip_nhwc = clip_nhwc.pin_memory()

        N = int(clip_nhwc.shape[0])
        if N < 2:
            raise ValueError("full_clip must contain at least 2 frames.")

        metric = "L1" if calibrate_metric == "L1" else "MSE"
        clip_nchw_cpu = self._to_nchw(clip_nhwc)
        deltas = self._adjacent_deltas(clip_nchw_cpu, metric)

        chosen = []
        if use_first_two:
            chosen.append(deltas[0])
        if use_last_two:
            chosen.append(deltas[-1])
        if use_global_median:
            if N < 3:
                raise ValueError("use_global_median requires full_clip with >= 3 frames.")
            chosen.append(torch.stack(deltas).median())

        if not chosen:
            raise ValueError("Enable at least one of: use_first_two, use_last_two, use_global_median.")

        d_target = (chosen[0] if len(chosen) == 1 else torch.stack(chosen).median()).item()
        iters = int(calibrate_iters)
        m = int(m)

        model = _get_rife_model(ckpt_name)
        device = next(model.parameters()).device
        dtype  = next(model.parameters()).dtype  # usually float32

        last_nchw  = self._to_nchw(clip_nhwc[N-1:N]).to(device=device, dtype=dtype, non_blocking=True, memory_format=torch.channels_last)
        first_nchw = self._to_nchw(clip_nhwc[0:1]).to(device=device, dtype=dtype, non_blocking=True, memory_format=torch.channels_last)

        t_min = float(max(0.0, min(1.0, t_min)))
        t_max = float(max(0.0, min(1.0, t_max)))
        if not (0.0 <= t_min < t_max <= 1.0):
            raise ValueError("Require 0 <= t_min < t_max <= 1.0")

        scale_list = [8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]

        @torch.no_grad()
        def synth_at(t_scalar: float):
            t_scalar = float(max(t_min, min(t_max, t_scalar)))
            return model(last_nchw, first_nchw, t_scalar, scale_list, False, ensemble)

        lo, hi = t_min, t_max
        with torch.inference_mode():
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                d_mid = float(self._dist(synth_at(mid), first_nchw, metric).item())
                if d_mid > d_target:
                    lo = mid
                else:
                    hi = mid
            t_last = 0.5 * (lo + hi)
            f_prev = synth_at(t_last)

            ts = [t_last]
            for _ in range(m - 1):
                lo, hi = t_min, ts[-1]
                for _ in range(iters):
                    mid = 0.5 * (lo + hi)
                    d_mid = float(self._dist(synth_at(mid), f_prev, metric).item())
                    if d_mid > d_target:
                        lo = mid
                    else:
                        hi = mid
                tk = 0.5 * (lo + hi)
                ts.append(tk)
                f_prev = synth_at(tk)

        ts_sorted = sorted(ts)
        csv = ", ".join(f"{t:.6f}" for t in ts_sorted)
        return (csv, m)
