# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2025 ArtificialSweetener <artificialsweetenerai@proton.me>

import torch


class PrepareLoopFrames:
    DESCRIPTION = (
        "Prepares the wrap seam: builds a tiny 2-frame batch [last, first] for your interpolator and also passes the original clip through unchanged."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Your clip as an IMAGE batch (frames×H×W×C, values 0–1). Outputs: [last, first] for the seam, plus the original clip."
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("interp_batch", "original_images")
    FUNCTION = "prepare"
    CATEGORY = "video utils"

    def prepare(self, images):
        last_frame = images[-1:]
        first_frame = images[0:1]
        interp_batch = torch.cat((last_frame, first_frame), dim=0)
        return (interp_batch, images)


class AssembleLoopFrames:
    DESCRIPTION = (
        "Builds the final loop: appends only the new in-between seam frames to your original clip—no duplicate of frame 1."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": (
                    "IMAGE",
                    {"tooltip": "Your original clip (frames×H×W×C)."},
                ),
                "interpolated_frames": (
                    "IMAGE",
                    {
                        "tooltip": "Frames that bridge last→first. The first and last of this batch are the originals; only the middle ones get added."
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "assemble"
    CATEGORY = "video utils"

    def assemble(self, original_images, interpolated_frames):
        original_images = original_images.to(interpolated_frames.device)
        in_between = interpolated_frames[1:-1]
        out = torch.cat((original_images, in_between), dim=0)
        return (out,)


class RollFrames:
    DESCRIPTION = (
        "Rolls the clip in a loop by an integer amount (cyclic shift). Also returns the same offset so you can undo it later."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Your clip (frames×H×W×C)."}),
                "offset": (
                    "INT",
                    {
                        "default": 1,
                        "min": -9999,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "How far to rotate the clip. Positive = forward in time; negative = backward.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "offset_out")
    FUNCTION = "roll"
    CATEGORY = "video utils"

    def roll(self, images, offset):
        B = images.shape[0]
        if B == 0:
            return (images, int(offset))
        k = int(offset) % B
        if k == 0:
            return (images, int(offset))
        rolled = torch.roll(images, shifts=-k, dims=0)  # +1 → [2,3,...,1]
        return (rolled, int(offset))


class UnrollFrames:
    DESCRIPTION = (
        "Undo a previous roll after interpolation by accounting for the inserted frames (rotate by base_offset × (m+1))."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {"tooltip": "Clip after interpolation (frames′×H×W×C)."},
                ),
                "base_offset": (
                    "INT",
                    {
                        "default": 1,
                        "min": -9999,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "Use the exact offset_out that came from RollFrames.",
                    },
                ),
                "m": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "How many in-betweens per gap were added (the interpolation multiple).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "unroll"
    CATEGORY = "video utils"

    def unroll(self, images, base_offset, m):
        Bp = images.shape[0]
        if Bp == 0:
            return (images,)
        eff = (int(base_offset) * (int(m) + 1)) % Bp
        return (torch.roll(images, shifts=+eff, dims=0),)


class AutocropToLoop:
    """
    Finds a natural loop by cropping frames from the END of the batch.
    Returns the cropped clip that makes the seam (last_kept -> first)
    feel like a normal step between real neighbors.

    Score = weighted mix of:
      - step-size match (L1/MSE distance)
      - similarity match (SSIM)
      - exposure continuity (luma)
      - motion consistency (optical flow; optional)

    Speed: can run metrics on GPU and use mixed precision for SSIM/conv math.
    Progress bar: one tick per candidate crop (0..max_end_crop_frames).
    """

    DESCRIPTION = (
        "Auto-crops the clip to create a smoother loop: tests crops from the end and scores the seam so it feels like a normal step."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_frames": (
                    "IMAGE",
                    {
                        "tooltip": "Your full clip (NHWC, 0–1). Tries every crop from 0..max_end_crop_frames and returns the best loop."
                    },
                ),
                "max_end_crop_frames": (
                    "INT",
                    {
                        "default": 12,
                        "min": 0,
                        "max": 10000,
                        "tooltip": "Largest crop to test at the END. Higher = more candidates (slower), but potentially better.",
                    },
                ),
                "include_first_step": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use the first neighbor pair (frame 0→1) as a target step size/similarity.",
                    },
                ),
                "include_last_step": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use the last neighbor pair inside the KEPT region as a target.",
                    },
                ),
                "include_global_median_step": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Also use the median step across the KEPT region (needs ≥3 frames). Helps ignore outliers.",
                    },
                ),
                "seam_window_frames": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 6,
                        "tooltip": "Average over multiple aligned pairs across the seam. Larger = more robust.",
                    },
                ),
                "distance_metric": (
                    ["L1", "MSE"],
                    {
                        "default": "L1",
                        "tooltip": "How to measure step size for matching. L1 is usually more forgiving; MSE penalizes big errors more.",
                    },
                ),
                "score_in_8bit": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Score with an 8-bit view (simulate export). Output video still stays float.",
                    },
                ),
                "use_ssim_similarity": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Include SSIM so the seam ‘looks’ like a normal neighbor—avoid freeze or jump.",
                    },
                ),
                "use_exposure_guard": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Promote smooth brightness across the seam (reduces flicker pops).",
                    },
                ),
                "use_flow_guard": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Encourage consistent motion across the seam (needs OpenCV; slower).",
                    },
                ),
                "weight_step_size": (
                    "FLOAT",
                    {
                        "default": 0.55,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Importance of matching step size. Higher = less freeze/jump risk.",
                    },
                ),
                "weight_similarity": (
                    "FLOAT",
                    {
                        "default": 0.30,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Importance of visual similarity (SSIM). Helps avoid a frozen-looking seam.",
                    },
                ),
                "weight_exposure": (
                    "FLOAT",
                    {
                        "default": 0.10,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Importance of even brightness across the seam.",
                    },
                ),
                "weight_flow": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Importance of motion continuity across the seam.",
                    },
                ),
                "ssim_downsample_scales": (
                    "STRING",
                    {
                        "default": "1,2",
                        "tooltip": "SSIM scales to average, as a comma list. Example: 1,2 = full-res and half-res.",
                    },
                ),
                "accelerate_with_gpu": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If ON and CUDA is available, run scoring on GPU for a big speedup (same results).",
                    },
                ),
                "use_mixed_precision": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If ON (with GPU), use mixed precision for SSIM/conv math (faster on larger clips).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "cropped_clip",
        "end_crop_frames",
        "cropped_length",
        "score",
        "diagnostics_csv",
    )
    FUNCTION = "find_and_crop"
    CATEGORY = "video utils"

    _gw_cache = {}  # gaussian window cache

    def _to_nchw(self, x):
        import torch

        if x.ndim == 4 and x.shape[-1] in (1, 3, 4):
            return x.permute(0, 3, 1, 2).contiguous()
        return x

    def _parse_scales(self, csv):
        scales = []
        for s in str(csv).split(","):
            s = s.strip()
            if not s:
                continue
            try:
                v = int(s)
                if v >= 1 and v not in scales:
                    scales.append(v)
            except Exception:
                pass
        return scales or [1]

    def _downsample(self, x, s):
        import torch.nn.functional as F

        if s == 1:
            return x
        H, W = x.shape[-2:]
        newH = max(1, H // s)
        newW = max(1, W // s)
        return F.interpolate(x, size=(newH, newW), mode="area", align_corners=None)

    def _dist(self, A, B, kind="L1"):
        if kind == "MSE":
            return ((A - B) ** 2).mean(dim=(1, 2, 3))
        return (A - B).abs().mean(dim=(1, 2, 3))

    def _luma(self, x_nchw):
        if x_nchw.shape[1] == 1:
            return x_nchw[:, 0:1]
        R = x_nchw[:, 0:1]
        G = x_nchw[:, 1:2]
        B = x_nchw[:, 2:3]
        return 0.2126 * R + 0.7152 * G + 0.0722 * B

    def _gaussian_window(self, C, k=7, sigma=1.5, device="cpu", dtype=None):
        import torch

        key = (int(C), int(k), float(sigma), str(device), str(dtype))
        w = self._gw_cache.get(key)
        if w is not None:
            return w
        ax = torch.arange(k, dtype=dtype, device=device) - (k - 1) / 2.0
        gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
        kernel1d = (gauss / gauss.sum()).unsqueeze(1)
        kernel2d = kernel1d @ kernel1d.t()
        w = kernel2d.expand(C, 1, k, k).contiguous()
        self._gw_cache[key] = w
        return w

    def _ssim_pair_batched(self, x, y, k=7, sigma=1.5, C1=0.01**2, C2=0.03**2):
        import torch
        import torch.nn.functional as F

        C = x.shape[1]
        w = self._gaussian_window(C, k=k, sigma=sigma, device=x.device, dtype=x.dtype)
        mu_x = F.conv2d(x, w, padding=k // 2, groups=C)
        mu_y = F.conv2d(y, w, padding=k // 2, groups=C)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sigma_x2 = F.conv2d(x * x, w, padding=k // 2, groups=C) - mu_x2
        sigma_y2 = F.conv2d(y * y, w, padding=k // 2, groups=C) - mu_y2
        sigma_xy = F.conv2d(x * y, w, padding=k // 2, groups=C) - mu_xy
        ssim_map = ((2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)) / (
            (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12
        )
        return ssim_map.mean(dim=(1, 2, 3))  # (N,)

    def _ssim_multiscale_batched(self, x, y, scales):
        vecs = []
        for s in scales:
            xs = self._downsample(x, s)
            ys = self._downsample(y, s)
            vecs.append(self._ssim_pair_batched(xs, ys))
        return sum(vecs) / float(len(vecs))  # (N,)

    def _precompute_adjacent_metrics(
        self, clip_nhwc_dev, kind, use_ssim, ds_scales, use_exp, use_flow
    ):
        """
        Returns dict with vectors of length (B-1):
          D_adj (torch), S_adj (torch), E_adj (torch), F_adj (np, CPU)
        All torch tensors are on the same device as clip_nhwc_dev.
        """
        import numpy as np
        import torch

        B = int(clip_nhwc_dev.shape[0])
        N = max(0, B - 1)
        result = {}
        if N == 0:
            result["D_adj"] = torch.empty(0, device=clip_nhwc_dev.device)
            result["S_adj"] = torch.empty(0, device=clip_nhwc_dev.device)
            result["E_adj"] = torch.empty(0, device=clip_nhwc_dev.device)
            result["F_adj"] = np.zeros((0,), dtype="float64")
            return result, self._to_nchw(clip_nhwc_dev)

        x_nchw = self._to_nchw(clip_nhwc_dev)  # B,C,H,W (device)
        X = x_nchw[:-1]
        Y = x_nchw[1:]  # N,C,H,W

        result["D_adj"] = self._dist(X, Y, kind=kind)  # (N,)

        if use_ssim:
            result["S_adj"] = self._ssim_multiscale_batched(X, Y, ds_scales)  # (N,)
        else:
            result["S_adj"] = torch.empty(0, device=x_nchw.device)

        if use_exp:
            Y_luma = self._luma(x_nchw).mean(dim=(1, 2, 3))  # (B,)
            result["E_adj"] = (Y_luma[:-1] - Y_luma[1:]).abs()  # (N,)
        else:
            result["E_adj"] = torch.empty(0, device=x_nchw.device)

        if use_flow:
            F_adj = []
            for i in range(N):
                a = clip_nhwc_dev[i : i + 1].detach().cpu()
                b = clip_nhwc_dev[i + 1 : i + 2].detach().cpu()
                F_adj.append(self._flow_mag_mean(a, b))
            import numpy as np

            result["F_adj"] = np.array(F_adj, dtype="float64")
        else:
            import numpy as np

            result["F_adj"] = np.zeros((N,), dtype="float64")

        return result, x_nchw

    def _precompute_seam_tables(self, x_nchw_dev, W, kind, use_ssim, ds_scales):
        """
        For k = 0..W-1, precompute per-frame metrics vs first+k:
          D_to_firstk[k] : (B,) distances to frame k
          S_to_firstk[k] : (B,) SSIM to frame k (if use_ssim)
          E_to_firstk[k] : (B,) |luma(i)-luma(k)|
        Tensors live on x_nchw_dev.device.
        """
        import torch

        B = int(x_nchw_dev.shape[0])
        W = max(1, min(int(W), B - 1))
        D_to_firstk, S_to_firstk, E_to_firstk = [], [], []

        Y = self._luma(x_nchw_dev).mean(dim=(1, 2, 3))

        for k in range(W):
            Bk = x_nchw_dev[k : k + 1].expand_as(x_nchw_dev)
            Dk = self._dist(x_nchw_dev, Bk, kind=kind)
            D_to_firstk.append(Dk)
            if use_ssim:
                Sk = self._ssim_multiscale_batched(x_nchw_dev, Bk, ds_scales)
                S_to_firstk.append(Sk)
            else:
                S_to_firstk.append(torch.empty(0, device=x_nchw_dev.device))
            Ek = (Y - Y[k]).abs()
            E_to_firstk.append(Ek)

        return D_to_firstk, S_to_firstk, E_to_firstk

    def _flow_mag_mean(self, a_nhwc, b_nhwc, max_side=256):
        """
        Mean optical-flow magnitude. Accepts NHWC with/without batch,
        RGB/RGBA/Gray. Soft-fails to 0.0 if OpenCV unavailable.
        """
        try:
            import cv2
            import numpy as np
        except Exception:
            return 0.0

        a = (a_nhwc.detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        b = (b_nhwc.detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        if a.ndim == 4 and a.shape[0] == 1:
            a = a[0]
        if b.ndim == 4 and b.shape[0] == 1:
            b = b[0]

        def to_gray(x: np.ndarray) -> np.ndarray:
            if x.ndim == 2:
                return x
            if x.ndim == 3:
                c = x.shape[-1]
                if c == 1:
                    return x[..., 0]
                if c == 3:
                    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
                if c == 4:
                    return cv2.cvtColor(x, cv2.COLOR_RGBA2GRAY)
                return x.mean(axis=-1).astype(x.dtype)
            x2 = np.squeeze(x)
            if x2.ndim == 2:
                return x2
            if x2.ndim == 3:
                return x2.mean(axis=-1).astype(x2.dtype)
            return None

        a_g, b_g = to_gray(a), to_gray(b)
        if a_g is None or b_g is None or a_g.ndim != 2 or b_g.ndim != 2:
            return 0.0

        H, W = a_g.shape
        scale = max(1.0, max(H, W) / float(max_side))
        if scale > 1.0:
            newW = int(round(W / scale))
            newH = int(round(H / scale))
            a_g = cv2.resize(a_g, (newW, newH), interpolation=cv2.INTER_AREA)
            b_g = cv2.resize(b_g, (newW, newH), interpolation=cv2.INTER_AREA)

        try:
            flow = cv2.calcOpticalFlowFarneback(
                a_g, b_g, None, 0.5, 3, 21, 3, 5, 1.1, 0
            )
            mag = (flow[..., 0] ** 2 + flow[..., 1] ** 2) ** 0.5
            return float(mag.mean())
        except Exception:
            return 0.0

    def find_and_crop(
        self,
        clip_frames,
        max_end_crop_frames,
        include_first_step,
        include_last_step,
        include_global_median_step,
        seam_window_frames,
        distance_metric,
        score_in_8bit,
        use_ssim_similarity,
        use_exposure_guard,
        use_flow_guard,
        weight_step_size,
        weight_similarity,
        weight_exposure,
        weight_flow,
        ssim_downsample_scales,
        accelerate_with_gpu,
        use_mixed_precision,
    ):
        import contextlib

        import numpy as np
        import torch
        from comfy.utils import ProgressBar

        clip_out = clip_frames
        clip_eval = (
            (clip_frames * 255.0).round().clamp(0, 255) / 255.0
            if score_in_8bit
            else clip_frames
        )

        B = int(clip_eval.shape[0])
        if B < 2:
            header = "end_crop,score,D_seam,D_target,S_seam,S_target,E_seam,E_target,F_seam,F_target"
            return (clip_out, 0, B, 0.0, header)

        dev = "cuda" if accelerate_with_gpu and torch.cuda.is_available() else "cpu"
        amp_ctx = (
            torch.cuda.amp.autocast
            if (dev == "cuda" and use_mixed_precision)
            else contextlib.nullcontext
        )

        ds_scales = self._parse_scales(ssim_downsample_scales)
        kind = distance_metric
        W = int(seam_window_frames)
        total_candidates = int(max(0, max_end_crop_frames)) + 1

        with torch.no_grad():
            with amp_ctx():

                clip_eval_dev = clip_eval.to(dev, non_blocking=True)

                pre, x_nchw_dev = self._precompute_adjacent_metrics(
                    clip_nhwc_dev=clip_eval_dev,
                    kind=kind,
                    use_ssim=use_ssim_similarity,
                    ds_scales=ds_scales,
                    use_exp=use_exposure_guard,
                    use_flow=use_flow_guard,
                )
                D_adj, S_adj, E_adj, F_adj = (
                    pre["D_adj"],
                    pre["S_adj"],
                    pre["E_adj"],
                    pre["F_adj"],
                )

                D_seam_tab, S_seam_tab, E_seam_tab = self._precompute_seam_tables(
                    x_nchw_dev=x_nchw_dev,
                    W=W,
                    kind=kind,
                    use_ssim=use_ssim_similarity,
                    ds_scales=ds_scales,
                )

                Y = self._luma(x_nchw_dev).mean(dim=(1, 2, 3))

        best_extra = 0
        best_score = float("inf")
        rows = []
        pbar = ProgressBar(total_candidates)

        for extra in range(0, total_candidates):
            keep = B - extra
            if keep < 2:
                pbar.update(1)
                continue

            last_idx = keep - 1
            W_eff = max(1, min(W, last_idx + 1, B - 1))

            chosen_D = []
            if include_first_step and keep >= 2:
                chosen_D.append(D_adj[0])
            if include_last_step and keep >= 2:
                chosen_D.append(D_adj[last_idx - 1])
            if include_global_median_step and keep >= 3:
                chosen_D.append(D_adj[: keep - 1].median())
            if not chosen_D and keep >= 2:
                chosen_D = [D_adj[0]]
            D_target = float(
                (
                    chosen_D[0]
                    if len(chosen_D) == 1
                    else torch.stack(chosen_D).median()
                ).item()
            )

            if use_ssim_similarity and S_adj.numel() > 0 and keep >= 2:
                chosen_S = []
                if include_first_step:
                    chosen_S.append(S_adj[0])
                if include_last_step:
                    chosen_S.append(S_adj[last_idx - 1])
                if include_global_median_step and keep >= 3:
                    chosen_S.append(S_adj[: keep - 1].median())
                S_target = float(
                    (
                        chosen_S[0]
                        if (chosen_S and len(chosen_S) == 1)
                        else (
                            torch.stack(chosen_S).median()
                            if chosen_S
                            else torch.tensor(0.0, device=S_adj.device)
                        )
                    ).item()
                )
            else:
                S_target = 0.0

            if use_exposure_guard and keep >= 2:
                e_first = (Y[0] - Y[1]).abs()
                e_last = (Y[last_idx] - Y[last_idx - 1]).abs()
                if include_global_median_step and keep >= 3:
                    e_med = (Y[: keep - 1] - Y[1:keep]).abs().median()
                    E_target = float(
                        torch.stack([e_first, e_last, e_med]).median().item()
                    )
                else:
                    E_target = float(torch.stack([e_first, e_last]).median().item())
            else:
                E_target = 0.0

            if use_flow_guard and keep >= 3 and F_adj.size > 0:
                import numpy as np

                F_target = float(np.median(F_adj[: keep - 1]))
            else:
                F_target = 0.0

            idxs = [last_idx - (W_eff - 1 - r) for r in range(W_eff)]
            idxs_t = torch.tensor(idxs, device=x_nchw_dev.device, dtype=torch.long)

            D_vals = torch.stack(
                [
                    D_seam_tab[r].index_select(0, idxs_t[r : r + 1]).squeeze(0)
                    for r in range(W_eff)
                ]
            )
            D_seam = float(D_vals.mean().item())

            if use_ssim_similarity and S_seam_tab[0].numel() > 0:
                S_vals = torch.stack(
                    [
                        S_seam_tab[r].index_select(0, idxs_t[r : r + 1]).squeeze(0)
                        for r in range(W_eff)
                    ]
                )
                S_seam = float(S_vals.mean().item())
            else:
                S_seam = 0.0

            if use_exposure_guard:
                E_vals = torch.stack(
                    [
                        E_seam_tab[r].index_select(0, idxs_t[r : r + 1]).squeeze(0)
                        for r in range(W_eff)
                    ]
                )
                E_seam = float(E_vals.mean().item())
            else:
                E_seam = 0.0

            F_seam = 0.0

            eps = 1e-12
            cost_step = abs(D_seam - D_target) / (D_target + eps)
            cost_sim = (
                abs(S_seam - S_target) / (abs(S_target) + eps)
                if use_ssim_similarity
                else 0.0
            )
            cost_exp = (
                abs(E_seam - E_target) / (E_target + eps)
                if (use_exposure_guard and E_target > 0.0)
                else 0.0
            )
            cost_flow = (
                abs(F_seam - F_target) / (F_target + eps)
                if (use_flow_guard and F_target > 0.0)
                else 0.0
            )

            score = (
                weight_step_size * cost_step
                + weight_similarity * cost_sim
                + weight_exposure * cost_exp
                + weight_flow * cost_flow
            )

            rows.append(
                f"{extra},{score:.6f},{D_seam:.6f},{D_target:.6f},{S_seam:.6f},{S_target:.6f},{E_seam:.6f},{E_target:.6f},{F_seam:.6f},{F_target:.6f}"
            )

            if score < best_score:
                best_score = score
                best_extra = extra

            pbar.update(1)

        final_keep = max(2, B - best_extra)
        cropped = clip_out[0:final_keep]
        header = "end_crop,score,D_seam,D_target,S_seam,S_target,E_seam,E_target,F_seam,F_target"
        diagnostics_csv = header + "\n" + "\n".join(rows) if rows else header
        return (
            cropped,
            int(best_extra),
            int(final_keep),
            float(best_score),
            diagnostics_csv,
        )


class TrimBatchEnds:
    """
    Trim frames from the START and/or END of an IMAGE batch (NHWC, [0..1]).
    Both trims are applied in one pass. Always leaves at least one frame.
    """

    DESCRIPTION = (
        "Quickly remove frames from the start and/or end of a clip. Always keeps at least one frame."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_frames": ("IMAGE", {"tooltip": "Your clip (frames×H×W×C, 0–1)."}),
                "trim_start_frames": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100000,
                        "tooltip": "Frames to remove from the START.",
                    },
                ),
                "trim_end_frames": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100000,
                        "tooltip": "Frames to remove from the END.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "crop"
    CATEGORY = "video utils"

    def crop(self, clip_frames, trim_start_frames, trim_end_frames):
        import torch

        if not isinstance(clip_frames, torch.Tensor) or clip_frames.ndim != 4:
            return (clip_frames,)

        B = int(clip_frames.shape[0])
        if B <= 1:
            return (clip_frames,)

        s = max(0, int(trim_start_frames))
        e = max(0, int(trim_end_frames))

        if s + e >= B:
            s = min(s, B - 1)
            e = max(0, B - s - 1)

        out = clip_frames[s : B - e] if e > 0 else clip_frames[s:]
        if out.shape[0] == 0:
            out = clip_frames[B - 1 : B]
        return (out,)
