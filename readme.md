# ComfyUI-Ouroboros: Video Loop Tools

Two nodes that build clean, seamless loops. You create a bridge between the last and first frame with your interpolator, then assemble the final sequence. Works with any interpolation node (RIFE, GMFSS, RAFT, WAN wrappers).

## What’s inside

- **PrepareLoopFrames**
  Takes an image batch `(B,H,W,C)` and outputs a 2-frame batch `[last_frame, first_frame]`.
  Feed this into your interpolation node to generate the in-between seam frames.

- **AssembleLoopFrames**
  Combines the original frames with the interpolated seam frames, trims duplicated endpoints, then inserts the interpolated frames at the end of the video file to close the loop.

## I/O quick reference

### PrepareLoopFrames
- **Input**: `images` — batch `(B,H,W,C)` in `[0,1]`
- **Outputs**:
  - `interp_batch` — `[last, first]` for your interpolator
  - `original_images` — passthrough of the original batch

### AssembleLoopFrames
- **Inputs**:
  - `original_images` — the original batch `(B,H,W,C)`
  - `interpolated_frames` — frames generated between last → first
    (if your interpolator includes endpoints, this node uses `[1:-1]`)
- **Output**: final looped batch `(B',H,W,C)`

## Typical use

1. Send your frames to **PrepareLoopFrames**.
2. Feed `interp_batch` into your interpolation node and choose how many in-betweens there.
3. Pipe the interpolated result and `original_images` into **AssembleLoopFrames**.
4. Encode the output batch to video or GIF.

## Interpolation backend

Tested with:
https://github.com/Fannovel16/ComfyUI-Frame-Interpolation

Recommended models: **RIFE v4.7** or **RIFE v4.9** (also used in AmazingSeek’s workflow).

## Credit and context

The loop-back interpolation step is based heavily on a workflow by **AmazingSeek**. Their “magic stuff” section achieves the same result inside the Comfy graph using existing nodes. It’s meticulous work and deserves credit.

- Original workflow: https://civitai.com/models/1720535/wan-21-image-to-video-loop-or-workflow?modelVersionId=1948904
- AmazingSeek on CivitAI: https://civitai.com/user/AmazingSeek

## Links

https://artificialsweetener.ai
