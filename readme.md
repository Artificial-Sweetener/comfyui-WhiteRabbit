# WhiteRabbit: Master the Flow of Time 🐇

This is **comfyui-WhiteRabbit**, a nodepack designed to help you work with video from within ComfyUI. The Rabbit's specialty is looping through time to help you create seamless looping video, but that's not all she brings to the tea party. Quality, arbitrary framerate resampling and super fast image resizing are also part of the kit!

While some of these nodes certainly can be used for single-image tasks, every one of them is designed with efficient **batch handling** in mind and that means the performance gains compound, letting you process whole video clips as fast as possible within your hardware constraints.


## Installation

This pack requires an external dependency: **[ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation)**. The provided `install.py` script will automatically find your `custom_nodes` folder and handle everything for you.

**Install in two steps:**

1. Drop the **comfyui-WhiteRabbit** folder into your `ComfyUI/custom_nodes/` directory.
2. From your terminal, run:

```bash
python install.py
```

## The Nodes

This pack of nodes helps you solve some of the trickiest problems in video creation.

### Time Benders

These nodes bend time itself to add or remove frames, all powered by the **RIFE** interpolation model. For a slight speed boost, they’re optimized to work together, caching the RIFE model for small efficiency gains in multi‑RIFE workflows.

- **RIFE VFI Interpolate by Multiple**: The basic tool for frame interpolation. Multiply your frames by 2×, 4×, etc., and it’ll generate the new frames needed to make your video silky smooth.
- **RIFE VFI FPS Resample**: A master of time travel. Convert your video to a specific target frame rate, automatically handling both adding and dropping frames as needed. Includes features to prevent common artifacts like flicker for a clean result.
- **RIFE VFI Custom Timing**: Ready for total control? Place every new frame with surgical precision. Create custom speed ramps or smooth out specific moments by providing a custom timing list.
- **RIFE Seam Timing Analyzer**: The perfect companion to the custom timing node. Automatically calculates the exact timing for a seamless loop, giving you the CSV values you need to make your transition feel flawless.

![resample_framerate](examples/resample_framerate.png)
> *Example:* The **RIFE VFI FPS Resample** node is a master of time, resampling your video to a new frame rate. Try for yourself; the workflow is attached!

### Loop Masters

Making a seamless video loop can feel like a riddle. These nodes give you the keys to the perfect, continuous loop.

- **Prepare Loop Frames**: The first step. This node takes your entire video and prepares the loop "seam" by isolating the last and first frames into a separate batch. This little pair is all your interpolator needs to get started on the transition.
- **Assemble Loop Frames**: The final piece. After your interpolator works its magic, this node takes your original video and appends the new seam frames to the end, assembling your complete, continuous loop.
- **Autocrop to Loop**: Don't get lost in the forest of frames! This clever node intelligently analyzes your video to find the best possible place to crop from the end, ensuring your loop flows as smoothly as can be.
- **Trim Batch Ends**: A simple tool for trimming a fixed number of frames from the beginning or end of your clip, perfect for removing unwanted intros or outros.

![interpolate_loop_seam](examples/interpolate_loop_seam.png)
> *Example:* Stitch a seamless loop with **Prepare Loop Frames** ➜ your interpolator ➜ **Assemble Loop Frames**. You can drop this png into ComfyUI and take it for a test drive!

![interpolate_loop_seam](examples/autocrop_to_loop.png)
> *Example:* The best loop is the one you already have. **Autocrop to Loop** can help you find the best end frame by analyzing the visual difference and timing between trailing frames in your clip.

### Post-Processing

These nodes play support!

- **Batch Resize w/ Lanczos**: Fast, principled, and uncompromising in quality. This CUDA‑accelerated node resizes a batch of images (or your single images, of course) using the high‑quality Lanczos algorithm. written for PyTorch; [TorchLanc](https://github.com/Artificial-Sweetener/TorchLanc) It’s dramatically faster than CPU alternatives like Pillow's own Lanczos, with potential for up to a *10× speed increase*.
- **Upscale w/ Model (Advanced)**: A version of ComfyUI's own "Upscale Image (Using Model)" but with direct controls exposed for batch size and tiling which can help speed up scaling dramatically if you tune the numbers to your system.
- **Pixel Hold**: Can be used to reduce video flicker and clean up static parts of a video by reducing small fluctuations caused by video diffusion or compression. There is the potential to use this creatively because it can also take an input image as its baseline.
- **Watermark**: For single images or batches. Very quick, especially when compared to doing the same task in pro editing tools.

![resize](examples/resize.png)
> *Example:* Resize images quickly with **Batch Resize w/ Lanczos**. Workflow attached!!

![resize_with_model_to_target](examples/resize_with_model_to_target.png)
> *Example:* Use **Upscale w/ Model (Advanced)** in concert with **Batch Resize w/ Lanczos** to reach a specific target size like so. The image is holding onto the workflow for you.

![watermark](examples/watermark.png)
> *Example:* Apply a watermark to each frame rapidly with smart configuration options. Workflow included.

## License & Acknowledgements
- **Project License:** GNU Affero General Public License v3.0 (**AGPL‑3.0**). Please read the full [LICENSE](LICENSE) included with this repo! The AGPL-3.0 is a strong copyleft license. If you convey the software, you must provide its corresponding source; and if you let users interact with a modified version over a network, you must offer them that modified version’s corresponding source.

- **Dependency License:** Portions of this code (contained within [interpolation.py](interpolation.py)) are adapted from **[ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) by [**Fannovel16**](https://github.com/Fannovel16_) and [other contributors](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/graphs/contributors), which is licensed under the **MIT License**. A full copy of that license can be found [here](LICENSES/MIT-ComfyUI-Frame-Interpolation.txt).

- UI for **Batch Resize w/ Lanczos** was inspired by the similar node from [Kijai](https://github.com/kijai/)'s excellent [KJNodes](thub.com/kijai/ComfyUI-KJNodes).

---

## From the Developer ❤️

I hope you love using these nodes as much as I loved putting them together!

- **Buy Me a Coffee**: You can help fuel more projects like this at my [Ko-fi page](https://ko-fi.com/artificial_sweetener).
- **My Website & Socials**: See my art, poetry, and other dev updates at [artificialsweetener.ai](https://artificialsweetener.ai).
- **If you like this project**, it would mean a lot to me if you gave me a star here on Github!! ⭐
