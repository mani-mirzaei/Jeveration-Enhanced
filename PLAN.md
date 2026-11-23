# Project Plan: Setup & Enhance `Carol-lyh/Jeveration`

## Context & Goal
I want to clone, optimize, and enhance the `Carol-lyh/Jeveration` repository (a single-forward-pass VLM-based image generation & dense prediction model).
- **Local Hardware:** Laptop with NVIDIA RTX 3050 (6GB VRAM).
- **Cloud Hardware:** Kaggle Notebook (GPU T4 x2, 16GB VRAM each).
- **Goal:** Make the project runnable on low-VRAM GPUs (6GB-15GB), improve output quality, and build a clean production-ready API/UI.

Please execute this plan step-by-step and verify each phase before moving to the next:

## Phase 1: Repository Analysis & Setup
1. Clone `https://github.com/Carol-lyh/Jeveration.git` and inspect the codebase (`jeveration/`, `app.py`, `generate.py`, `image_to_image.py`, `dense_predict.py`, `requirements.txt`).
2. Identify exactly where `Qwen3.5-4B`, `TA-TiTok`, `CLIP`, `decoder512.pt`, and `source_decoder512.pt` are loaded into memory and how `JeverationPipeline` manages CUDA devices.

## Phase 2: Low-VRAM Optimization (High Priority)
Refactor `JeverationPipeline` to add a `low_vram: bool = False` flag (and `--low-vram` CLI argument) that enables:
1. **Lazy Loading:** Only load `decoder512.pt` or `source_decoder512.pt` when the specific task requires it, instead of loading both at startup.
2. **4-bit Quantization:** Use `bitsandbytes` (NF4) to load the `Qwen3.5-4B` backbone in 4-bit precision when `low_vram=True`.
3. **Sequential CPU Offloading:** Move intermediate models (CLIP / Qwen / Decoders) between CPU and GPU stage-by-stage, calling `torch.cuda.empty_cache()` after each stage, so peak VRAM stays under **5.5 GB** (compatible with RTX 3050 6GB).

## Phase 3: Quality & Feature Enhancements
1. **Optional Upscaler:** Integrate a lightweight post-processing upscaler (e.g., Real-ESRGAN x2/x4) as an optional toggle (`--upscale`) to upgrade the native 512x512 output to 1024x1024+.
2. **Task Chaining:** Add a helper method that chains `dense_predict` (depth map extraction) directly into `image_to_image` (`depth` task) in one call.
3. **FastAPI Service:** Create a `server.py` using FastAPI that exposes endpoints for `/text-to-image`, `/image-to-image`, and `/dense-predict` with proper request validation and async queue handling.

## Phase 4: Kaggle Deployment Notebook
Create a ready-to-run `kaggle_runner.ipynb` notebook that:
1. Clones my modified repository and installs dependencies cleanly.
2. Downloads the weights (`linlinlinlinsdada/Jeveration`) to a local directory.
3. Launches the enhanced Gradio app (`app.py --share`) or FastAPI server with `--low-vram` support.

Start with **Phase 1**: analyze the repo structure and show me a brief technical breakdown of how the pipeline loads models before editing any code.