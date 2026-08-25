## Summary — `--qwen38` support for `videsc` (all code changes complete)

All edits from the task are done. Here's what was implemented across the four files:

### 1. `videsc/cli/args.py`
- Added `--qwen38` to the mutually exclusive VL model group (`--omni | --qwen38 | --qwen35 | --gemma4`), with help text noting the default model.
- Added the post-parse default resolution (matching the existing `--qwen35`/`--gemma4` pattern):
  ```python
  if args.qwen38 and args.model == _VL_DEFAULT_MODEL:
      args.model = "Inferact/Qwen3.8-27B-NVFP4"
      args.model_hf = True
  ```
- Updated the `--vl` help text to include Qwen3.8 in the list of supported models.

### 2. `videsc/model/loader.py`
- Added `load_qwen38_model_and_processor(args)` between the Qwen3.5 and Gemma 4 loaders. It follows the established loader conventions exactly:
  - Shared `_SHARED_MODEL`/`_SHARED_PROCESSOR` cache for threaded batch mode.
  - Path resolution via `model_hf` / `model_full` / `model_dir` (so both HF ids and local dirs work).
  - Optional `--processor` override, `--half_cpu` thread limiting, `--quant` (8/4-bit) config, `--reader` env setup.
  - Model loaded via `AutoModelForMultimodalLM.from_pretrained(...)` with `device_map="auto"`, `torch_dtype="auto"`, `attn_implementation=args.attn` — this deliberately auto-resolves the Qwen3.8 architecture (robust for the NVFP4-quantized `Inferact/Qwen3.8-27B-NVFP4` checkpoint rather than hard-coding a class that may not exist in the installed transformers version).
  - Optional `torch.compile` when `--optimize` is set (same as other loaders).
  - Processor via `AutoProcessor` with `min_pixels`/`max_pixels` derived from `_patch_size_for_model()` (which yields patch size 32 for "qwen3" model ids, including "Qwen3.8").

### 3. `videsc/main.py`
- Added `load_qwen38_model_and_processor` to the `_run_vl` import list.
- Added the dispatch branch: `elif args.qwen38: model, processor = load_qwen38_model_and_processor(args)` (Qwen3.8 uses the standard `run_single_video` pipeline, same as `--qwen35` — unlike `--gemma4` which needs its chunked runner, Qwen3.8 is a standard Qwen video-capable model).
- Updated the module docstring and the `_run_vl` debug log to include `qwen38`.

### 4. `videsc/pipeline/runner.py`
- Added the import of `load_qwen38_model_and_processor`.
- Added the branch in `run_batch_threads()` so batch/threaded mode also loads the Qwen3.8 model and shares it across workers; Qwen3.8 then flows through the standard `run_single_video` path (only `--gemma4` is routed to `run_single_video_gemma4`).

### Verification already completed (passed)
- `py_compile` succeeded on all four modified files.
- `parse_args(['--vl','--qwen38','--video','x.mp4'])` → `model = Inferact/Qwen3.8-27B-NVFP4`, `model_hf = True`, `qwen38 = True` ✓
- Custom model override respected: `--qwen38 --model my/local-qwen38` keeps the custom model, `model_hf = False` ✓
- `videsc --help` shows `--qwen38` in the mutually exclusive group ✓
- `--qwen38 --gemma4` together correctly rejected: `error: argument --gemma4: not allowed with argument --qwen38` ✓

### Remaining (deferred to Act mode — this turn is Plan mode, so no further state-changing commands)
- One lightweight AST-based wiring check script was staged at `C:\Users\other\AppData\Local\Temp\videsc_qwen38_check.py` to assert the function exists and is imported/dispatched in all files (belt-and-suspenders beyond the runtime checks already passed).
- Optional: a live end-to-end test requires the `Inferact/Qwen3.8-27B-NVFP4` checkpoint to be available (local dir or HF download) and ~27B-class GPU VRAM; e.g.
  `python -m videsc.main --vl --qwen38 --video sample.mp4`
  or with a local copy: `--vl --qwen38 --model C:\models\Qwen3.8-27B-NVFP4 --model_full --video sample.mp4`.

Switch to Act mode and say the word if you'd like me to run the final wiring check or a dry-run (`--dry` loads the model but skips generation).