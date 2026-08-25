All tests pass. Let me provide a final summary of the changes:

## Summary of Changes

### 1. `C:/Users/other/aitools/videsc/cli/args.py`
- **Added `--qwen38` flag** (line 163): A mutually exclusive group argument that sets `model is Qwen3.8 (e.g. Inferact/qwen3.8-27b-nvfp4)`
- **Added default model handling for `--qwen38`** (lines 435-439): When `--qwen38` is set and the user didn't explicitly change `--model`, it defaults to `"Inferact/qwen3.8-27b-nvfp4"` with `model_hf=True`, consistent with how `--qwen35` and `--gemma4` work.

### 2. `C:/Users/other/aitools/videsc/model/loader.py`
- **Added `load_qwen38_model_and_processor(args)` function** (lines 363+): 
  - Uses `AutoModelForMultimodalLM` and `AutoProcessor` (following the HF model card for Inferact/qwen3.8-27b-nvfp4 which specifies `auto_model: "AutoModelForMultimodalLM"`)
  - Similar pattern to `load_gemma4_model_and_processor` but without Gemma 4's special `padding_side="left"` requirement
  - Uses `_patch_size_for_model()` which correctly returns 32 for any model_id containing "qwen3"
  - Supports quantization, torch.compile, and pixel limits like other loaders
  - Shares cached model/processor across threads for batch mode

### 3. `C:/Users/other/aitools/videsc/main.py`
- **Added import for `load_qwen38_model_and_processor`** (line 109)
- **Added `elif args.qwen38:` branch** (lines 161-162) in `_run_vl()` to route to the new loader function, using the standard `run_single_video` pipeline (not a Gemma4-style special pipeline since Qwen3.8 NVFP4 follows the standard VL pattern)

### 4. `C:/Users/other/aitools/videsc/pipeline/runner.py`
- **Added import for `load_qwen38_model_and_processor`** (line 19)
- **Added `elif args.qwen38:` branch** (lines 848-849) in `run_batch()` to load the model for batch mode, using the standard `run_single_video` worker (not Gemma4 special handling)

All code follows existing patterns and conventions in the codebase. The `--qwen38` flag behaves consistently with `--qwen35` and `--gemma4` in terms of default model selection, argument parsing, and integration with batch mode.