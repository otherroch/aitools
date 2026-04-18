# aitools

**AI dataset preparation toolkit for diffusion model LoRA training.**

`aitools` provides four command-line tools and Python APIs for preparing image and video datasets. Also included is a tool called `chararep` which uses the output of the other tools to swap faces in videos. 

| Tool | Command | Description |
|------|---------|-------------|
| Portrait Prep | `portrait-prep` | End-to-end portrait image preparation (`convert → crop → caption → augment`) |
| Video Crop | `vicrop` | Extract face-cropped PNG frames from video files |
| Video Description | `videsc` | Generate text descriptions for video files — fast WD14 tag-based captions (default) or rich natural-language descriptions via Qwen3-VL / Qwen3-omni / Qwen3.5   (`--vl`) or Gemma4 (`--gemma4`) |
| Character Replace | `chararep` | Replace character faces in a video using deep face-swapping models (inswapper, SimSwap, uniface, hyperswap, blendswap). Other tools like `vicrop` and `portrait-prep` can be used to create the portrait galleries required by `chararep` |

---

## Quick start

```bash
git clone https://github.com/otherroch/aitools.git
cd aitools

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# upgrade pip to get --group support
python -m pip install -U pip

# install base dependencies (CPU only support)
pip install --group base

# install aitools
pip install -e .
```

For individual feature installs, GPU support, `chararep` dependencies (including basicsr),
**docker images**, install extras, system prerequisites, or project structure 
see [installation](doc/INSTALLATION.md).

## Documentation

For a complete reference of each tool's command line options, usage examples, and Python API:

- [portrait-prep](doc/PORTRAIT_PREP.md) — portrait dataset preparation pipeline: **convert, crop, caption, augment, cpcap**
- [vicrop](doc/VICROP.md) — video face-crop extraction with identity clustering and reference photo selection
- [videsc](doc/VIDESC.md) — video description generator (WD14 tags, Qwen3-VL, Qwen3-omni, Qwwn3.5, Gemma4) natural language)
- [chararep](doc/CHARAREP.md) — video character face-replacement pipeline (architecture, CLI, config, input requirements, VRAM guidelines)
- [Python API and Testing](doc/API_AND_TESTING.md) — Python API examples for each tool, test commands, and coverage

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
