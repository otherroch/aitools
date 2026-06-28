# SCAIL-2 Integration for Character Replacement

## Overview

SCAIL-2 is an open-source model for **end-to-end controlled character animation** that enables character replacement in videos without relying on intermediate pose representations. This integration extends chararep to support SCAIL-2 as an alternative to traditional ONNX-based face swapping.

## Key Differences from ONNX-based Swap

| Feature | ONNX-based (inswapper/simswap) | SCAIL-2 (Diffusion-based) |
|---------|-------------------------------|---------------------------|
| **Approach** | Frame-by-frame face swap | End-to-end video generation |
| **Temporal Consistency** | Manual tracking/blending | Native temporal coherence |
| **Face Detection** | Required (RetinaFace) | Not required |
| **GPU Memory** | ~1-2 GB | ~12-16 GB (704p) |
| **Speed** | Fast (real-time) | Slower (diffusion steps) |
| **Quality** | Good | Excellent (better identity transfer) |
| **Reference** | Portrait images | Reference video or images |

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU with 12+ GB VRAM (for 704p resolution)
- cuDNN 8.0+

### Install Dependencies

```bash
# Install required packages
pip install torch torchvision torchaudio
pip install opencv-python numpy
```

### Download SCAIL-2 Model

Download the SCAIL-2 model from HuggingFace:

```bash
# Official PyTorch checkpoint
# https://huggingface.co/zai-org/SCAIL-2

# GGUF quantized version (for CPU/GPU inference)
# https://huggingface.co/realrebelai/SCAIL-2_GGUF
```

Place the downloaded model in your working directory or specify the path via CLI.

## Quick Start

### 1. Video-to-Video Mode (Recommended)

Use a reference video of the original character for best results:

```bash
# Basic usage with reference video
chararep --scail2 \
    --scail2-model-path path/to/scail2.ckpt \
    --scail2-reference-video reference_character.mp4 \
    -i input_video.mp4 \
    -o output_video.mp4

# With specific resolution
chararep --scail2 \
    --scail2-model-path path/to/scail2.ckpt \
    --scail2-reference-video reference_character.mp4 \
    --scail2-resolution 512p \
    -i input_video.mp4 \
    -o output_video.mp4
```

### 2. Image-to-Video Mode (Simpler)

Use portrait images of the original character:

```bash
# Using multiple portrait images
chararep --scail2 \
    --scail2-model-path path/to/scail2.ckpt \
    --scail2-reference-images ref1.jpg ref2.png ref3.jpg \
    -i input_video.mp4 \
    -o output_video.mp4
```

### 3. GGUF Quantized Model (Lower VRAM)

Use the GGUF quantized version for reduced memory usage:

```bash
chararep --scail2 \
    --scail2-model-path path/to/scail2.gguf \
    --scail2-use-gguf \
    --scail2-reference-video reference_character.mp4 \
    -i input_video.mp4 \
    -o output_video.mp4
```

## Configuration Options

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--scail2` | Enable SCAIL-2 mode | False |
| `--scail2-model-path` | Path to SCAIL-2 model (checkpoint or GGUF) | Required |
| `--scail2-reference-video` | Reference character video | Not required |
| `--scail2-reference-images` | Portrait images (repeat for multiple) | Not required |
| `--scail2-resolution` | Output resolution | 704p |
| `--scail2-use-gguf` | Use GGUF quantized model | False |
| `--scail2-device` | CUDA device ID | 0 |
| `--scail2-no-fp16` | Disable FP16 inference | False |

### Supported Resolutions

- **512p**: 512×896 (lower VRAM, faster)
- **704p**: 704×1280 (higher quality, recommended)

Both dimensions must be divisible by 32.

## Model Architecture

SCAIL-2 uses the following components:

1. **Wan VAE**: Variational Autoencoder for video encoding/decoding
2. **T5 Text Encoder**: For conditioning and identity transfer
3. **Diffusion Model**: Core SCAIL-2 model for character animation

The model is bundled with integrated Wan VAE and T5 components for convenience.

## Performance Considerations

### GPU Memory Requirements

| Resolution | VRAM Required |
|-----------|---------------|
| 512p | ~8-10 GB |
| 704p | ~12-16 GB |

### Speed Comparison

- **512p**: ~2-5 fps (depends on GPU)
- **704p**: ~1-3 fps (depends on GPU)

For real-time processing, consider using the GGUF quantized version or reducing resolution.

### Tips for Better Results

1. **Use high-quality reference videos**: Clear, well-lit footage of the original character
2. **Match resolution**: Use 704p for HD content, 512p for smaller videos
3. **Enable FP16**: Use FP16 inference for better performance (default)
4. **Multiple reference images**: Provide 3-5 diverse portrait images for better identity transfer

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptoms**: CUDA out of memory error during inference

**Solutions**:
- Use 512p resolution instead of 704p
- Use GGUF quantized model
- Reduce batch size if applicable
- Close other GPU-intensive applications

#### 2. Poor Identity Transfer

**Symptoms**: Replaced character doesn't look like the reference

**Solutions**:
- Use higher-quality reference videos/images
- Provide more diverse reference images
- Check that reference material is clear and well-lit
- Ensure the reference character matches the target character's age/gender

#### 3. Slow Processing

**Symptoms**: Very low FPS during inference

**Solutions**:
- Use 512p resolution
- Use GGUF quantized model
- Enable FP16 inference
- Consider using a more powerful GPU

#### 4. Model Loading Errors

**Symptoms**: Error when loading SCAIL-2 model

**Solutions**:
- Verify model file exists and is not corrupted
- Check that PyTorch version is compatible
- For GGUF models, ensure `llama-cpp-python` is installed
- Check GPU compatibility

### Debug Mode

Run with verbose logging for detailed diagnostics:

```bash
chararep --scail2 \
    --scail2-model-path path/to/scail2.ckpt \
    --scail2-reference-video reference.mp4 \
    -v \
    -i input.mp4 \
    -o output.mp4
```

## Advanced Usage

### JSON Configuration

Create a JSON config file for complex setups:

```json
{
    "input_video": "input.mp4",
    "output_video": "output.mp4",
    "enable_scail2": true,
    "scail2_model_path": "path/to/scail2.ckpt",
    "scail2_reference_video": "reference.mp4",
    "scail2_resolution": "704p",
    "scail2_use_gguf": false,
    "scail2_device_id": 0,
    "scail2_use_fp16": true
}
```

Then run:

```bash
chararep --config scail2_config.json
```

### Custom Resolution

SCAIL-2 requires both H and W to be divisible by 32. Supported examples:

- 512×896 (512p)
- 704×1280 (704p)
- 576×1024 (custom)
- 640×1152 (custom)

## References

- [SCAIL-2 Paper](https://huggingface.co/papers/2606.10804)
- [Project Page](https://teal024.github.io/SCAIL-2/)
- [GitHub Repository](https://github.com/zai-org/SCAIL-2)
- [HuggingFace Model](https://huggingface.co/zai-org/SCAIL-2)
- [GGUF Quantized Version](https://huggingface.co/realrebelai/SCAIL-2_GGUF)

## License

SCAIL-2 is licensed under MIT. See the LICENSE file for details.

## Contributing

Contributions to the SCAIL-2 integration are welcome! Please open issues or pull requests on the chararep repository.