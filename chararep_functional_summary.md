# Functional Summary of chararep

## Overview
chararep is a character recognition and replacement tool that processes image files to perform character detection and substitution operations. It leverages multiple AI models and open-source libraries to recognize and replace characters in images.

## Module Structure

**Location**: `chararep/` directory in the project
**Main Entry Point**: `chararep/main.py`
**Core Implementation**: `chararep/chararep.py`

**Files**:
- `__init__.py`: Module initialization and exports
- `chararep.py`: Core character recognition logic (`CharacterRec` class)
- `main.py`: Command-line interface (CLI) and file processing

## Key Components

### 1. CharacterRec Class (`chararep.py`)
This is the core class that handles character recognition and replacement functionality:

**Attributes**:
- `model`: Pre-trained model for character recognition
- `patch_size`: Processing parameter for image patches
- `device`: Computation device (CPU/GPU)

**Methods**:
- `__init__()`: Initializes the model and processing parameters
- `match()`: Processes input image and returns character recognition results

### 2. Main Function (`main.py`)
CLI entry point that:
- Parses command-line arguments for input images
- Processes each image through the CharacterRec pipeline
- Handles file operations and result output

**Arguments**:
- `--image_path`: Path to input image file
- `--char_index`: Character index for replacement operations

## Core Functionality

### Processing Pipeline
1. **Image Loading**: Reads input image files
2. **Character Recognition**: Uses AI models to detect/recognize characters
3. **Character Replacement**: Applies specified character transformations
4. **Result Output**: Saves processed images

### Supported Models
- **Google Gemini**: Used for high-level processing and analysis
- **Qwen**: Alternative model for character recognition
- **Open-source tools**: Complementary libraries for processing

### Key Operations
- OCR (Optical Character Recognition)
- Character detection and classification
- Selective character replacement
- Batch processing of multiple images

## Technical Specifications

### Dependencies
- AI/ML frameworks (TensorFlow/PyTorch-based)
- Computer vision libraries
- Image processing utilities

### Processing Parameters
- Patch-based image analysis
- Multi-model support for robustness
- Configurable device placement (CPU/GPU)

## Usage Examples

```bash
# Basic usage
python -m chararep.main --image_path input.jpg --char_index 0

# Batch processing
# (Command-line iteration supported)
```

## Applications
- Document digitization
- Form processing
- Image-based character editing
- Data extraction from images

## Output
- Processed images with character replacements
- Character recognition metadata
- Processing statistics and logs

## Notes
- Designed for flexibility with multiple AI model backends
- Supports both CPU and GPU processing
- Batch-oriented for efficiency with multiple files
- Modular architecture for easy extension and modification

**Last Updated**: Based on current codebase structure and analysis
**Maintained by**: Other Roch (original repository context)**