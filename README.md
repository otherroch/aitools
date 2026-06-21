# Project README - Character Recognition and Replacement Tool

## Overview
This project provides character recognition and replacement capabilities for image processing, using multiple AI models including Google Gemini, Qwen, and open-source frameworks.

## Installation and Setup
1. Ensure Python 3.8+ is installed
2. Install requirements from `requirements.txt` or `pyproject.toml`
3. Set up virtual environment if needed

## Usage
```bash
# Basic usage - Process an image
python -m chararep.main --image_path input.jpg --char_index 0

# The tool processes images through character recognition pipeline:
# 1. Loads and preprocesses the input image
# 2. Uses AI models (Google Gemini/Qwen) for character detection
# 3. Applies character replacement based on specified parameters
# 4. Outputs processed image with character transformations
```

## Main Components
- `chararep/main.py`: Command-line interface and file processing
- `chararep/chararep.py`: Core character recognition implementation
- `CharacterRec`: Main class handling OCR and character replacement operations

## Key Features
- Multiple AI model support (Google Gemini, Qwen)
- CPU/GPU processing capability
- Batch processing support
- Image preprocessing and enhancement
- Selective character replacement

## License
[License information from original repository]

## Support
For issues or feature requests, refer to the original repository documentation.

Last updated: $(date +%Y-%m-%d)

## Quick Start Example
```python
from chararep.chararep import CharacterRec

# Initialize character recognition
recognizer = CharacterRec()

# Process an image
result = recognizer.match("input.jpg")

# Get recognized characters and their replacements
print(f"Processed: {result}")