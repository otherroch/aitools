"""Modify main.py to add SCAIL-2 CLI args."""

import pathlib

with open("C:/Users/other/aitools/chararep/main.py", "r", encoding="utf-8") as f:
    content = f.read()

old = '        help=(\n            "A find/replace pair for each character. "\n            "FIND is a directory of original-face photos; "\n            "REPLACE is a directory of new-face photos."\n        ),\n    )'
new = old + '\n\n    # ── SCAIL-2 options ──────────────────────────────────────────────\n    p.add_argument(\n        "--scail2",\n        dest="scail2_enabled",\n        action="store_true",\n        help="Enable SCAIL-2 end-to-end character replacement.",\n    )\n    p.add_argument(\n        "--scail2-mode",\n        dest="scail2_mode",\n        choices=["replacement", "animation"],\n        default="replacement",\n        help="SCAIL-2 mode (default: replacement).",\n    )\n    p.add_argument(\n        "--scail2-res",\n        dest="scail2_resolution",\n        choices=["512p", "704p"],\n        default="704p",\n        help="Output resolution for SCAIL-2 (default: 704p).",\n    )\n    p.add_argument(\n        "--scail2-steps",\n        dest="scail2_steps",\n        type=int,\n        default=30,\n        help="Number of diffusion steps (default: 30).",\n    )\n    p.add_argument(\n        "--scail2-cfg",\n        dest="scail2_cfg_scale",\n        type=float,\n        default=3.5,\n        help="Classifier-free guidance scale (default: 3.5).",\n    )\n    p.add_argument(\n        "--scail2-fps",\n        dest="scail2_fps",\n        type=float,\n        default=24.0,\n        help="Target output FPS (default: 24.0).",\n    )\n    p.add_argument(\n        "--scail2-seed",\n        dest="scail2_seed",\n        type=int,\n        default=42,\n        help="Random seed for reproducibility (default: 42).",\n    )\n    p.add_argument(\n        "--scail2-model",\n        dest="scail2_model_path",\n        type=str,\n        default="",\n        help="Path to SCAIL-2 model checkpoint directory.",\n    )'

if old in content:
    content = content.replace(old, new)
    with open("C:/Users/other/aitools/chararep/main.py", "w", encoding="utf-8", newline="") as f:
        f.write(content)
    print("Updated successfully")
else:
    print("Pattern not found!")