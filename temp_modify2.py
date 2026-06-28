"""Modify __init__.py"""

import pathlib

content = pathlib.Path("C:/Users/other/aitools/chararep/__init__.py").read_text("utf-8")
new = content.replace(
    '    "CharacterReplacementPipeline",\r\n]',
    '    "CharacterReplacementPipeline",\r\n    "SCAIL2Config",\r\n    "SCAIL2Runner",\r\n]',
)
pathlib.Path("C:/Users/other/aitools/chararep/__init__.py").write_text(new, "utf-8")
print("done")