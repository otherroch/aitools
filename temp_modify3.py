"""Modify __init__.py"""

# Use LF since that's what the file uses
old = '    "CharacterReplacementPipeline",\n]'
new = '    "CharacterReplacementPipeline",\n    "SCAIL2Config",\n    "SCAIL2Runner",\n]'

with open("C:/Users/other/aitools/chararep/__init__.py", "r", encoding="utf-8") as f:
    content = f.read()

print("Searching for:", repr(old))
if old in content:
    content = content.replace(old, new)
    with open("C:/Users/other/aitools/chararep/__init__.py", "w", encoding="utf-8", newline="") as f:
        f.write(content)
    print("Updated successfully")
else:
    print("Pattern not found!")
    # Debug: show what we have around __all__
    idx = content.find("CharacterReplacementPipeline")
    if idx >= 0:
        print("Found at:", idx)
        print("Context:", repr(content[idx:idx+100]))