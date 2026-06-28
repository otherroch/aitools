import subprocess
import sys

result = subprocess.run(['git', 'show', 'HEAD:vicrop/crop.py'], capture_output=True)
with open('vicrop/crop.py', 'wb') as f:
    f.write(result.stdout)

print(f"Restored {len(result.stdout)} bytes")