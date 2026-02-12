#!/usr/bin/env python3
"""Resolve api_client.py merge conflicts."""
import subprocess, re

# Read the current conflicted file
with open('/home/dq4443/dev/work/hyperwave-community/hyperwave_community/api_client.py', 'r') as f:
    lines = f.readlines()

# Get compute_adjoint_gradient from Jim's branch
result = subprocess.run(
    ['git', 'show', 'origin/feb-inverse-design-notebook:hyperwave_community/api_client.py'],
    cwd='/home/dq4443/dev/work/hyperwave-community',
    capture_output=True, text=True
)
jim_lines = result.stdout.split('\n')

# Extract compute_adjoint_gradient (lines 2996-3305 in Jim's, 0-indexed: 2995-3304)
# Find the function start and end
cag_start = None
cag_end = None
for i, line in enumerate(jim_lines):
    if line.startswith('def compute_adjoint_gradient('):
        cag_start = i
    if cag_start is not None and i > cag_start and line.startswith('def _run_optimization_ws('):
        cag_end = i
        break

compute_adjoint_gradient_lines = jim_lines[cag_start:cag_end]

# Build the resolved file
output = []

# Part 1: Keep lines 1-3030 (0-indexed: 0-3029) - everything before run_optimization
# But first resolve conflict 1 and 2 (they are at lines 860 and 925)
for i in range(3030):  # lines 0-3029
    line = lines[i]
    # Skip conflict markers and resolve inline
    output.append(line)

# Part 2: Insert compute_adjoint_gradient
output.append('\n')
for line in compute_adjoint_gradient_lines:
    output.append(line + '\n')

# Part 3: Take Jim's side from conflict 3 (lines 3352-3960, 0-indexed)
# Lines 3353 to 3960 (skip >>>>>>> at line 3961)
for i in range(3352, 3961):  # 0-indexed: lines 3353-3961 in 1-indexed
    line = lines[i]
    if line.startswith('>>>>>>> origin/feb-inverse-design-notebook'):
        continue
    output.append(line)

# Join and fix the earlier two conflicts
content = ''.join(output)

# Fix conflict 1: _API_CONFIG
content = content.replace(
    "<<<<<<< HEAD\n    'api_url': 'https://hyperwave-gateway-production.up.railway.app'\n=======\n    'api_url': 'https://spinsphotonics--hyperwave-api-fastapi-app.modal.run',\n    'gateway_url': 'https://hyperwave-gateway-production.up.railway.app',\n>>>>>>> origin/feb-inverse-design-notebook",
    "    'api_url': 'https://hyperwave-gateway-production.up.railway.app',\n    'gateway_url': None,"
)

# Fix conflict 2: timeout
content = content.replace(
    "<<<<<<< HEAD\n                timeout=120  # Modal cold start can take time\n=======\n                timeout=30\n>>>>>>> origin/feb-inverse-design-notebook",
    "                timeout=30"
)

# Fix gpu_type defaults: change H100 to B200 everywhere in function signatures
content = content.replace("gpu_type: str = \"H100\"", "gpu_type: str = \"B200\"")

# Write resolved file
with open('/home/dq4443/dev/work/hyperwave-community/hyperwave_community/api_client.py', 'w') as f:
    f.write(content)

print("Resolved api_client.py")

# Verify no conflict markers remain
if '<<<<<<' in content or '>>>>>>>' in content or '\n=======\n' in content:
    print("WARNING: Conflict markers still present!")
    for i, line in enumerate(content.split('\n')):
        if '<<<<<<<' in line or '>>>>>>>' in line or line.strip() == '=======':
            print(f"  Line {i+1}: {line.rstrip()}")
else:
    print("No conflict markers found - clean!")
