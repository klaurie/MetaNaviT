import os
import subprocess

# Paths to remove
paths_to_remove = [
    "LLM/models/blobs/sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6",
    "LLM/models/blobs/sha256-819c2adf5ce6df2b6bd2ae4ca90d2a69f060afeb438d0c171db57daa02e39c3d"
]

# Run commands
for path in paths_to_remove:
    command = f'git filter-branch --force --index-filter "git rm --cached --ignore-unmatch {path}" --prune-empty --tag-name-filter cat -- --all'
    subprocess.run(command, shell=True)

print("Large files removed from history. Now push the changes.")
