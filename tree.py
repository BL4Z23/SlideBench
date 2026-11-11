import sys
from pathlib import Path

IGNORE = {"venv", ".venv", "env", "__pycache__", ".git", ".idea", ".vscode",
          "node_modules", ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache",
          "build", "dist"}
MAX_DEPTH = 4      # how many folder levels to open
MAX_ITEMS = 12     # show at most this many entries per folder

def walk(path, prefix="", depth=0):
    if depth >= MAX_DEPTH:
        return
    entries = sorted(
        (p for p in path.iterdir()
         if p.name not in IGNORE and not p.name.endswith(".egg-info")),
        key=lambda p: (p.is_file(), p.name.lower()),   # folders first
    )
    hidden = max(0, len(entries) - MAX_ITEMS)
    entries = entries[:MAX_ITEMS]
    for i, p in enumerate(entries):
        last = (i == len(entries) - 1) and hidden == 0
        print(prefix + ("└── " if last else "├── ") + p.name + ("/" if p.is_dir() else ""))
        if p.is_dir():
            walk(p, prefix + ("    " if last else "│   "), depth + 1)
    if hidden:
        print(prefix + f"└── ... (+{hidden} more)")

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
print(root.resolve().name + "/")
walk(root)