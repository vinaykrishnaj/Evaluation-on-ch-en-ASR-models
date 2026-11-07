"""Convert a '# %%' cell-marked .py file into a .ipynb. Usage: python py_to_ipynb.py in.py out.ipynb [KEY=VALUE ...]
Each KEY="..." line at the top of the first code cell can be overridden, e.g. MODEL_KEY=madlad."""
import json, re, sys

src, dst, *overrides = sys.argv[1:]
text = open(src, encoding="utf-8").read()
parts = re.split(r"^# %%.*\n", text, flags=re.M)
cells = []
for i, p in enumerate(parts):
    p = p.strip("\n")
    if not p:
        continue
    if i == 0:   # header comments before the first marker
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [l[2:] + "\n" if l.startswith("# ") else l + "\n" for l in p.splitlines()]})
        continue
    for ov in overrides:
        k, v = ov.split("=", 1)
        p = re.sub(rf'^{k}\s*=\s*"[^"]*"', f'{k} = "{v}"', p, flags=re.M)
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": p.splitlines(keepends=True)})
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", dst, len(cells), "cells")
