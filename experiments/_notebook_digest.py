"""Print the text output of every executed notebook cell.

    python experiments/_notebook_digest.py            # all
    python experiments/_notebook_digest.py E4

A notebook's conclusions live in its printed tables, but reading them means
opening six large JSON files. This dumps just the stdout of each cell so a
re-run can be diffed against what README claims, which is the check that catches
a headline quietly ceasing to reproduce after an engine change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def digest(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n{'=' * 78}\n{path.name}\n{'=' * 78}")
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        chunks = []
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                chunks.append("".join(out.get("text", [])))
            elif out.get("output_type") == "error":
                chunks.append(f"!! {out.get('ename')}: {out.get('evalue')}")
            elif "text/plain" in out.get("data", {}):
                chunks.append("".join(out["data"]["text/plain"]))
        text = "".join(chunks).rstrip()
        if text:
            print(f"\n--- cell {i} (exec {cell.get('execution_count')}) ---")
            print(text)


def main() -> None:
    pats = sys.argv[1:] or ["E"]
    seen = sorted({p for pat in pats for p in HERE.glob(f"{pat}*.ipynb")})
    if not seen:
        sys.exit("no matching notebooks")
    for path in seen:
        digest(path)


if __name__ == "__main__":
    main()
