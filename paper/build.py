"""Build the paper and the arXiv upload, and check that the citations resolved.

    TECTONIC=/path/to/tectonic python paper/build.py

Writes ``paper/dist/reverie-bench.pdf`` and ``paper/dist/reverie-arxiv.tar.gz``.
Needs tectonic (a self-contained TeX engine; set $TECTONIC or put it on PATH) and
pypdf.

The check is the point of this script. Two earlier builds looked fine and were
not: one had a natbib/bibliography-style clash that only surfaced when compiling
from the upload files alone, and one produced a PDF with an empty reference list
and "[?]" citations while every log grep passed, because tectonic re-runs BibTeX
and reports a missing .bib as a warning. So the PDF compiled from the unpacked
upload is inspected directly: every citation must render as a number and the
reference list must contain one entry per cited key.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
DIST = PAPER / "dist"
UPLOAD = ["main.tex", "neurips.sty", "refs.bib", "main.bbl", "figures/fig1.pdf"]


def tectonic() -> str:
    exe = os.environ.get("TECTONIC") or shutil.which("tectonic")
    if not exe:
        sys.exit("tectonic not found: set $TECTONIC or put it on PATH")
    return exe


def compile_tex(workdir: Path, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [tectonic(), "-X", "compile", "main.tex", "--outdir", str(outdir),
         "--keep-intermediates"],
        cwd=workdir, capture_output=True, text=True,
    )
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        sys.exit(f"compile failed in {workdir}:\n{log[-3000:]}")
    if "errors were issued by BibTeX" in log:
        sys.exit(f"BibTeX reported errors in {workdir}:\n{log[-3000:]}")
    return outdir / "main.pdf"


def cited_keys() -> set[str]:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    keys: set[str] = set()
    for group in re.findall(r"\\cite[pt]?\{([^}]*)\}", tex):
        keys.update(k.strip() for k in group.split(","))
    return keys


def check_pdf(pdf: Path, n_keys: int) -> None:
    from pypdf import PdfReader

    text = "\n".join(page.extract_text() for page in PdfReader(pdf).pages)
    problems = []
    if re.search(r"\[\s*\?", text):
        problems.append("unresolved citations ([?]) in the text")
    refs = text.split("References", 1)[-1].split("Results by environment instance")[0]
    entries = len(re.findall(r"^\[\d+\]", refs, flags=re.M))
    if entries != n_keys:
        problems.append(f"reference list has {entries} entries, {n_keys} keys are cited")
    if problems:
        sys.exit(f"{pdf}: " + "; ".join(problems))
    print(f"  {pdf.name}: {entries} references, all citations resolved")


def main() -> None:
    keys = cited_keys()
    DIST.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        print("1. full build from paper/")
        pdf = compile_tex(PAPER, tmp / "full")
        check_pdf(pdf, len(keys))
        shutil.copy(tmp / "full" / "main.bbl", PAPER / "main.bbl")

        print("2. arXiv package")
        tarball = DIST / "reverie-arxiv.tar.gz"
        with tarfile.open(tarball, "w:gz") as tar:
            for name in UPLOAD:
                tar.add(PAPER / name, arcname=name)
        (PAPER / "main.bbl").unlink()
        print(f"  {tarball.relative_to(PAPER)}: {', '.join(UPLOAD)}")

        print("3. rebuild from the unpacked package alone")
        unpacked = tmp / "unpacked"
        with tarfile.open(tarball) as tar:
            tar.extractall(unpacked, filter="data")
        pdf = compile_tex(unpacked, tmp / "standalone")
        check_pdf(pdf, len(keys))
        shutil.copy(pdf, DIST / "reverie-bench.pdf")
        print(f"  wrote {(DIST / 'reverie-bench.pdf').relative_to(PAPER)}")


if __name__ == "__main__":
    main()
