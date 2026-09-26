"""Sanity-check the draft: every \\cite key resolves, and no placeholder reaches the text."""
import pathlib
import re

HERE = pathlib.Path(__file__).parent
tex = (HERE / "main.tex").read_text(encoding="utf-8")
bib = (HERE / "refs.bib").read_text(encoding="utf-8")

cited = {
    k.strip()
    for group in re.findall(r"\\cite\{([^}]*)\}", tex)
    for k in group.split(",")
    if k.strip()
}
defined = set(re.findall(r"@\w+\{([^,]+),", bib))

print("cited keys        :", len(cited))
print("defined in refs.bib:", len(defined))
print("cited but UNDEFINED:", sorted(cited - defined) or "none")
print("defined but unused :", sorted(defined - cited) or "none")

placeholders = sorted(k for k in defined if "PLACEHOLDER" in k)
reaching_text = sorted(k for k in placeholders if k in cited)
print()
print("placeholder entries:", placeholders)
print("placeholders cited in text:", reaching_text or "NONE (safe)")

body = re.sub(r"\\[a-zA-Z]+|[{}$&\\]", " ", tex)
print()
print("approx words in source:", len(body.split()))
