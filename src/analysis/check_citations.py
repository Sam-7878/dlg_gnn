import re
from pathlib import Path

bib_text = Path("manuscript/_41_01_DLG_StreamMC/references.bib").read_text(encoding="utf-8")
bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,]+),", bib_text))

for tex_path in [
    Path("manuscript/_41_01_DLG_StreamMC/DLG-SelectiveStream_submission_r4.tex"),
    Path("docs/papers/_41_01_DLG_StreamMC/DLG-StreamMC.tex")
]:
    if not tex_path.exists():
        continue
    text = tex_path.read_text(encoding="utf-8")
    cites = set()
    for m in re.finditer(r"\\cite\{([^}]+)\}", text):
        for c in m.group(1).split(","):
            cites.add(c.strip())
    missing = cites - bib_keys
    print(f"File: {tex_path.name}")
    print(f"  Total cites: {len(cites)}, Missing from bib: {len(missing)}")
    if missing:
        print(f"  Missing: {sorted(list(missing))}")
