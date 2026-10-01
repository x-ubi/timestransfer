"""Copy the reviewed writing package into the local thesis checkout; no Git operations."""

from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
source = root / "thesis-package/thesis"
target = Path("/home/ubuntu/thesis")
assert (target / "AGENTS.md").exists()
files = [p for p in sorted(source.rglob("*")) if p.is_file()]
assert len(files) == 46, len(files)
# Preserve every replaced local file for review/recovery outside the thesis Git tree.
backup = root / "thesis-package/evidence/pre-integration"
for path in files:
    relative = path.relative_to(source)
    dest = target / relative
    if dest.exists() and dest.read_bytes() == path.read_bytes():
        continue
    if dest.exists():
        saved = backup / relative
        if not saved.exists():
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dest, saved)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
for path in files:
    assert path.read_bytes() == (target / path.relative_to(source)).read_bytes()
print(
    f"Integrated and byte-verified {len(files)} files into {target}; existing AGENTS.md preserved."
)
