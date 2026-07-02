"""Download and pin the Cricsheet snapshot; record date + SHA256 in data/MANIFEST.

Cricsheet revises past files; the snapshot is downloaded once and never
re-downloaded silently (SPEC §2). Re-running against an existing snapshot only
verifies the hash against the MANIFEST.
"""

import hashlib
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

ARCHIVE_URL = "https://cricsheet.org/downloads/all_json.zip"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MANIFEST = DATA_DIR / "MANIFEST"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_dir() -> Path:
    """Return the pinned snapshot directory recorded in the MANIFEST."""
    for line in MANIFEST.read_text().splitlines():
        if line.startswith("snapshot_dir="):
            return DATA_DIR / "raw" / line.split("=", 1)[1]
    raise FileNotFoundError(f"no snapshot_dir entry in {MANIFEST}")


def verify() -> Path:
    """Verify the existing snapshot against the MANIFEST hash; return its dir."""
    entries = dict(line.split("=", 1) for line in MANIFEST.read_text().splitlines() if "=" in line)
    snap = DATA_DIR / "raw" / entries["snapshot_dir"]
    actual = sha256_of(snap / "all_json.zip")
    if actual != entries["sha256"]:
        raise ValueError(
            f"snapshot hash mismatch: MANIFEST says {entries['sha256']}, zip is {actual}"
        )
    return snap


def download() -> Path:
    today = date.today().isoformat()
    snap = DATA_DIR / "raw" / f"snapshot_{today}"
    if MANIFEST.exists():
        print("MANIFEST exists — verifying pinned snapshot instead of re-downloading")
        snap = verify()
        print(f"verified: {snap}")
        return snap
    snap.mkdir(parents=True, exist_ok=True)
    zip_path = snap / "all_json.zip"
    print(f"downloading {ARCHIVE_URL} → {zip_path}")
    urllib.request.urlretrieve(ARCHIVE_URL, zip_path)
    digest = sha256_of(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(snap / "json")
    n_files = len(list((snap / "json").glob("*.json")))
    MANIFEST.write_text(
        f"source={ARCHIVE_URL}\n"
        f"snapshot_date={today}\n"
        f"snapshot_dir=snapshot_{today}\n"
        f"sha256={digest}\n"
        f"n_json_files={n_files}\n"
    )
    print(f"pinned snapshot_{today}: {n_files} files, sha256={digest}")
    return snap


if __name__ == "__main__":
    sys.exit(0 if download() else 1)
