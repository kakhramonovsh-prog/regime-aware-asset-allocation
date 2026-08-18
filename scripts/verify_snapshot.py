"""Verify data on disk against the frozen snapshot manifest.

Reports, file by file, whether the current data matches the snapshot the
paper was computed from. A mismatch is not necessarily an error: Yahoo
restates adjusted closes when distributions occur, so a rebuilt dataset
legitimately differs from the frozen one. The point is to make the
difference visible rather than silent.

    python scripts/verify_snapshot.py

Exit code 0 when every hash matches (exact reproduction is possible),
1 when any differs (only a live rebuild is possible).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eda import sha256_of  # noqa: E402

MANIFEST = PROJECT_ROOT / "data" / "snapshots" / "manifest_2026-08-06.json"


def main() -> None:
    if not MANIFEST.exists():
        sys.exit(f"Snapshot manifest not found: {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    print(f"Snapshot:    {MANIFEST.name}")
    print(f"Downloaded:  {manifest['downloaded_at_utc']}")
    print(f"Git commit:  {manifest['git_commit'][:12]}")
    print(f"Files:       {len(manifest['files'])}\n")

    missing, mismatched, matched = [], [], []
    for rel_path, info in sorted(manifest["files"].items()):
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            missing.append(rel_path)
        elif sha256_of(path) != info["sha256"]:
            mismatched.append(rel_path)
        else:
            matched.append(rel_path)

    for rel_path in matched:
        print(f"  [MATCH]    {rel_path}")
    for rel_path in mismatched:
        info = manifest["files"][rel_path]
        print(f"  [DIFFERS]  {rel_path}"
              + (f"  (snapshot: {info['rows']} rows, "
                 f"{info['first_date']} to {info['last_date']})"
                 if "rows" in info else ""))
    for rel_path in missing:
        print(f"  [MISSING]  {rel_path}")

    print(f"\n{len(matched)} match, {len(mismatched)} differ, {len(missing)} missing")

    if missing or mismatched:
        print("\nEXACT REPRODUCTION NOT POSSIBLE with the data on disk.")
        if missing:
            print("  Missing files: run scripts/download_data.py, or restore the")
            print("  frozen snapshot (see docs/REPRODUCTION.md).")
        if mismatched:
            print("  Differing files: the vendor has restated data since the")
            print("  snapshot was frozen. A live rebuild will run and produce")
            print("  valid outputs, but its numbers will NOT equal the paper's.")
        sys.exit(1)

    print("\nAll hashes match the frozen snapshot.")
    print("Exact reproduction is possible.")


if __name__ == "__main__":
    main()
