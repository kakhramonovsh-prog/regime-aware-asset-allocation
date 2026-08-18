"""Pre-publication audit: what would become public, and what must not.

Making a repository public is effectively irreversible: content is
indexed, forked and cached within minutes and a later deletion does not
recall it. Git history is published too, so a secret that was committed
and later removed is still public. This audit therefore scans the whole
history, not just the working tree.

    python scripts/public_release_audit.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Credential-shaped strings. Kept specific: a pattern that fires on every
# hex string trains the reader to ignore the report.
SECRET_PATTERNS = {
    "AWS access key": r"AKIA[0-9A-Z]{16}",
    "GitHub token": r"gh[pousr]_[A-Za-z0-9]{16,}",
    "Slack token": r"xox[baprs]-[A-Za-z0-9-]{10,}",
    "OpenAI key": r"sk-[A-Za-z0-9]{20,}",
    "Anthropic key": r"sk-ant-[A-Za-z0-9-]{20,}",
    "private key block": r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY",
    "generic assignment": (r"(?i)(password|passwd|secret|api[_-]?key|token)"
                           r"\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"),
}

PERSONAL_PATTERNS = {
    "email address": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "US phone number": r"\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
    "street address": r"\b\d{1,5}\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd)\b",
}

# Vendor series that may not be redistributed.
VENDOR_HINTS = ("data/raw/", "data/processed/", "prices_", "fred_")

SIZE_LIMIT_MB = 5.0


def run(*args: str) -> str:
    return subprocess.run(args, cwd=PROJECT_ROOT, capture_output=True,
                          text=True, errors="ignore").stdout


def tracked_files() -> list[str]:
    return [f for f in run("git", "ls-files").splitlines() if f]


def all_paths_in_history() -> set[str]:
    output = run("git", "log", "--all", "--pretty=format:", "--name-only",
                 "--diff-filter=A")
    return {line for line in output.splitlines() if line.strip()}


def main() -> None:
    problems: list[str] = []
    warnings: list[str] = []

    print("=" * 72)
    print("1. Vendor data (licensed; redistribution prohibited)")
    print("=" * 72)
    history = all_paths_in_history()
    leaked = sorted(p for p in history
                    if any(h in p for h in VENDOR_HINTS)
                    and not p.endswith((".gitkeep", "README.md"))
                    and "manifest" not in p)
    if leaked:
        problems.append(f"vendor data in git history: {leaked}")
        for path in leaked:
            print(f"  [BLOCK] {path}")
    else:
        print(f"  clean: no vendor file in any of {len(history)} paths ever added")

    print("\n" + "=" * 72)
    print("2. Credentials, across full history")
    print("=" * 72)
    blob = run("git", "log", "--all", "-p", "--no-color")
    found_secret = False
    for label, pattern in SECRET_PATTERNS.items():
        hits = re.findall(pattern, blob)
        if hits:
            found_secret = True
            problems.append(f"{label}: {len(hits)} match(es) in history")
            print(f"  [BLOCK] {label}: {len(hits)} match(es)")
    if not found_secret:
        print(f"  clean: no credential pattern in {len(blob):,} chars of history")

    print("\n" + "=" * 72)
    print("3. Personal information in tracked files")
    print("=" * 72)
    for label, pattern in PERSONAL_PATTERNS.items():
        occurrences: dict[str, set[str]] = {}
        for name in tracked_files():
            path = PROJECT_ROOT / name
            if not path.is_file() or path.suffix in {".png", ".pdf", ".parquet"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for match in set(re.findall(pattern, text)):
                value = match if isinstance(match, str) else match[0]
                occurrences.setdefault(value, set()).add(name)
        for value, files in sorted(occurrences.items()):
            warnings.append(f"{label} {value!r} in {sorted(files)}")
            print(f"  [REVIEW] {label}: {value}")
            for name in sorted(files):
                print(f"             {name}")
    if not warnings:
        print("  none found")

    print("\n" + "=" * 72)
    print(f"4. Oversized artifacts (> {SIZE_LIMIT_MB} MB)")
    print("=" * 72)
    big = []
    for name in tracked_files():
        path = PROJECT_ROOT / name
        if path.is_file():
            mb = path.stat().st_size / 1_048_576
            if mb > SIZE_LIMIT_MB:
                big.append((mb, name))
    for mb, name in sorted(big, reverse=True):
        warnings.append(f"large file {name} ({mb:.1f} MB)")
        print(f"  [REVIEW] {mb:6.1f} MB  {name}")
    if not big:
        print(f"  none: largest tracked file is under {SIZE_LIMIT_MB} MB")

    packed = run("git", "count-objects", "-vH")
    for line in packed.splitlines():
        if line.startswith("size-pack"):
            print(f"  repository {line}")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    if problems:
        print("BLOCKED - do not publish:")
        for item in problems:
            print(f"  - {item}")
    else:
        print("No blocking issue found.")
    if warnings:
        print(f"\n{len(warnings)} item(s) for human review before publishing:")
        for item in warnings:
            print(f"  - {item}")

    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
