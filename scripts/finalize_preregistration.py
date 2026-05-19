"""Finalize Phase 2 pre-registration.

Loads GPQA Diamond (requires HF_TOKEN in .env), runs stratified_sample,
computes the problem-ID list hash, writes problem IDs into preregistration.md,
locks prompt hashes in pilot/prompts.py (already done), and prints the commands
needed to commit and tag pre-pilot-v6.0.

Usage:
    source .venv/bin/activate
    python scripts/finalize_preregistration.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    # Load .env
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        print("ERROR: HF_TOKEN not set in .env — fill it in and re-run.")
        sys.exit(1)

    print("Loading GPQA Diamond from HuggingFace…")
    sys.path.insert(0, str(ROOT))
    from pilot.data_loader import load_gpqa_diamond, stratified_sample

    problems = load_gpqa_diamond()
    print(f"Loaded {len(problems)} problems.")

    sample = stratified_sample(problems, n=50, seed=42)
    print(f"Stratified sample: {len(sample)} problems.")

    subject_counts: dict[str, int] = {}
    for p in sample:
        subject_counts[p.subject] = subject_counts.get(p.subject, 0) + 1
    print("Subject distribution:", subject_counts)

    problem_ids = [p.id for p in sample]
    ids_json = json.dumps(problem_ids, indent=2)
    ids_hash = hashlib.sha256(json.dumps(problem_ids).encode("utf-8")).hexdigest()

    print(f"\nProblem-ID list hash: {ids_hash}")

    # Patch preregistration.md
    prereg_path = ROOT / "preregistration.md"
    if not prereg_path.exists():
        print(
            "ERROR: preregistration.md not found — run this script after writing the skeleton."
        )
        sys.exit(1)

    text = prereg_path.read_text()

    # Replace placeholder blocks
    if "PROBLEM_IDS_PLACEHOLDER" in text:
        text = text.replace(
            "PROBLEM_IDS_PLACEHOLDER",
            ids_json,
        )
    else:
        print(
            "WARNING: PROBLEM_IDS_PLACEHOLDER not found in preregistration.md — check manually."
        )

    if "PROBLEM_IDS_HASH_PLACEHOLDER" in text:
        text = text.replace("PROBLEM_IDS_HASH_PLACEHOLDER", ids_hash)
    else:
        print("WARNING: PROBLEM_IDS_HASH_PLACEHOLDER not found — check manually.")

    if "SUBJECT_DIST_PLACEHOLDER" in text:
        dist_str = ", ".join(f"{s}: {n}" for s, n in sorted(subject_counts.items()))
        text = text.replace("SUBJECT_DIST_PLACEHOLDER", dist_str)

    prereg_path.write_text(text)
    print(f"\nUpdated {prereg_path}")
    print("\nNext steps:")
    print("  git add preregistration.md pilot/prompts.py pilot/data_loader.py")
    print("  git commit -m 'phase-2: pre-registration lock-in'")
    print("  git tag pre-pilot-v6.0")
    print("  git push origin main --tags")


if __name__ == "__main__":
    main()
