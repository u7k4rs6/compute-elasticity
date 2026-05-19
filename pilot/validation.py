"""Data quality validation for pilot outputs.

validate_sample checks a single raw dict against the §4.5 schema.
scan_outputs scans a directory of .jsonl files for anomalies.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pilot.config import SCHEMA_VERSION

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "problem_id",
    "subject",
    "ground_truth",
    "model",
    "provider",
    "temperature",
    "sample_idx",
    "n_total_in_batch",
    "seed_hex",
    "prompt_template_hash",
    "full_response",
    "extracted_answer",
    "extraction_pass",
    "correct",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "timestamp",
    "api_metadata",
)

_VALID_ANSWERS = frozenset(["A", "B", "C", "D", "UNPARSEABLE"])
_VALID_SUBJECTS = frozenset(["physics", "chemistry", "biology"])
_VALID_PROVIDERS = frozenset(["together_ai"])
_VALID_PASS_NUMBERS = frozenset([1, 2, 3, 4, 5, 6])


def validate_sample(line: dict) -> list[str]:
    """Validate a single sample dict against the §4.5 schema.

    Returns a list of issue strings; empty list = valid.
    """
    issues: list[str] = []

    # Required fields
    for f in _REQUIRED_FIELDS:
        if f not in line:
            issues.append(f"missing_field:{f}")

    if issues:
        return issues  # no point checking values if fields are missing

    # schema_version
    if line["schema_version"] != SCHEMA_VERSION:
        issues.append(
            f"schema_version:{line['schema_version']!r} != {SCHEMA_VERSION!r}"
        )

    # subject
    if line["subject"] not in _VALID_SUBJECTS:
        issues.append(f"invalid_subject:{line['subject']!r}")

    # ground_truth
    if line["ground_truth"] not in "ABCD":
        issues.append(f"invalid_ground_truth:{line['ground_truth']!r}")

    # extracted_answer
    if line["extracted_answer"] not in _VALID_ANSWERS:
        issues.append(f"invalid_extracted_answer:{line['extracted_answer']!r}")

    # extraction_pass
    if line["extraction_pass"] not in _VALID_PASS_NUMBERS:
        issues.append(f"invalid_extraction_pass:{line['extraction_pass']!r}")

    # correct must be bool
    if not isinstance(line["correct"], bool):
        issues.append(f"correct_not_bool:{type(line['correct']).__name__}")

    # tokens non-negative
    for tok_field in ("input_tokens", "output_tokens"):
        if not isinstance(line[tok_field], int) or line[tok_field] < 0:
            issues.append(f"invalid_{tok_field}:{line[tok_field]!r}")

    # latency_ms non-negative
    if not isinstance(line["latency_ms"], (int, float)) or line["latency_ms"] < 0:
        issues.append(f"invalid_latency_ms:{line['latency_ms']!r}")

    # temperature a positive number
    if not isinstance(line["temperature"], (int, float)) or line["temperature"] < 0:
        issues.append(f"invalid_temperature:{line['temperature']!r}")

    # sample_idx >= 0
    if not isinstance(line["sample_idx"], int) or line["sample_idx"] < 0:
        issues.append(f"invalid_sample_idx:{line['sample_idx']!r}")

    # suspicious latency (> 5 minutes = likely stale / re-used)
    if isinstance(line["latency_ms"], (int, float)) and line["latency_ms"] > 300_000:
        issues.append(f"suspicious_latency_ms:{line['latency_ms']}")

    return issues


@dataclass
class ValidationReport:
    """Summary of a full outputs/ scan."""

    n_files: int = 0
    n_samples: int = 0
    n_invalid: int = 0
    issues_by_file: dict[str, list[str]] = field(default_factory=dict)
    duplicate_seeds: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return self.n_invalid == 0 and not self.duplicate_seeds


def scan_outputs(directory: Path) -> ValidationReport:
    """Scan all .jsonl files in directory for schema violations and anomalies.

    Checks:
    - All required fields present and valid
    - Duplicate (problem_id, seed_hex) pairs
    - Suspicious latencies
    - Token count mismatches (input_tokens + output_tokens > 0)
    """
    report = ValidationReport()
    seed_registry: dict[str, list[str]] = defaultdict(list)

    jsonl_files = sorted(directory.glob("**/*.jsonl"))
    report.n_files = len(jsonl_files)

    for path in jsonl_files:
        file_issues: list[str] = []
        try:
            lines = path.read_text().splitlines()
        except OSError as exc:
            file_issues.append(f"read_error:{exc}")
            report.issues_by_file[str(path)] = file_issues
            report.n_invalid += 1
            continue

        for lineno, raw in enumerate(lines, start=1):
            if not raw.strip():
                continue
            try:
                sample = json.loads(raw)
            except json.JSONDecodeError as exc:
                file_issues.append(f"line{lineno}:json_decode_error:{exc}")
                report.n_invalid += 1
                continue

            report.n_samples += 1
            sample_issues = validate_sample(sample)
            if sample_issues:
                report.n_invalid += 1
                for issue in sample_issues:
                    file_issues.append(f"line{lineno}:{issue}")

            # Track seed duplicates
            seed_key = f"{sample.get('problem_id','')}|{sample.get('seed_hex','')}"
            seed_registry[seed_key].append(f"{path}:{lineno}")

        if file_issues:
            report.issues_by_file[str(path)] = file_issues

    # Identify duplicates
    for seed_key, locations in seed_registry.items():
        if len(locations) > 1:
            report.duplicate_seeds.append(f"{seed_key} at {locations}")

    return report
