"""Safely copy protected inputs into three model-local artifact contracts.

This user-facing script is intentionally configuration-only: run it without
arguments from any working directory. It never moves or deletes a protected
source and never overwrites an existing destination with different content.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent / "experiments"
RUN_SMOKES = True
CHUNK_BYTES = 8 * 1024 * 1024
PROGRESS_BYTES = 256 * 1024 * 1024
PROGRESS_SECONDS = 10.0


@dataclass(frozen=True)
class ArtifactSpec:
    source: str
    destination: str
    sha256: str


PHEME_SOURCE = "data/protected_research_assets/pheme_v5_strict30"
TWITTER_DATA = "data/derived/20260830_024514_251_twitter15_16_30min_time_respecting_preventable_impact"
TWITTER_SPLIT = "data/derived/20260830_024515_294_twitter15_16_source_level_split_v1"

ARTIFACTS = (
    ArtifactSpec(
        f"{TWITTER_DATA}/thread_level_records.csv",
        "effective_models/twitter_graph_only_rf/artifacts/20260830_024514_251_twitter15_16_30min_time_respecting_preventable_impact/thread_level_records.csv",
        "0F92EAD327D6B8DE1B78E5499EF91552DDEA9F0DDAF83E7E486A1BEDC58A2CCC",
    ),
    ArtifactSpec(
        f"{TWITTER_SPLIT}/assignments.csv",
        "effective_models/twitter_graph_only_rf/artifacts/20260830_024515_294_twitter15_16_source_level_split_v1/assignments.csv",
        "B129C0AF12A04348FF5111C7B53F41DFF411EBD27987F699D85F1CE6A9BB5916",
    ),
    ArtifactSpec(
        f"{TWITTER_SPLIT}/manifest.json",
        "effective_models/twitter_graph_only_rf/artifacts/20260830_024515_294_twitter15_16_source_level_split_v1/manifest.json",
        "99EC7D9A43030F20A7922C8889FFF1BD9BFA6BD2F1D1D0653EFFEEAD625E63A4",
    ),
    ArtifactSpec(
        f"{PHEME_SOURCE}/pheme_graphs_roberta_30min_replyv4_preventableimpact_semantic.pt",
        "effective_models/pheme_v5_text_rf/artifacts/pheme_v5_strict30/pheme_graphs_roberta_30min_replyv4_preventableimpact_semantic.pt",
        "DD5637B76ACEE343537A866272D4D488F31918802C6EB229FD642678A3D0019B",
    ),
    ArtifactSpec(
        f"{PHEME_SOURCE}/pheme_node_features_roberta_replyv4.csv",
        "effective_models/pheme_v5_text_rf/artifacts/pheme_v5_strict30/pheme_node_features_roberta_replyv4.csv",
        "BF66708A642B0924C0FD964186972BA72A646F6B4ACFA72661D221340DF2849D",
    ),
    ArtifactSpec(
        f"{PHEME_SOURCE}/pheme_graphs_roberta_30min_replyv4_preventableimpact_semantic.pt",
        "effective_models/pheme_active_quiet_calibrated_rf/artifacts/pheme_v5_strict30/pheme_graphs_roberta_30min_replyv4_preventableimpact_semantic.pt",
        "DD5637B76ACEE343537A866272D4D488F31918802C6EB229FD642678A3D0019B",
    ),
    ArtifactSpec(
        f"{PHEME_SOURCE}/pheme_node_features_roberta_replyv4.csv",
        "effective_models/pheme_active_quiet_calibrated_rf/artifacts/pheme_v5_strict30/pheme_node_features_roberta_replyv4.csv",
        "BF66708A642B0924C0FD964186972BA72A646F6B4ACFA72661D221340DF2849D",
    ),
    ArtifactSpec(
        f"{PHEME_SOURCE}/pheme_reply_level_v4.csv",
        "effective_models/pheme_active_quiet_calibrated_rf/artifacts/pheme_v5_strict30/pheme_reply_level_v4.csv",
        "573D284C2F6EAF63EF72E9A1227082D734E2A4DC58DC60C8F4E6FFCB56AA9B78",
    ),
)

SMOKE_SCRIPTS = (
    "effective_models/twitter_graph_only_rf/training/run_smoke.py",
    "effective_models/pheme_v5_text_rf/training/run_smoke.py",
    "effective_models/pheme_active_quiet_calibrated_rf/training/run_smoke.py",
)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path, *, label: str = "hash") -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    completed = 0
    last_report_bytes = 0
    last_report_time = time.monotonic()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK_BYTES):
            digest.update(block)
            completed += len(block)
            now = time.monotonic()
            if (
                completed == total
                or completed - last_report_bytes >= PROGRESS_BYTES
                or now - last_report_time >= PROGRESS_SECONDS
            ):
                percent = 100.0 if not total else completed * 100.0 / total
                print(f"      {label}: {completed:,}/{total:,} bytes ({percent:.1f}%)", flush=True)
                last_report_bytes = completed
                last_report_time = now
    return digest.hexdigest().upper()


def copy_verified(
    source: Path,
    destination: Path,
    expected_sha256: str,
    *,
    verified_source_sha256: str | None = None,
) -> str:
    """Copy atomically after source validation; return ``copied`` or ``skipped``."""
    expected = expected_sha256.upper()
    if not source.is_file():
        raise FileNotFoundError(f"Missing protected source: {source}")
    source_hash = (
        verified_source_sha256.upper()
        if verified_source_sha256 is not None
        else sha256_file(source, label="source hash")
    )
    if source_hash != expected:
        raise ValueError(f"Source SHA-256 mismatch for {source}: {source_hash} != {expected}")

    if destination.exists():
        if not destination.is_file():
            raise FileExistsError(f"Destination exists but is not a file: {destination}")
        destination_hash = sha256_file(destination, label="existing destination hash")
        if destination_hash != expected:
            raise FileExistsError(
                f"Refusing to overwrite different destination: {destination} ({destination_hash})"
            )
        return "skipped"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise FileExistsError(f"Refusing to reuse existing partial file: {temporary}")
    digest = hashlib.sha256()
    total = source.stat().st_size
    completed = 0
    last_report_bytes = 0
    last_report_time = time.monotonic()
    try:
        with source.open("rb") as src, temporary.open("xb") as dst:
            while block := src.read(CHUNK_BYTES):
                dst.write(block)
                digest.update(block)
                completed += len(block)
                now = time.monotonic()
                if (
                    completed == total
                    or completed - last_report_bytes >= PROGRESS_BYTES
                    or now - last_report_time >= PROGRESS_SECONDS
                ):
                    percent = 100.0 if not total else completed * 100.0 / total
                    print(f"      copy: {completed:,}/{total:,} bytes ({percent:.1f}%)", flush=True)
                    last_report_bytes = completed
                    last_report_time = now
            dst.flush()
            os.fsync(dst.fileno())
        copied_hash = digest.hexdigest().upper()
        if copied_hash != expected:
            raise ValueError(f"Copied SHA-256 mismatch for {destination}: {copied_hash} != {expected}")
        if destination.exists():
            raise FileExistsError(f"Destination appeared during copy; refusing overwrite: {destination}")
        os.link(temporary, destination)
        temporary.unlink()
        return "copied"
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate_contract() -> tuple[int, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    bytes_needed = 0
    effective_root = (REPO_ROOT / "effective_models").resolve()
    for spec in ARTIFACTS:
        source = (REPO_ROOT / spec.source).resolve()
        destination = (REPO_ROOT / spec.destination).resolve()
        if not destination.is_relative_to(effective_root):
            raise ValueError(f"Artifact destination escapes effective_models: {destination}")
        if "artifacts" not in destination.parts:
            raise ValueError(f"Artifact destination is not under artifacts/: {destination}")
        ignored = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", str(destination)],
            cwd=REPO_ROOT,
            check=False,
        )
        if ignored.returncode != 0:
            raise ValueError(f"Artifact destination is not covered by .gitignore: {destination}")
        if not source.is_file():
            raise FileNotFoundError(f"Missing protected source: {source}")
        if not destination.exists():
            bytes_needed += source.stat().st_size
        rows.append(
            {
                **asdict(spec),
                "source_absolute": str(source),
                "destination_absolute": str(destination),
                "source_bytes": source.stat().st_size,
                "destination_existed_at_start": destination.exists(),
            }
        )
    free = shutil.disk_usage(REPO_ROOT).free
    if free < bytes_needed + 1024**3:
        raise OSError(f"Insufficient free space: need {bytes_needed:,} bytes plus 1 GiB reserve")
    return bytes_needed, rows


def main() -> None:
    started = datetime.now(timezone.utc)
    output = OUTPUT_ROOT / f"{started:%Y%m%d_%H%M%S_%f}_local_artifact_copy"
    output.mkdir(parents=True, exist_ok=False)
    record_path = output / "run_record.json"
    record: dict[str, object] = {
        "status": "running",
        "started_at": started.isoformat(),
        "configuration": {
            "copy_only_never_move": True,
            "overwrite_different_destination": False,
            "run_smokes": RUN_SMOKES,
            "chunk_bytes": CHUNK_BYTES,
            "artifact_specs": [asdict(spec) for spec in ARTIFACTS],
            "smoke_scripts": list(SMOKE_SCRIPTS),
        },
        "input_artifacts": sorted({spec.source for spec in ARTIFACTS}),
        "split": "unchanged; artifact restoration only",
        "cutoff": "unchanged; model-specific canonical settings",
        "seed": 42,
        "copy_results": [],
        "smoke_results": [],
        "output_file_inventory": ["run_record.json"],
        "failure_details": None,
    }
    _write_json(record_path, record)
    print(f"[1/5] Starting safe model-local artifact restoration", flush=True)
    print(f"      Output directory: {output.resolve()}", flush=True)
    try:
        print("[2/5] Validating sources, destinations, ignore contract, and free space...", flush=True)
        bytes_needed, contract_rows = _validate_contract()
        record["contract"] = contract_rows
        record["bytes_to_copy_at_start"] = bytes_needed
        record["free_bytes_at_start"] = shutil.disk_usage(REPO_ROOT).free
        _write_json(record_path, record)
        print(f"      bytes requiring copy: {bytes_needed:,}", flush=True)

        print(f"[3/5] Copying/verifying {len(ARTIFACTS)} artifact destinations...", flush=True)
        verified_sources: dict[Path, str] = {}
        for index, spec in enumerate(ARTIFACTS, start=1):
            source = (REPO_ROOT / spec.source).resolve()
            destination = (REPO_ROOT / spec.destination).resolve()
            print(f"  artifact {index}/{len(ARTIFACTS)}: {destination}", flush=True)
            if source not in verified_sources:
                verified_sources[source] = sha256_file(source, label="source hash")
            action = copy_verified(
                source,
                destination,
                spec.sha256,
                verified_source_sha256=verified_sources[source],
            )
            row = {"destination": str(destination), "action": action, "sha256": spec.sha256}
            record["copy_results"].append(row)  # type: ignore[union-attr]
            _write_json(record_path, record)
            print(f"      {action}: SHA-256 verified", flush=True)

        print("[4/5] Running canonical smoke entry points...", flush=True)
        if RUN_SMOKES:
            for index, relative_script in enumerate(SMOKE_SCRIPTS, start=1):
                script = (REPO_ROOT / relative_script).resolve()
                command = [sys.executable, str(script)]
                print(f"  smoke {index}/{len(SMOKE_SCRIPTS)}: {relative_script}", flush=True)
                completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
                smoke_row = {
                    "script": relative_script,
                    "command": command,
                    "exit_code": completed.returncode,
                    "status": "passed" if completed.returncode == 0 else "failed",
                }
                record["smoke_results"].append(smoke_row)  # type: ignore[union-attr]
                _write_json(record_path, record)
                if completed.returncode != 0:
                    raise RuntimeError(f"Smoke failed ({completed.returncode}): {relative_script}")
        else:
            print("      RUN_SMOKES=False; smoke stage intentionally skipped", flush=True)

        print("[5/5] Writing final run record...", flush=True)
        record.update(
            {
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "metrics_results": {
                    "artifacts_copied": sum(r["action"] == "copied" for r in record["copy_results"]),
                    "artifacts_skipped_verified": sum(r["action"] == "skipped" for r in record["copy_results"]),
                    "smokes_passed": sum(r["status"] == "passed" for r in record["smoke_results"]),
                },
                "research_safety": {
                    "sources_moved_or_deleted": False,
                    "different_destinations_overwritten": False,
                    "datasets_labels_splits_modified": False,
                },
            }
        )
        _write_json(record_path, record)
        print(f"SUCCESS: model-local artifacts and smoke validation saved to: {output.resolve()}", flush=True)
    except Exception:
        record.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "failure_details": traceback.format_exc(),
            }
        )
        _write_json(record_path, record)
        print(f"FAILURE: artifact setup record preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
