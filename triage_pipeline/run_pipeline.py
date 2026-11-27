"""Foreground entry point: RUN_STAGE=unit|consistency|profile|full, PLATFORM=adapter name."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from triage_pipeline import core  # noqa: E402

STAGE = os.environ.get("RUN_STAGE", "unit")
PLATFORM = os.environ.get("PLATFORM") or sorted(path.stem for path in (HERE / "configs").glob("*.json"))[0]
STAGES = {"unit", "consistency", "profile", "full"}
EXPECTED_HASHES = {
    "SPEC.md": "4d922e28fa249b2844b5be2ff65b4e2022425271a9daefb12f1649f9ff2cb73c",
    "SPEC_AMENDMENT_1.md": "eb22e07bdaff7a3369b8c710c6adc307eec3bbff64cc3e869e9f21aba8c09bce",
    "PREREG.md": "fe6b07e4fa136dcaaa0ddc4464efd32b7265d69e000798ece0ef6b6db0986f0c",
    "PREREG_AMENDMENT_1.md": "5b13f4c704b1884f4b453e2202060d1f276a578d8acd444cabe6067159a53ae1",
}
OUT_BASE = ROOT / "analysis/20261127_triage_pipeline_dev_run"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def verify_inputs() -> dict:
    paths = {"SPEC.md": HERE / "SPEC.md", "SPEC_AMENDMENT_1.md": HERE / "SPEC_AMENDMENT_1.md",
             "PREREG.md": OUT_BASE / "PREREG.md",
             "PREREG_AMENDMENT_1.md": OUT_BASE / "PREREG_AMENDMENT_1.md"}
    hashes = {name: sha256(path) for name, path in paths.items()}
    if hashes != EXPECTED_HASHES:
        raise ValueError("specification hash mismatch")
    sources = [HERE / name for name in ("core.py", "methods.py", "metrics.py", "selection.py", "reporting.py", "full.py", "run_pipeline.py")]
    sources += [HERE / "adapters" / f"{PLATFORM}.py", HERE / "configs" / f"{PLATFORM}.json",
                HERE / "tests" / f"consistency_{PLATFORM}.py"]
    sources += sorted((HERE / "tests").glob("test_*.py"))
    return {"specification_sha256": hashes,
            "code_and_config": {str(path.resolve()): sha256(path) for path in sources}}


def config() -> dict:
    return json.loads((HERE / "configs" / f"{PLATFORM}.json").read_text(encoding="utf-8"))


def forbidden_output_text(value: str) -> bool:
    return re.search(r"at://|did:|https?://|@[A-Za-z0-9]|\"(?:handle|text|full_text)\"\s*:", value, re.I) is not None


def output_scan(out: Path) -> dict:
    files = []
    for path in sorted(out.iterdir()):
        if path.is_file():
            files.append(path.name)
            if path.suffix in {".json", ".jsonl", ".md"} and forbidden_output_text(path.read_text(encoding="utf-8")):
                raise ValueError("output privacy scan failed")
    return {"passed": True, "scanned_files": files}


def cache_write(out: Path, threads: list[core.Thread], replies: list[core.Reply],
                subsets: dict[str, dict[str, bool]] | None = None) -> dict:
    thread_path = out / "threads.jsonl"
    reply_path = out / "replies.jsonl"
    with thread_path.open("w", encoding="utf-8") as stream:
        for thread in threads:
            row = asdict(thread)
            for key in ("created_at", "group_time", "observed_until"):
                row[key] = row[key].isoformat() if row[key] else None
            if subsets is not None:
                row["report_subsets"] = subsets.get(thread.thread_id, {})
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    with reply_path.open("w", encoding="utf-8") as stream:
        for reply in replies:
            stream.write(json.dumps(asdict(reply), separators=(",", ":")) + "\n")
    return {"threads": {"path": str(thread_path.resolve()), "sha256": sha256(thread_path)},
            "replies": {"path": str(reply_path.resolve()), "sha256": sha256(reply_path)}}


def cache_read(cache_record_path: Path) -> tuple[list[core.Thread], list[core.Reply], dict, dict]:
    source = json.loads(cache_record_path.read_text(encoding="utf-8"))
    if source["status"] != "complete" or source["stage"] != "consistency" or source["platform"] != PLATFORM:
        raise ValueError("incompatible cache record")
    if source["inputs"]["specification_sha256"] != EXPECTED_HASHES:
        raise ValueError("cache specification hash mismatch")
    cache = source["results"]["cache"]
    for item in cache.values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("cache content hash mismatch")
    threads, replies, subsets = [], [], {}
    with Path(cache["threads"]["path"]).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            for key in ("created_at", "group_time", "observed_until"):
                row[key] = core.parse_time(row[key])
            flags = row.pop("report_subsets", {})
            thread = core.Thread(**row)
            threads.append(thread)
            if flags:
                subsets[thread.thread_id] = flags
    with Path(cache["replies"]["path"]).open(encoding="utf-8") as stream:
        for line in stream:
            replies.append(core.Reply(**json.loads(line)))
    return threads, replies, subsets, {"record_path": str(cache_record_path.resolve()),
                                      "record_sha256": sha256(cache_record_path), "cache": cache}


def latest_cache_record() -> Path:
    override = os.environ.get("CACHE_RUN")
    if override:
        path = Path(override)
        return path / "run_record.json" if path.is_dir() else path
    candidates = sorted(OUT_BASE.glob(f"consistency_*/{PLATFORM}/run_record.json"), reverse=True)
    for path in candidates:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "complete":
            return path
    raise FileNotFoundError("no completed consistency cache")


def cleaned(threads: list[core.Thread], replies: list[core.Reply]) -> tuple[dict, dict]:
    grouped = defaultdict(list)
    for reply in replies:
        grouped[reply.thread_id].append(reply)
    output, counts = {}, Counter()
    for thread in threads:
        output[thread.thread_id], count = core.basic_clean(thread, grouped[thread.thread_id])
        counts.update(count)
    return output, dict(counts)


def run_consistency(out: Path, rec: dict) -> None:
    adapter = importlib.import_module(f"triage_pipeline.adapters.{PLATFORM}")
    print("[2/5] reading protected source data", flush=True)
    def progress(done, total):
        print(f"    source progress {done}/{total}", flush=True)
    threads, replies, source_counts, extra = adapter.load(ROOT, progress)
    rec["inputs"]["source"] = adapter.manifest(ROOT, source_counts)
    rec["results"]["source_counts"] = source_counts
    print("[3/5] validating schema and source-specific consistency gates", flush=True)
    rec["results"]["schema"] = core.validate_schema(threads, replies)
    checker = importlib.import_module(f"triage_pipeline.tests.consistency_{PLATFORM}")
    rec["results"]["consistency"] = checker.check(threads, replies, source_counts, extra, ROOT)
    print("[4/5] writing hashed, text-free unified cache", flush=True)
    rec["results"]["cache"] = cache_write(out, threads, replies, extra if isinstance(extra, dict) else None)
    rec["results"]["cleaning_counts"] = cleaned(threads, replies)[1]


def run_profile(out: Path, rec: dict) -> None:
    print("[2/5] loading and verifying prior unified cache", flush=True)
    threads, replies, subsets, source = cache_read(latest_cache_record())
    rec["inputs"]["cache"] = source
    core.validate_schema(threads, replies)
    checker = importlib.import_module(f"triage_pipeline.tests.consistency_{PLATFORM}")
    if hasattr(checker, "check_cached"):
        rec["results"]["cache_count_gate"] = checker.check_cached(threads, subsets)
    clean_by_thread, counts = cleaned(threads, replies)
    rec["results"]["cleaning_counts"] = counts
    print("[3/5] profiling training portions only", flush=True)
    if rec["config"]["split_mode"] == "time":
        raw, starts = core.raw_time_groups(threads)
        result = (core.profile(raw["train"], clean_by_thread, rec["config"], next_start=starts["selection"])
                  if starts["selection"] is not None else
                  {"status": "insufficient_data", "reason": "empty_selection_segment"})
        result["raw_segment_starts"] = {name: value.isoformat() if value else None for name, value in starts.items()}
        if result["status"] == "complete":
            segmented, report = core.time_segments(threads, result["H_hours"] * 3600)
            result["split"] = report
            result["segments"] = segment_summary(segmented, clean_by_thread, result, rec["config"])
    else:
        groups = sorted({thread.group_id for thread in threads})
        result = {"status": "complete", "folds": {}, "group_count": len(groups),
                  "unknown_followup": True, "time_forward": False}
        for completed, held in enumerate(groups, 1):
            train = [thread for thread in threads if thread.group_id != held]
            fold = core.profile(train, clean_by_thread, rec["config"])
            fold["outer_test_group"] = held
            if fold["status"] == "complete":
                fold["segments"] = segment_summary({"train": train,
                    "selection": [], "test": [thread for thread in threads if thread.group_id == held]},
                    clean_by_thread, fold, rec["config"])
            result["folds"][held] = fold
            print(f"    outer-fold profile {completed}/{len(groups)}", flush=True)
    rec["results"]["profile"] = result
    rec["cutoff"] = result.get("T_minutes") if rec["config"]["split_mode"] == "time" else {
        key: value.get("T_minutes") for key, value in result["folds"].items()}
    print("[4/5] writing profile and split report", flush=True)
    save_json(out / "profile.json", result)


def segment_summary(segments: dict, clean_by_thread: dict, profile_result: dict, configuration: dict) -> dict:
    result = {}
    cutoff = profile_result["T_minutes"] * 60
    horizon = profile_result["H_hours"] * 3600
    for name, rows in segments.items():
        mature_rows = [thread for thread in rows if core.mature(thread, horizon)]
        candidates = 0
        candidate_groups = set()
        for thread in mature_rows:
            observed, _ = core.snapshot(thread, clean_by_thread[thread.thread_id], cutoff)
            if core.candidate(core.features(thread, observed, cutoff), configuration["min_replies"]):
                candidates += 1
                candidate_groups.add(thread.group_id)
        result[name] = {"threads_after_boundary": len(rows), "groups_after_boundary": len({t.group_id for t in rows}),
                        "maturity_dropped": len(rows) - len(mature_rows),
                        "mature_threads": len(mature_rows), "candidate_threads": candidates,
                        "candidate_groups": len(candidate_groups)}
    return result


def run_unit(out: Path, rec: dict) -> None:
    import unittest
    suite = unittest.defaultTestLoader.discover(str(HERE / "tests"), pattern="test_*.py")
    print(f"[2/5] executing {suite.countTestCases()} synthetic unit tests", flush=True)
    result = unittest.TextTestRunner(verbosity=1, stream=sys.stdout).run(suite)
    rec["results"]["unit"] = {"run": result.testsRun, "failures": len(result.failures),
                               "errors": len(result.errors), "passed": result.wasSuccessful()}
    if not result.wasSuccessful():
        raise ValueError("unit tests failed")


def main() -> int:
    started = datetime.now(timezone.utc)
    out = OUT_BASE / f"{STAGE}_{started:%Y%m%d_%H%M%S_%f}" / PLATFORM
    out.mkdir(parents=True, exist_ok=False)
    record_path = out / "run_record.json"
    rec = {"status": "running", "started_at": started.isoformat(), "completed_at": None,
           "stage": STAGE, "platform": PLATFORM, "config": None, "inputs": {},
           "split": None, "cutoff": None, "seed": None, "metrics": {}, "results": {},
           "outputs": [], "failure": None}
    print(f"START output directory: {out.resolve()}", flush=True)
    save_json(record_path, rec)
    try:
        print("[1/5] verifying fixed specification and configuration hashes", flush=True)
        if STAGE not in STAGES or not re.fullmatch(r"[a-z_]+", PLATFORM):
            raise ValueError("unsupported stage or adapter")
        rec["inputs"] = verify_inputs()
        rec["config"] = config()
        rec["split"] = rec["config"]["split_mode"]
        rec["cutoff"] = rec["config"]["T_minutes"]
        if STAGE == "full":
            rec["seed"] = {"model": [20261110, 20261111, 20261112], "bootstrap": 42}
        save_json(record_path, rec)
        if STAGE == "unit":
            run_unit(out, rec)
        elif STAGE == "consistency":
            run_consistency(out, rec)
        elif STAGE == "profile":
            run_profile(out, rec)
        else:
            from triage_pipeline.full import run_full
            run_full(out, rec)
        print("[5/5] scanning outputs and finalizing run record", flush=True)
        rec["results"]["output_scan"] = output_scan(out)
        rec["status"] = "complete"
    except Exception as exc:
        rec["status"] = "failed"
        rec["failure"] = {"type": type(exc).__name__, "stage": STAGE}
    rec["completed_at"] = datetime.now(timezone.utc).isoformat()
    rec["outputs"] = sorted(path.name for path in out.iterdir() if path.is_file())
    save_json(record_path, rec)
    print(f"{'SUCCESS' if rec['status'] == 'complete' else 'FAILURE'} output directory: {out.resolve()}", flush=True)
    return 0 if rec["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
