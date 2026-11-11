"""Run a model over the test set.

A run directory holds:
    manifest.json   - everything needed to say what was measured (model config, test-set
                      hash, prompt and grader versions, git commit, timestamps)
    results.jsonl   - one row per (case, rep) that produced model output, with its grade
    errors.jsonl    - attempts that produced no scorable output (infra failures). These
                      never occupy a result slot, and are retried on resume.

Runs are append-only: resuming skips (case, rep) pairs already in results.jsonl.
"""

from __future__ import annotations

import json
import platform
import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from . import prompt
from .grader import GRADER_VERSION, GoldExecutionError, grade
from .models import FatalError, InfraError, make_model
from .paths import ROOT, RUNS_DIR
from .testset import Case, TestSet

# Manifest fields that must match for a resumed run to be the same experiment.
_IDENTITY_KEYS = ("model_name", "model_config", "testset_sha256", "prompt_fingerprint", "grader_version", "reps")

# Error kinds that mean the harness doesn't understand what the provider sent back. One is
# noise; several in a row is a harness bug that would otherwise be paid for on every case.
_SYSTEMATIC_ERROR_KINDS = {"provider_error", "response_parse_error", "served_model_mismatch"}
MAX_CONSECUTIVE_SYSTEMATIC_ERRORS = 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_state() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def _sdk_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for package in ("anthropic", "openai", "google-genai"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def select_cases(testset: TestSet, case_ids: list[str] | None = None, limit: int | None = None) -> list[Case]:
    cases = list(testset.cases)
    if case_ids:
        known = testset.by_id()
        missing = [c for c in case_ids if c not in known]
        if missing:
            raise SystemExit(f"unknown case ids: {missing}")
        cases = [known[c] for c in case_ids]
    if limit is not None:
        cases = cases[:limit]
    return cases


def run(
    model_name: str,
    testset: TestSet,
    *,
    reps: int = 1,
    workers: int = 8,
    case_ids: list[str] | None = None,
    limit: int | None = None,
    run_dir: Path | None = None,
    runs_root: Path = RUNS_DIR,
    quiet: bool = False,
) -> Path:
    model = make_model(model_name)

    identity = {
        "model_name": model_name,
        "model_config": model.config,
        "testset_sha256": testset.sha256,
        "prompt_fingerprint": prompt.prompt_fingerprint(),
        "grader_version": GRADER_VERSION,
        "reps": reps,
    }
    if run_dir is not None and (run_dir / "manifest.json").exists():
        manifest = json.loads((run_dir / "manifest.json").read_text())
        diffs = [k for k in _IDENTITY_KEYS if manifest.get(k) != identity[k]]
        if diffs:
            raise SystemExit(f"cannot resume {run_dir}: {', '.join(diffs)} changed since it started")
        # A resumed run keeps the cases it started with; asking for different ones is an error.
        saved_ids = manifest["selection"]["case_ids"]
        if (case_ids or limit is not None) and [c.case_id for c in select_cases(testset, case_ids, limit)] != saved_ids:
            raise SystemExit(f"cannot resume {run_dir}: it covers {len(saved_ids)} saved cases; "
                             "resume without --cases/--limit")
        cases = select_cases(testset, saved_ids)
        manifest["resumed_at"] = [*manifest.get("resumed_at", []), _now()]
    else:
        cases = select_cases(testset, case_ids, limit)
        if run_dir is None:
            run_dir = _new_run_dir(runs_root, model_name)
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": run_dir.name,
            **identity,
            "testset_path": str(testset.path.relative_to(ROOT)) if testset.path.is_relative_to(ROOT) else str(testset.path),
            "testset_version": testset.version,
            "prompt_version": prompt.PROMPT_VERSION,
            "selection": {"case_ids": [c.case_id for c in cases], "limit": limit},
            "git": _git_state(),
            "environment": {
                "python": platform.python_version(),
                "sqlite": sqlite3.sqlite_version,
                "sdks": _sdk_versions(),
            },
            "started_at": _now(),
        }

    results_path, errors_path = run_dir / "results.jsonl", run_dir / "errors.jsonl"
    done = {(r["case_id"], r["rep"]) for r in read_jsonl(results_path)}
    todo = [(c, rep) for c in cases for rep in range(reps) if (c.case_id, rep) not in done]
    manifest["status"] = "running"
    _write_manifest(run_dir, manifest)

    if not quiet:
        print(f"run {run_dir.name}: {len(todo)} to do, {len(done)} already done "
              f"({len(cases)} cases x {reps} reps)", file=sys.stderr)

    stop = threading.Event()
    fatal: list[str] = []

    def work(case: Case, rep: int) -> tuple[str, dict]:
        if stop.is_set():
            return "skipped", {}
        system, user = prompt.build(case.db_path, case.question)
        base = {"case_id": case.case_id, "rep": rep, "attempted_at": _now()}
        try:
            completion = model.complete(case, system, user)
        except InfraError as e:
            return "error", {**base, "kind": e.kind, "message": str(e), "attempts": e.attempts}
        except FatalError:
            raise
        except Exception as e:  # provider bug: record it rather than crash the run
            return "error", {**base, "kind": "provider_error", "message": f"{type(e).__name__}: {e}"}
        try:
            g = grade(case.db_path, case.gold_sqls, completion.sql)
        except GoldExecutionError as e:
            return "error", {**base, "kind": "gold_error", "message": str(e), "completion": asdict(completion)}
        except Exception as e:  # a grader crash is a harness bug, not a model failure
            return "error", {**base, "kind": "grader_error", "message": f"{type(e).__name__}: {e}",
                             "completion": asdict(completion)}
        row = {
            **base,
            "pair_id": case.pair_id,
            "cluster": case.cluster,
            "difficulty": case.difficulty,
            "db_id": case.db_id,
            "question": case.question,
            "gold_sql": case.gold_sql,
            **asdict(completion),
            "passed": g.passed,
            "lenient_passed": g.lenient_passed,
            "reason": g.reason,
            "grade": g.to_dict(),
        }
        return "result", row

    def abort(message: str, pending_futures) -> None:
        if not stop.is_set():
            fatal.append(message)
            stop.set()
            for p in pending_futures:
                p.cancel()

    progress_total = len(todo)
    finished = 0
    consecutive_systematic = 0
    # The pool is shut down explicitly rather than by a `with` block: leaving a `with` on
    # Ctrl-C would wait for every queued case to run (and be billed) before stopping.
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        with open(results_path, "a") as results_f, open(errors_path, "a") as errors_f:
            pending = {pool.submit(work, c, rep) for c, rep in todo}
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in completed:
                    if fut.cancelled():  # dropped after a fatal error
                        continue
                    try:
                        kind, row = fut.result()
                    except FatalError as e:
                        abort(str(e), pending)
                        continue
                    if kind == "skipped":
                        continue
                    finished += 1
                    if kind == "result":
                        results_f.write(json.dumps(row) + "\n")
                        results_f.flush()
                        consecutive_systematic = 0
                    else:
                        errors_f.write(json.dumps(row) + "\n")
                        errors_f.flush()
                        if row["kind"] in _SYSTEMATIC_ERROR_KINDS:
                            consecutive_systematic += 1
                            if consecutive_systematic >= MAX_CONSECUTIVE_SYSTEMATIC_ERRORS:
                                abort(f"{consecutive_systematic} consecutive {row['kind']} errors - likely a harness "
                                      f"problem, stopping before more cases are billed. Last: {row['message'][:300]}",
                                      pending)
                    if not quiet:
                        _progress(finished, progress_total, kind, row)
    except BaseException as e:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)  # requests already in flight still finish
        manifest["status"] = "interrupted" if isinstance(e, KeyboardInterrupt) else "crashed"
        if not isinstance(e, KeyboardInterrupt):
            manifest["crash"] = f"{type(e).__name__}: {e}"
        manifest["updated_at"] = _now()
        _write_manifest(run_dir, manifest)
        print(f"\n{manifest['status']} - resume with: python -m harness run {model_name} --resume {run_dir}",
              file=sys.stderr)
        raise
    pool.shutdown(wait=True)

    all_results = read_jsonl(results_path)
    have = {(r["case_id"], r["rep"]) for r in all_results}
    missing = [(c.case_id, rep) for c in cases for rep in range(reps) if (c.case_id, rep) not in have]
    if fatal:
        manifest["status"] = "aborted"
        manifest["fatal_error"] = fatal[0]
    else:
        manifest["status"] = "complete" if not missing else "incomplete"
    manifest["finished_at"] = _now()
    manifest["counts"] = {"expected": len(cases) * reps, "results": len(all_results), "missing": len(missing)}
    _write_manifest(run_dir, manifest)

    if fatal:
        print(f"\nrun aborted: {fatal[0]}", file=sys.stderr)
    elif missing and not quiet:
        print(f"\n{len(missing)} (case, rep) slots have no result (see errors.jsonl); "
              f"resume with: python -m harness run {model_name} --resume {run_dir}", file=sys.stderr)
    return run_dir


def _new_run_dir(runs_root: Path, model_name: str) -> Path:
    """A fresh directory per run - never another run's, even if started in the same second."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = runs_root / f"{stamp}_{model_name}"
    for n in range(1, 1000):
        candidate = base if n == 1 else base.with_name(f"{base.name}-{n}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a unique run directory under {runs_root}")


def _progress(n: int, total: int, kind: str, row: dict) -> None:
    if kind == "error":
        line = f"ERROR {row['kind']}: {row['message'][:200]}"
    else:
        verdict = "pass" if row["passed"] else row["reason"]
        if row["status"] != "ok":
            verdict = row["status"]
        cost = f"${row['cost_usd']:.4f}" if row.get("cost_usd") is not None else ""
        lat = f"{row['latency_s']:.1f}s" if row.get("latency_s") is not None else ""
        line = f"{row['difficulty']:6s} {verdict:18s} {lat:>6s} {cost}"
    print(f"[{n:>4}/{total}] {row['case_id']} r{row['rep']} {line}", file=sys.stderr)


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    tmp = run_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(run_dir / "manifest.json")


def regrade(run_dir: Path, testset: TestSet, runs_root: Path = RUNS_DIR) -> Path:
    """Re-grade a run's saved SQL with the current grader and cases.yaml (e.g. after adding
    alt_gold_sql during adjudication). Writes a new run directory; the original is untouched
    and no model is called."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cases = testset.by_id()
    rows = read_jsonl(run_dir / "results.jsonl")
    # Grade everything before creating the output directory, so a failure leaves nothing behind.
    regraded, changed = [], 0
    for row in rows:
        case = cases.get(row["case_id"])
        if case is None:
            raise SystemExit(f"{row['case_id']} is not in the current cases.yaml")
        try:
            g = grade(case.db_path, case.gold_sqls, row["sql"])
        except GoldExecutionError as e:
            raise SystemExit(f"{row['case_id']}: {e}") from e
        changed += g.passed != row["passed"]
        regraded.append({**row, "passed": g.passed, "lenient_passed": g.lenient_passed, "reason": g.reason,
                         "grade": g.to_dict()})
    out_dir = _new_run_dir(runs_root, f"{run_dir.name}.regrade")
    with open(out_dir / "results.jsonl", "w") as f:
        for row in regraded:
            f.write(json.dumps(row) + "\n")
    if (run_dir / "errors.jsonl").exists():
        (out_dir / "errors.jsonl").write_text((run_dir / "errors.jsonl").read_text())
    new_manifest = {
        **manifest,
        "run_id": out_dir.name,
        "regraded_from": {"run_id": manifest["run_id"], "grader_version": manifest["grader_version"],
                          "testset_sha256": manifest["testset_sha256"]},
        "grader_version": GRADER_VERSION,
        "testset_sha256": testset.sha256,
        "regraded_at": _now(),
    }
    _write_manifest(out_dir, new_manifest)
    print(f"regraded {len(rows)} rows into {out_dir.name}; {changed} verdicts changed", file=sys.stderr)
    return out_dir
