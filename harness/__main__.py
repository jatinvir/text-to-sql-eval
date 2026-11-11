"""Command line entry point: python -m harness <command> ...

    run MODEL [--reps N] [--workers N] [--cases c001,c002] [--limit N] [--resume RUN_DIR]
    report RUN_DIR            metrics for one run (also writes RUN_DIR/report.md, summary.json)
    compare RUN_DIR RUN_DIR…  side-by-side table + paired differences vs the first run
    audit RUN_DIR             adjudication sheet of failures (RUN_DIR/audit.md)
    regrade RUN_DIR           re-grade saved SQL with the current grader + cases.yaml
    selfcheck                 oracle must score 100%, null baseline 0% (no API calls)
    prompt CASE_ID            print the exact prompt sent for a case
    models                    list the model roster and whether each vendor's API key is set

API keys come from the environment or from a .env file in the repo root (gitignored;
copy .env.example). Variables already set in the environment take precedence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from . import audit, prompt, report, runner, testset
from .models import KEY_ENV_VARS, load_roster
from .paths import ROOT


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=value lines from .env into the environment without overriding anything
    already set. Deliberately tiny: no interpolation, no multi-line values."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def _run_dir(p: str) -> Path:
    path = Path(p)
    if not (path / "manifest.json").exists():
        raise SystemExit(f"{path} is not a run directory (no manifest.json)")
    return path


def cmd_run(args) -> None:
    ts = testset.load()
    case_ids = args.cases.split(",") if args.cases else None
    run_dir = runner.run(
        args.model, ts, reps=args.reps, workers=args.workers, case_ids=case_ids, limit=args.limit,
        run_dir=Path(args.resume) if args.resume else None,
    )
    _report(run_dir)


def _report(run_dir: Path) -> None:
    run = report.load_run(run_dir)
    summary = report.summarize(run)
    text = report.render_run(summary)
    (run_dir / "report.md").write_text(text)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(text)


def cmd_report(args) -> None:
    _report(_run_dir(args.run_dir))


def cmd_compare(args) -> None:
    runs = [report.load_run(_run_dir(p)) for p in args.run_dirs]
    text = report.render_comparison(runs)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


def cmd_audit(args) -> None:
    out = audit.write_audit(report.load_run(_run_dir(args.run_dir)))
    print(f"wrote {out}")


def cmd_regrade(args) -> None:
    out = runner.regrade(_run_dir(args.run_dir), testset.load())
    _report(out)


def cmd_selfcheck(args) -> None:
    ts = testset.load()
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for model, expect in (("oracle", 1.0), ("null-baseline", 0.0)):
            run_dir = runner.run(model, ts, workers=4, runs_root=Path(tmp), quiet=True)
            s = report.summarize(report.load_run(run_dir))
            acc = s["accuracy"]["rate"]
            good = acc == expect and s["coverage"]["scored"] == len(ts.cases)
            ok &= good
            print(f"{'PASS' if good else 'FAIL'}  {model:14s} accuracy {100 * acc:.1f}% (expected {100 * expect:.0f}%) "
                  f"over {s['coverage']['scored']} cases; failure reasons {s['failure_reasons']}")
    print(f"test set {ts.short_sha} ({len(ts.cases)} cases): harness {'OK' if ok else 'BROKEN'}")
    sys.exit(0 if ok else 1)


def cmd_prompt(args) -> None:
    case = testset.load().by_id().get(args.case_id)
    if case is None:
        raise SystemExit(f"unknown case {args.case_id}")
    system, user = prompt.build(case.db_path, case.question)
    print(f"--- system ({prompt.PROMPT_VERSION}, {prompt.prompt_fingerprint()}) ---\n{system}\n\n--- user ---\n{user}")


def cmd_models(args) -> None:
    for name, cfg in load_roster().items():
        extra = {k: v for k, v in cfg.items() if k != "provider"}
        env_var = KEY_ENV_VARS.get(cfg["provider"])
        key = "no key needed" if env_var is None else f"{env_var} {'set' if os.environ.get(env_var) else 'MISSING'}"
        print(f"{name:16s} {cfg['provider']:10s} {key:28s} {extra}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m harness", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run a model over the test set")
    p.add_argument("model", help="name from models.yaml")
    p.add_argument("--reps", type=int, default=1, help="repetitions per case (default 1)")
    p.add_argument("--workers", type=int, default=8, help="concurrent requests (default 8)")
    p.add_argument("--cases", help="comma-separated case ids (smoke tests)")
    p.add_argument("--limit", type=int, help="only the first N cases (smoke tests)")
    p.add_argument("--resume", metavar="RUN_DIR", help="continue an existing run directory")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="metrics for one run")
    p.add_argument("run_dir")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("compare", help="compare runs; the first is the baseline")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--out", help="also write the markdown here")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("audit", help="write the adjudication sheet for a run")
    p.add_argument("run_dir")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("regrade", help="re-grade a run with the current grader and cases.yaml")
    p.add_argument("run_dir")
    p.set_defaults(func=cmd_regrade)

    p = sub.add_parser("selfcheck", help="oracle and null-baseline sanity runs")
    p.set_defaults(func=cmd_selfcheck)

    p = sub.add_parser("prompt", help="print the prompt for a case")
    p.add_argument("case_id")
    p.set_defaults(func=cmd_prompt)

    p = sub.add_parser("models", help="list models.yaml")
    p.set_defaults(func=cmd_models)

    args = ap.parse_args(argv)
    load_dotenv()
    args.func(args)


if __name__ == "__main__":
    main()
