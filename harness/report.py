"""Metrics for one run and comparisons across runs. Everything is recomputed from the raw
rows in results.jsonl - there is no stored aggregate to drift out of sync.

Denominators:
  accuracy         passed / scored, where scored = status "ok" or "refusal". A refusal is
                   the model's answer, so it counts as not passed (and is also listed).
  strict accuracy  passed / every row with model output, so truncated and unparseable
                   responses count as failures too. If the two numbers differ, look at why.
Infra errors (errors.jsonl) are never in either denominator; they show up as coverage.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .grader import REASONS
from .runner import read_jsonl
from .testset import DIFFICULTIES

SCORED_STATUSES = {"ok", "refusal"}
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0


@dataclass
class Run:
    path: Path
    manifest: dict
    rows: list[dict]
    errors: list[dict]

    @property
    def name(self) -> str:
        return self.manifest.get("model_name", self.path.name)


def load_run(path: Path) -> Run:
    manifest = json.loads((path / "manifest.json").read_text())
    rows, seen = [], set()
    for r in read_jsonl(path / "results.jsonl"):
        key = (r["case_id"], r["rep"])
        if key not in seen:  # first result for a slot wins
            seen.add(key)
            rows.append(r)
    return Run(path=path, manifest=manifest, rows=rows, errors=read_jsonl(path / "errors.jsonl"))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def cluster_bootstrap_ci(values_by_cluster: dict[str, list[float]], alpha: float = 0.05) -> tuple[float, float] | None:
    """Percentile CI for the mean, resampling whole clusters (a pair's two questions, and
    all reps of a case, move together)."""
    clusters = [v for v in values_by_cluster.values() if v]
    if len(clusters) < 2:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    k = len(clusters)
    means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        total = count = 0
        for _ in range(k):
            c = clusters[rng.randrange(k)]
            total += sum(c)
            count += len(c)
        means.append(total / count)
    means.sort()
    lo = means[int(alpha / 2 * BOOTSTRAP_SAMPLES)]
    hi = means[int((1 - alpha / 2) * BOOTSTRAP_SAMPLES) - 1]
    return lo, hi


def _rate(rows: list[dict], field: str = "passed") -> dict:
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_cluster[r.get("cluster") or r["case_id"]].append(1.0 if r[field] else 0.0)
    n = len(rows)
    mean = sum(1 for r in rows if r[field]) / n if n else None
    return {"n": n, "rate": mean, "ci": cluster_bootstrap_ci(by_cluster) if n else None}


def _percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def summarize(run: Run) -> dict:
    rows = run.rows
    scored = [r for r in rows if r["status"] in SCORED_STATUSES]
    have = {(r["case_id"], r["rep"]) for r in rows}
    unresolved_errors = [e for e in run.errors if (e["case_id"], e["rep"]) not in have]
    selection = run.manifest.get("selection", {}).get("case_ids") or []
    expected = len(selection) * run.manifest.get("reps", 1)

    summary = {
        "run_id": run.manifest.get("run_id"),
        "model_name": run.name,
        "model": run.manifest.get("model_config", {}).get("model") or run.manifest.get("model_config", {}).get("provider"),
        "status": run.manifest.get("status"),
        "testset_sha": (run.manifest.get("testset_sha256") or "")[:12],
        "grader_version": run.manifest.get("grader_version"),
        "prompt_version": run.manifest.get("prompt_version"),
        "reps": run.manifest.get("reps", 1),
        "coverage": {"expected": expected, "with_output": len(rows), "scored": len(scored),
                     "unresolved_infra_errors": len(unresolved_errors)},
        "statuses": dict(Counter(r["status"] for r in rows)),
        "accuracy": _rate(scored),
        "strict_accuracy": _rate(rows),
        "lenient_accuracy": _rate(scored, "lenient_passed"),
        "by_difficulty": {d: _rate([r for r in scored if r["difficulty"] == d]) for d in DIFFICULTIES},
        "failure_reasons": dict(Counter(r["reason"] for r in scored if not r["passed"]).most_common()),
        "infra_error_kinds": dict(Counter(e["kind"] for e in unresolved_errors)),
        "pairs": _pair_consistency(scored),
        "cost": _cost(rows),
        "latency_s": {
            "p50": _percentile([r["latency_s"] for r in rows if r.get("latency_s") is not None], 0.5),
            "p95": _percentile([r["latency_s"] for r in rows if r.get("latency_s") is not None], 0.95),
        },
        "retried_rows": sum(1 for r in rows if (r.get("attempts") or 1) > 1),
    }
    if summary["reps"] > 1:
        per_rep = defaultdict(list)
        for r in scored:
            per_rep[r["rep"]].append(r["passed"])
        accs = [sum(v) / len(v) for v in per_rep.values() if v]
        summary["rep_accuracy_stdev"] = statistics.stdev(accs) if len(accs) > 1 else None
    return summary


def _pair_consistency(scored: list[dict]) -> dict:
    """A pair flips when one paraphrase passes and the other fails (within the same rep)."""
    halves = defaultdict(list)
    for r in scored:
        if r.get("pair_id"):
            halves[(r["pair_id"], r["rep"])].append(r["passed"])
    complete = {k: v for k, v in halves.items() if len(v) == 2}
    if not complete:
        return {"pairs": 0}
    flips_by_pair: dict[str, list[float]] = defaultdict(list)
    both = neither = 0
    for (pair_id, _), (a, b) in complete.items():
        flips_by_pair[pair_id].append(1.0 if a != b else 0.0)
        both += a and b
        neither += (not a) and (not b)
    n = len(complete)
    flips = sum(sum(v) for v in flips_by_pair.values())
    return {"pairs": n, "flip_rate": flips / n, "flip_ci": cluster_bootstrap_ci(flips_by_pair),
            "both_pass": both / n, "both_fail": neither / n}


def _cost(rows: list[dict]) -> dict:
    costs = [r.get("cost_usd") for r in rows]
    known = [c for c in costs if c is not None]
    usage = Counter()
    for r in rows:
        usage.update({k: v for k, v in (r.get("usage") or {}).items() if isinstance(v, int)})
    return {
        "total_usd": sum(known) if known else None,
        "per_case_usd": (sum(known) / len(known)) if known else None,
        "rows_without_price": len(costs) - len(known),
        "tokens": dict(usage),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _pct(rate: dict | float | None, with_ci: bool = True) -> str:
    if isinstance(rate, dict):
        if rate.get("rate") is None:
            return "-"
        s = f"{100 * rate['rate']:.1f}%"
        if with_ci and rate.get("ci"):
            lo, hi = rate["ci"]
            s += f" [{100 * lo:.0f}-{100 * hi:.0f}]"
        return s
    return "-" if rate is None else f"{100 * rate:.1f}%"


def _usd(x: float | None, digits: int = 2) -> str:
    return "-" if x is None else f"${x:,.{digits}f}"


def render_run(summary: dict) -> str:
    s = summary
    cov = s["coverage"]
    lines = [
        f"# {s['model_name']} - {s['run_id']}",
        "",
        f"model `{s['model']}` · test set `{s['testset_sha']}` · prompt {s['prompt_version']} · "
        f"grader v{s['grader_version']} · reps {s['reps']} · status **{s['status']}**",
        "",
        f"Coverage: {cov['scored']} scored / {cov['with_output']} with output / {cov['expected']} expected"
        + (f" · **{cov['unresolved_infra_errors']} unresolved infra errors** {s['infra_error_kinds']}"
           if cov["unresolved_infra_errors"] else ""),
        "",
        "| metric | value (95% CI) |",
        "|---|---|",
        f"| accuracy | {_pct(s['accuracy'])} |",
        f"| strict accuracy (truncated/unparseable = fail) | {_pct(s['strict_accuracy'])} |",
        f"| lenient accuracy (extra columns allowed) | {_pct(s['lenient_accuracy'])} |",
    ]
    for d in DIFFICULTIES:
        r = s["by_difficulty"][d]
        lines.append(f"| {d} (n={r['n']}) | {_pct(r)} |")
    p = s["pairs"]
    if p.get("pairs"):
        ci = f" [{100 * p['flip_ci'][0]:.0f}-{100 * p['flip_ci'][1]:.0f}]" if p.get("flip_ci") else ""
        lines.append(f"| paraphrase flip rate ({p['pairs']} pairs) | {100 * p['flip_rate']:.1f}%{ci} |")
    c = s["cost"]
    lines += [
        f"| cost | {_usd(c['total_usd'], 4)} total · {_usd(c['per_case_usd'], 5)}/case |",
        f"| latency | p50 {s['latency_s']['p50'] or 0:.2f}s · p95 {s['latency_s']['p95'] or 0:.2f}s |",
        "",
        f"Statuses: {s['statuses']} · rows needing retries: {s['retried_rows']}",
        "",
        "Failure reasons:",
        "",
    ]
    for reason, n in s["failure_reasons"].items():
        lines.append(f"- `{reason}` {n} - {REASONS.get(reason, '')}")
    if not s["failure_reasons"]:
        lines.append("- none")
    if s.get("rep_accuracy_stdev") is not None:
        lines += ["", f"Accuracy stdev across reps: {100 * s['rep_accuracy_stdev']:.1f} points"]
    tokens = c["tokens"]
    if tokens:
        lines += ["", "Tokens: " + ", ".join(f"{k}={v:,}" for k, v in tokens.items())]
    return "\n".join(lines) + "\n"


def _paired_diff(base: Run, other: Run, difficulty: str | None = None) -> dict | None:
    """Per-case mean pass rate of `other` minus `base`, over cases scored in both."""

    def per_case(run: Run):
        acc = defaultdict(list)
        meta = {}
        for r in run.rows:
            if r["status"] in SCORED_STATUSES and (difficulty is None or r["difficulty"] == difficulty):
                acc[r["case_id"]].append(1.0 if r["passed"] else 0.0)
                meta[r["case_id"]] = r.get("cluster") or r["case_id"]
        return {k: sum(v) / len(v) for k, v in acc.items()}, meta

    a, meta = per_case(base)
    b, _ = per_case(other)
    common = sorted(set(a) & set(b))
    if not common:
        return None
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for cid in common:
        by_cluster[meta[cid]].append(b[cid] - a[cid])
    diff = sum(b[c] - a[c] for c in common) / len(common)
    return {"n": len(common), "diff": diff, "ci": cluster_bootstrap_ci(by_cluster)}


def render_comparison(runs: list[Run]) -> str:
    summaries = [summarize(r) for r in runs]
    lines = ["# Model comparison", ""]
    for key, label in (("testset_sha256", "test set"), ("grader_version", "grader"), ("prompt_fingerprint", "prompt")):
        values = {r.manifest.get(key) for r in runs}
        if len(values) > 1:
            lines.append(f"> **Warning:** runs differ in {label} ({', '.join(str(v)[:12] for v in values)}); "
                         "differences below mix model effects with that change.")
    selections = {tuple(r.manifest.get("selection", {}).get("case_ids") or []) for r in runs}
    if len(selections) > 1:
        lines.append("> **Warning:** runs cover different case selections; paired differences use the overlap.")
    if len(lines) > 2:
        lines.append("")

    header = "| model | accuracy | " + " | ".join(DIFFICULTIES) + " | flip rate | strict | $ / case | total $ | p50 latency |"
    lines += [header, "|" + "---|" * (header.count("|") - 1)]
    for s in summaries:
        flip = s["pairs"].get("flip_rate")
        lines.append(
            f"| {s['model_name']} | {_pct(s['accuracy'])} | "
            + " | ".join(_pct(s["by_difficulty"][d]) for d in DIFFICULTIES)
            + f" | {_pct(flip)} | {_pct(s['strict_accuracy'], False)} | {_usd(s['cost']['per_case_usd'], 5)} | "
            f"{_usd(s['cost']['total_usd'], 3)} | {(s['latency_s']['p50'] or 0):.2f}s |"
        )

    base, base_summary = runs[0], summaries[0]
    if len(runs) > 1:
        lines += ["", f"Paired difference vs **{base.name}** (accuracy points, 95% CI; "
                      "an interval containing 0 is not a detectable difference at this sample size):", ""]
        header = "| model | overall | " + " | ".join(DIFFICULTIES) + " | cost ratio |"
        lines += [header, "|" + "---|" * (header.count("|") - 1)]
        for run, s in zip(runs[1:], summaries[1:]):
            cells = []
            for d in (None, *DIFFICULTIES):
                pd = _paired_diff(base, run, d)
                if pd is None:
                    cells.append("-")
                    continue
                ci = f" [{100 * pd['ci'][0]:+.0f}, {100 * pd['ci'][1]:+.0f}]" if pd["ci"] else ""
                cells.append(f"{100 * pd['diff']:+.1f}{ci}")
            b_cost, o_cost = base_summary["cost"]["per_case_usd"], s["cost"]["per_case_usd"]
            ratio = f"{o_cost / b_cost:.1f}x" if b_cost and o_cost is not None else "-"
            lines.append(f"| {s['model_name']} | " + " | ".join(cells) + f" | {ratio} |")
    return "\n".join(lines) + "\n"
