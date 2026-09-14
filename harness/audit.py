"""Human adjudication sheet for a run.

Lists every failed row (grouped by grader reason), every lenient-only pass, and every
paraphrase flip, with both queries and a preview of both results side by side. Tick a
box per item; where the grader was wrong, fix it in cases.yaml (alt_gold_sql / notes) or
in the grader, then `python -m harness regrade <run>` - no model calls needed.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .grader import REASONS
from .report import SCORED_STATUSES, Run


def _table(columns: list[str], preview: list, total: int | None) -> str:
    if total is None:
        return "_(no result)_"
    if not columns:
        return f"_{total} rows_"
    head = "| " + " | ".join(str(c) for c in columns) + " |"
    sep = "|" + "---|" * len(columns)
    body = ["| " + " | ".join("NULL" if v is None else str(v).replace("|", "\\|") for v in row) + " |" for row in preview]
    shown = f"first {len(preview)} of {total} rows" if total > len(preview) else f"{total} rows"
    return "\n".join([head, sep, *body]) + f"\n\n_{shown}_"


def _item(r: dict) -> list[str]:
    g = r["grade"]
    pair = f" · pair {r['pair_id']}" if r.get("pair_id") else ""
    flags = f" · flags: {', '.join(g['flags'])}" if g.get("flags") else ""
    status = f" · status **{r['status']}**" if r["status"] != "ok" else ""
    return [
        f"### {r['case_id']} r{r['rep']} · {r['difficulty']} · `{r['reason']}` · {r['db_id']}{pair}{status}{flags}",
        "",
        f"**Question:** {r['question']}",
        "",
        "**Gold SQL**",
        "```sql",
        r["gold_sql"],
        "```",
        "**Predicted SQL**",
        "```sql",
        r.get("sql") or "-- (none)",
        "```",
        *([f"Grader detail: `{g['detail']}`", ""] if g.get("detail") else []),
        "<table><tr><th>gold</th><th>predicted</th></tr><tr><td>",
        "",
        _table(g.get("gold_columns") or [], g.get("gold_preview") or [], g.get("gold_row_count")),
        "",
        "</td><td>",
        "",
        _table(g.get("pred_columns") or [], g.get("pred_preview") or [], g.get("pred_row_count")),
        "",
        "</td></tr></table>",
        "",
        "- [ ] grader verdict is right",
        "- [ ] grader verdict is wrong - fix: ",
        "",
    ]


def render_audit(run: Run) -> str:
    scored = [r for r in run.rows if r["status"] in SCORED_STATUSES]
    failures = defaultdict(list)
    for r in scored:
        if not r["passed"]:
            failures[r["reason"]].append(r)
    lenient_only = [r for r in scored if not r["passed"] and r["lenient_passed"]]
    unscored = [r for r in run.rows if r["status"] not in SCORED_STATUSES]

    halves = defaultdict(list)
    for r in scored:
        if r.get("pair_id"):
            halves[(r["pair_id"], r["rep"])].append(r)
    flips = [v for v in halves.values() if len(v) == 2 and v[0]["passed"] != v[1]["passed"]]

    n_fail = sum(len(v) for v in failures.values())
    lines = [
        f"# Adjudication sheet - {run.name} ({run.manifest.get('run_id')})",
        "",
        f"{n_fail} failures across {len(scored)} scored rows · {len(lenient_only)} would pass with extra columns allowed · "
        f"{len(flips)} paraphrase flips · {len(unscored)} truncated/unparseable",
        "",
        "Goal: confirm each failure is a model error, not a grader or gold error. "
        "If more than ~1 in 10 are grader errors, fix the grader before trusting any comparison.",
        "",
        "## Contents",
        "",
    ]
    for reason, rows in sorted(failures.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"- `{reason}` ({len(rows)}) - {REASONS.get(reason, '')}")
    lines.append("")

    for reason, rows in sorted(failures.items(), key=lambda kv: -len(kv[1])):
        lines += [f"## `{reason}` ({len(rows)})", "", REASONS.get(reason, ""), ""]
        for r in sorted(rows, key=lambda r: (r["case_id"], r["rep"])):
            lines += _item(r)

    if flips:
        lines += ["## Paraphrase flips", "",
                  "One wording passed and the other failed. Check whether the questions really ask the same thing.", ""]
        for a, b in sorted(flips, key=lambda v: v[0]["case_id"]):
            ok, bad = (a, b) if a["passed"] else (b, a)
            lines += [f"- **{ok['pair_id']}** r{ok['rep']}: {ok['case_id']} passed - \"{ok['question']}\"; "
                      f"{bad['case_id']} failed (`{bad['reason']}`) - \"{bad['question']}\""]
        lines.append("")

    if unscored:
        lines += ["## Truncated / unparseable (not in accuracy)", ""]
        for r in unscored:
            lines.append(f"- {r['case_id']} r{r['rep']} · {r['status']} · stop_reason={r.get('stop_reason')} · "
                         f"output tokens={(r.get('usage') or {}).get('output_tokens')}")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_audit(run: Run) -> Path:
    out = run.path / "audit.md"
    out.write_text(render_audit(run))
    return out
