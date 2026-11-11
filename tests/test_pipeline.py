"""Runner -> results -> report/audit/regrade, end to end, without spending money."""

import json
import time

import anthropic
import httpx2
import pytest

from harness import audit, models, report, runner, testset
from harness.models import FatalError, InfraError, OracleModel


@pytest.fixture(scope="module")
def ts():
    return testset.load()


def test_oracle_and_null_runs(ts, tmp_path):
    oracle_dir = runner.run("oracle", ts, runs_root=tmp_path, quiet=True)
    null_dir = runner.run("null-baseline", ts, runs_root=tmp_path, quiet=True)
    o, n = report.load_run(oracle_dir), report.load_run(null_dir)
    so, sn = report.summarize(o), report.summarize(n)
    assert so["accuracy"]["rate"] == 1.0 and so["coverage"]["scored"] == 150
    assert sn["accuracy"]["rate"] == 0.0
    assert so["pairs"]["pairs"] == 50 and so["pairs"]["flip_rate"] == 0.0
    manifest = json.loads((oracle_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["testset_sha256"] == ts.sha256

    text = report.render_comparison([o, n])
    assert "null-baseline" in text and "-100.0" in text
    assert "# Adjudication sheet" in audit.render_audit(n)
    report.render_run(sn)

    regraded = report.load_run(runner.regrade(null_dir, ts, runs_root=tmp_path))
    assert [r["passed"] for r in regraded.rows] == [r["passed"] for r in n.rows]


class Flaky(OracleModel):
    """Fails c002 with an infra error until `healed`."""

    healed = False

    def complete(self, case, system, user):
        if case.case_id == "c002" and not Flaky.healed:
            raise InfraError("api_error", "simulated outage")
        return super().complete(case, system, user)


def test_infra_errors_stay_out_of_results_and_resume_fills_them(ts, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "make_model", lambda name: Flaky(name, {"provider": "oracle"}))
    run_dir = runner.run("oracle", ts, limit=5, runs_root=tmp_path, quiet=True)
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "incomplete"
    assert len(runner.read_jsonl(run_dir / "results.jsonl")) == 4
    assert runner.read_jsonl(run_dir / "errors.jsonl")[0]["kind"] == "api_error"
    s = report.summarize(report.load_run(run_dir))
    assert s["accuracy"]["rate"] == 1.0 and s["coverage"]["unresolved_infra_errors"] == 1

    Flaky.healed = True
    runner.run("oracle", ts, limit=5, run_dir=run_dir, quiet=True)
    assert len(runner.read_jsonl(run_dir / "results.jsonl")) == 5
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "complete"
    assert report.summarize(report.load_run(run_dir))["coverage"]["unresolved_infra_errors"] == 0


def test_resume_refuses_a_different_experiment(ts, tmp_path):
    run_dir = runner.run("oracle", ts, limit=2, runs_root=tmp_path, quiet=True)
    with pytest.raises(SystemExit):
        runner.run("oracle", ts, limit=2, reps=2, run_dir=run_dir, quiet=True)


def test_resume_keeps_the_saved_case_selection(ts, tmp_path):
    run_dir = runner.run("oracle", ts, limit=2, runs_root=tmp_path, quiet=True)
    runner.run("oracle", ts, run_dir=run_dir, quiet=True)  # the printed resume command has no --limit
    assert len(runner.read_jsonl(run_dir / "results.jsonl")) == 2
    with pytest.raises(SystemExit, match="saved cases"):
        runner.run("oracle", ts, limit=5, run_dir=run_dir, quiet=True)


def test_runs_started_together_get_separate_directories(tmp_path):
    first = runner._new_run_dir(tmp_path, "oracle")
    second = runner._new_run_dir(tmp_path, "oracle")
    assert first != second and first.exists() and second.exists()


def test_interrupt_stops_queued_cases(ts, tmp_path, monkeypatch):
    calls = []

    class InterruptedOracle(OracleModel):
        def complete(self, case, system, user):
            calls.append(case.case_id)
            if len(calls) == 1:
                raise KeyboardInterrupt
            time.sleep(0.2)  # a real API call takes a while
            return super().complete(case, system, user)

    monkeypatch.setattr(runner, "make_model", lambda name: InterruptedOracle(name, {}))
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        runner.run("oracle", ts, limit=30, runs_root=tmp_path, quiet=True, workers=1)
    # At most the one request already in flight may finish; the other 28 must not run.
    assert len(calls) <= 2 and time.monotonic() - started < 2, \
        "queued cases kept running (and would be billed) after the interrupt"
    [run_dir] = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "interrupted"


def test_repeated_provider_errors_stop_the_run(ts, tmp_path, monkeypatch):
    calls = []

    class Garbled(OracleModel):
        def complete(self, case, system, user):
            calls.append(case.case_id)
            time.sleep(0.05)  # a real API call takes a while
            raise RuntimeError("response shape not understood")

    monkeypatch.setattr(runner, "make_model", lambda name: Garbled(name, {}))
    run_dir = runner.run("oracle", ts, limit=30, runs_root=tmp_path, quiet=True, workers=1)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "aborted" and "consecutive provider_error" in manifest["fatal_error"]
    # The breaker trips after 3; at most the request already in flight also completes.
    assert len(calls) <= runner.MAX_CONSECUTIVE_SYSTEMATIC_ERRORS + 1


def test_unresolved_errors_are_counted_per_slot(ts, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "make_model", lambda name: Flaky(name, {"provider": "oracle"}))
    Flaky.healed = False
    run_dir = runner.run("oracle", ts, limit=5, runs_root=tmp_path, quiet=True)
    runner.run("oracle", ts, run_dir=run_dir, quiet=True)  # c002 fails again on resume
    assert len(runner.read_jsonl(run_dir / "errors.jsonl")) == 2
    assert report.summarize(report.load_run(run_dir))["coverage"]["unresolved_infra_errors"] == 1


def test_fatal_error_aborts_run(ts, tmp_path, monkeypatch):
    class Broken(OracleModel):
        def complete(self, case, system, user):
            raise FatalError("AuthenticationError: bad key")

    monkeypatch.setattr(runner, "make_model", lambda name: Broken(name, {}))
    run_dir = runner.run("oracle", ts, limit=20, runs_root=tmp_path, quiet=True, workers=2)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "aborted" and "bad key" in manifest["fatal_error"]


def test_anthropic_provider_through_runner(ts, tmp_path, monkeypatch):
    """Mock API that answers with the gold SQL: exercises real SDK objects end to end."""
    gold_by_question = {c.question: c.gold_sql for c in ts.cases}

    def handler(request):
        body = json.loads(request.content)
        question = body["messages"][0]["content"].removeprefix("Question: ")
        return httpx2.Response(200, json={
            "id": "msg", "type": "message", "role": "assistant", "model": "claude-haiku-4-5-20251001",
            "content": [{"type": "text", "text": json.dumps({"sql": gold_by_question[question]})}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 900, "output_tokens": 40},
        })

    client = anthropic.Anthropic(api_key="test", base_url="http://mock", max_retries=0,
                                 http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    cfg = models.load_roster()["haiku-4.5"]
    monkeypatch.setattr(runner, "make_model", lambda name: models.AnthropicModel(name, cfg, client=client))
    run_dir = runner.run("haiku-4.5", ts, limit=12, reps=2, runs_root=tmp_path, quiet=True)
    s = report.summarize(report.load_run(run_dir))
    assert s["accuracy"]["rate"] == 1.0 and s["coverage"]["scored"] == 24
    assert s["cost"]["total_usd"] == pytest.approx(24 * (900 * 1 + 40 * 5) / 1e6)
    assert s["rep_accuracy_stdev"] == 0.0


def test_dotenv_loads_without_overriding(tmp_path, monkeypatch):
    import os

    from harness.__main__ import load_dotenv

    env = tmp_path / ".env"
    env.write_text('# comment\nexport OPENAI_API_KEY="sk-from-file"\nGEMINI_API_KEY=\nANTHROPIC_API_KEY=from-file\n')
    # setenv first so monkeypatch records a restore even when the variable wasn't set.
    for var in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(var, "placeholder")
        monkeypatch.delenv(var)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")
    load_dotenv(env)
    assert os.environ["OPENAI_API_KEY"] == "sk-from-file"
    assert "GEMINI_API_KEY" not in os.environ  # blank values are ignored
    assert os.environ["ANTHROPIC_API_KEY"] == "from-shell"
