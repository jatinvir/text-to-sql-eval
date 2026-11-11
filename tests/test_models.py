"""Each API provider against a mock HTTP server, through the vendor's real SDK: request
shape, usage normalization, retries, fatal errors, served-model check, and statuses."""

import json

import anthropic
import httpx  # google-genai uses httpx; anthropic and openai use httpx2
import httpx2
import openai
import pytest

from harness import models, pricing
from harness.models import AnthropicModel, FatalError, GeminiModel, InfraError, OpenAIModel
from harness.testset import Case

CASE = Case(case_id="c1", pair_id=None, difficulty="easy", db_id="x", source_index=0,
            question="How many?", gold_sql="SELECT count(*) FROM t")


def message(text='{"sql": "SELECT 1"}', model="claude-opus-5", stop_reason="end_turn", thinking=False):
    content = [{"type": "text", "text": text}]
    if thinking:
        content.insert(0, {"type": "thinking", "thinking": "", "signature": "sig"})
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": content,
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 1000, "output_tokens": 200},
    }


def make(responses, cfg=None, requests=None):
    """responses: list of (status, body) returned in order."""
    queue = list(responses)

    def handler(request):
        if requests is not None:
            requests.append(json.loads(request.content))
        status, body = queue.pop(0)
        return httpx2.Response(status, json=body, headers={"request-id": "req_test"})

    client = anthropic.Anthropic(api_key="test", base_url="http://mock", max_retries=0,
                                 http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    config = {"provider": "anthropic", "model": "claude-opus-5", "max_tokens": 16000, "effort": "high", **(cfg or {})}
    return AnthropicModel("test", config, client=client)


def error(kind, msg="nope"):
    return {"type": "error", "error": {"type": kind, "message": msg}}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(models.time, "sleep", lambda s: None)


def test_success_parses_sql_usage_and_cost():
    requests = []
    m = make([(200, message(thinking=True))], requests=requests)
    c = m.complete(CASE, "system prompt", "Question: How many?")
    assert c.status == "ok" and c.sql == "SELECT 1"
    assert c.usage["input_tokens"] == 1000 and c.usage["output_tokens"] == 200
    assert c.cost_usd == pytest.approx((1000 * 5 + 200 * 25) / 1e6)
    assert c.attempts == 1 and c.request_id == "req_test"
    body = requests[0]
    assert body["output_config"]["effort"] == "high"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["system"] == "system prompt" and "thinking" not in body and "temperature" not in body


def test_retries_transient_errors_and_counts_attempts():
    m = make([(429, error("rate_limit_error")), (529, error("overloaded_error")), (200, message())])
    c = m.complete(CASE, "s", "u")
    assert c.status == "ok" and c.attempts == 3


def test_gives_up_after_max_attempts_as_infra_error():
    m = make([(500, error("api_error"))] * 3, cfg={"max_attempts": 3})
    with pytest.raises(InfraError) as exc:
        m.complete(CASE, "s", "u")
    assert exc.value.kind == "api_error" and exc.value.attempts == 3


@pytest.mark.parametrize("status,kind", [(401, "authentication_error"), (404, "not_found_error"), (400, "invalid_request_error")])
def test_config_errors_are_fatal(status, kind):
    with pytest.raises(FatalError):
        make([(status, error(kind))]).complete(CASE, "s", "u")


def test_served_model_mismatch_is_infra_error():
    with pytest.raises(InfraError) as exc:
        make([(200, message(model="claude-sonnet-5"))]).complete(CASE, "s", "u")
    assert exc.value.kind == "served_model_mismatch"


def test_alias_resolving_to_snapshot_is_accepted():
    m = make([(200, message(model="claude-haiku-4-5-20251001"))], cfg={"model": "claude-haiku-4-5", "effort": None})
    assert m.complete(CASE, "s", "u").status == "ok"


def test_refusal_truncation_and_unparseable_are_distinct_statuses():
    assert make([(200, message(stop_reason="refusal", text=""))]).complete(CASE, "s", "u").status == "refusal"
    assert make([(200, message(stop_reason="max_tokens", text='{"sql": "SEL'))]).complete(CASE, "s", "u").status == "truncated"
    assert make([(200, message(text="SELECT 1"))]).complete(CASE, "s", "u").status == "unparseable"


def test_roster_is_complete():
    api_providers = {"anthropic", "openai", "gemini"}
    for name, cfg in models.load_roster().items():
        assert cfg["provider"] in api_providers | {"oracle", "null"}, name
        if cfg["provider"] in api_providers:
            assert cfg["model"] in pricing.PRICES_PER_MTOK, f"{name}: no price, cost would be unknown"
            assert cfg["max_tokens"] == 16000, f"{name}: every model gets the same output budget"
        else:
            assert models.make_model(name).complete(CASE, "s", "u").status == "ok"


# ---------------------------------------------------------------------------
# OpenAI (Responses API) through the real SDK
# ---------------------------------------------------------------------------

def oai_response(text='{"sql": "SELECT 1"}', model="gpt-5.6-sol-2026-08-01", status="completed",
                 incomplete=None, refusal=False):
    content = ([{"type": "refusal", "refusal": "I can't help with that."}] if refusal
               else [{"type": "output_text", "text": text, "annotations": []}])
    return {
        "id": "resp_1", "object": "response", "created_at": 0, "status": status, "model": model,
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": content},
        ],
        "incomplete_details": {"reason": incomplete} if incomplete else None,
        "error": None, "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 200},
                  "output_tokens": 300, "output_tokens_details": {"reasoning_tokens": 250}, "total_tokens": 1300},
    }


def make_openai(responses, cfg=None, requests=None):
    queue = list(responses)

    def handler(request):
        if requests is not None:
            requests.append(json.loads(request.content))
        status, body = queue.pop(0)
        return httpx2.Response(status, json=body, headers={"x-request-id": "req_oai"})

    client = openai.OpenAI(api_key="test", base_url="http://mock/v1", max_retries=0,
                           http_client=openai.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    config = {"provider": "openai", "model": "gpt-5.6-sol", "max_tokens": 16000, "reasoning_effort": "high", **(cfg or {})}
    return OpenAIModel("test-openai", config, client=client)


def oai_error(status, code="server_error"):
    return status, {"error": {"message": "nope", "type": code, "code": code, "param": None}}


def test_openai_success_request_shape_and_normalized_usage():
    requests = []
    c = make_openai([(200, oai_response())], requests=requests).complete(CASE, "system prompt", "Question: How many?")
    assert c.status == "ok" and c.sql == "SELECT 1" and c.served_model == "gpt-5.6-sol-2026-08-01"
    # 1000 input incl. 200 cached -> 800 full-price + 200 cached; reasoning is part of output.
    assert c.usage == {"input_tokens": 800, "output_tokens": 300, "cache_read_input_tokens": 200,
                       "cache_creation_input_tokens": 0, "reasoning_tokens": 250}
    assert c.cost_usd == pytest.approx((800 * 4 + 200 * 0.4 + 300 * 20) / 1e6)
    body = requests[0]
    assert body["instructions"] == "system prompt" and body["input"] == "Question: How many?"
    assert body["text"]["format"]["type"] == "json_schema" and body["text"]["format"]["strict"] is True
    assert body["reasoning"] == {"effort": "high"} and body["max_output_tokens"] == 16000 and body["store"] is False


def test_openai_retries_then_fatal_and_statuses():
    c = make_openai([oai_error(429, "rate_limit_exceeded"), oai_error(503), (200, oai_response())]).complete(CASE, "s", "u")
    assert c.attempts == 3 and c.status == "ok"
    with pytest.raises(FatalError):
        make_openai([oai_error(401, "invalid_api_key")]).complete(CASE, "s", "u")
    with pytest.raises(InfraError) as exc:
        make_openai([(200, oai_response(model="gpt-5.6-luna"))]).complete(CASE, "s", "u")
    assert exc.value.kind == "served_model_mismatch"
    assert make_openai([(200, oai_response(refusal=True))]).complete(CASE, "s", "u").status == "refusal"
    truncated = oai_response(text='{"sql": "SEL', status="incomplete", incomplete="max_output_tokens")
    assert make_openai([(200, truncated)]).complete(CASE, "s", "u").status == "truncated"


# ---------------------------------------------------------------------------
# Gemini (google-genai generate_content) through the real SDK
# ---------------------------------------------------------------------------


def gem_response(text='{"sql": "SELECT 1"}', model="gemini-3.8-flash", finish="STOP", thought=True, block=None):
    parts = ([{"text": "thinking about joins", "thought": True}] if thought else []) + [{"text": text}]
    body = {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}],
        "usageMetadata": {"promptTokenCount": 1000, "cachedContentTokenCount": 100,
                          "candidatesTokenCount": 40, "thoughtsTokenCount": 400, "totalTokenCount": 1440},
        "modelVersion": model, "responseId": "gem_resp_1",
    }
    if block:
        body = {"promptFeedback": {"blockReason": block}, "usageMetadata": {"promptTokenCount": 1000},
                "modelVersion": model, "responseId": "gem_resp_1"}
    return body


def make_gemini(responses, cfg=None, requests=None):
    from google import genai
    from google.genai import types

    queue = list(responses)

    def handler(request):
        if requests is not None:
            requests.append((str(request.url), json.loads(request.content)))
        status, body = queue.pop(0)
        return httpx.Response(status, json=body)

    client = genai.Client(api_key="test", http_options=types.HttpOptions(
        base_url="http://mock", httpx_client=httpx.Client(transport=httpx.MockTransport(handler))))
    config = {"provider": "gemini", "model": "gemini-3.8-flash", "max_tokens": 16000, "thinking_level": "high", **(cfg or {})}
    return GeminiModel("test-gemini", config, client=client)


def gem_error(status, name="UNAVAILABLE"):
    return status, {"error": {"code": status, "message": "nope", "status": name}}


def test_gemini_success_request_shape_and_normalized_usage():
    requests = []
    c = make_gemini([(200, gem_response())], requests=requests).complete(CASE, "system prompt", "Question: How many?")
    assert c.status == "ok" and c.sql == "SELECT 1", c.raw_text  # thought parts are not the answer
    assert c.usage == {"input_tokens": 900, "output_tokens": 440, "cache_read_input_tokens": 100,
                       "cache_creation_input_tokens": 0, "reasoning_tokens": 400}
    assert c.cost_usd == pytest.approx((900 * 0.75 + 100 * 0.075 + 440 * 3.75) / 1e6)
    assert c.request_id == "gem_resp_1"
    url, body = requests[0]
    assert "gemini-3.8-flash:generateContent" in url
    gen = body["generationConfig"]
    assert gen["responseMimeType"] == "application/json" and gen["maxOutputTokens"] == 16000
    assert gen["responseJsonSchema"]["required"] == ["sql"]
    # google-genai sends this one field in proto snake_case; the API accepts either spelling.
    thinking = gen["thinkingConfig"]
    assert thinking.get("thinkingLevel", thinking.get("thinking_level")) == "HIGH"
    assert body["systemInstruction"]["parts"][0]["text"] == "system prompt"


def test_gemini_retries_then_fatal_and_statuses():
    c = make_gemini([gem_error(429, "RESOURCE_EXHAUSTED"), gem_error(503), (200, gem_response())]).complete(CASE, "s", "u")
    assert c.attempts == 3 and c.status == "ok"
    with pytest.raises(FatalError):
        make_gemini([gem_error(400, "INVALID_ARGUMENT")]).complete(CASE, "s", "u")
    with pytest.raises(InfraError) as exc:
        make_gemini([(200, gem_response(model="gemini-2.5-flash"))]).complete(CASE, "s", "u")
    assert exc.value.kind == "served_model_mismatch"
    assert make_gemini([(200, gem_response(finish="SAFETY", text=""))]).complete(CASE, "s", "u").status == "refusal"
    assert make_gemini([(200, gem_response(block="SAFETY"))]).complete(CASE, "s", "u").status == "refusal"
    assert make_gemini([(200, gem_response(finish="MAX_TOKENS", text='{"sql'))]).complete(CASE, "s", "u").status == "truncated"


def test_gemini_transport_timeout_is_infra_error():
    from google import genai
    from google.genai import types

    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    client = genai.Client(api_key="test", http_options=types.HttpOptions(
        base_url="http://mock", httpx_client=httpx.Client(transport=httpx.MockTransport(handler))))
    m = GeminiModel("g", {"provider": "gemini", "model": "gemini-3.8-flash", "max_tokens": 16000, "max_attempts": 2},
                    client=client)
    with pytest.raises(InfraError) as exc:
        m.complete(CASE, "s", "u")
    assert exc.value.kind == "timeout" and exc.value.attempts == 2


@pytest.mark.parametrize("requested,served,ok", [
    ("claude-haiku-4-5", "claude-haiku-4-5-20251001", True),
    ("gpt-5.6-sol", "gpt-5.6-sol-2026-08-01", True),
    ("gemini-3.8-flash", "models/gemini-3.8-flash-001", True),
    ("claude-fable-5", "claude-fable-5-1", False),
    ("gpt-5.6", "gpt-5.6-luna", False),
    ("gpt-5.6-sol", "gpt-5.6-sol-pro", False),
])
def test_served_model_check_only_accepts_snapshots(requested, served, ok):
    m = models.APIModel("x", {"model": requested})
    if ok:
        assert m._check_served(served, 1)
    else:
        with pytest.raises(InfraError):
            m._check_served(served, 1)


def test_unexpected_sdk_errors_and_unreadable_responses_are_infra_errors():
    class Broken(models.APIModel):
        def _call(self, system, user):
            raise ValueError("SDK could not validate the response")

        def _classify(self, e):
            return None

    with pytest.raises(InfraError) as exc:
        Broken("b", {"model": "m"}).complete(CASE, "s", "u")
    assert exc.value.kind == "provider_error"

    class OddShape(models.APIModel):
        def _call(self, system, user):
            return {"surprise": True}

        def _to_completion(self, resp, latency, attempts):
            return resp.candidates[0]

    with pytest.raises(InfraError) as exc:
        OddShape("o", {"model": "m"}).complete(CASE, "s", "u")
    assert exc.value.kind == "response_parse_error" and "surprise" in str(exc.value)


def test_openai_failed_response_is_infra_error():
    failed = oai_response(text="", status="failed")
    failed["error"] = {"code": "server_error", "message": "boom"}
    with pytest.raises(InfraError) as exc:
        make_openai([(200, failed)]).complete(CASE, "s", "u")
    assert exc.value.kind == "api_error"


def test_gemini_empty_response_without_reason_is_infra_error():
    with pytest.raises(InfraError) as exc:
        make_gemini([(200, gem_response(finish="OTHER", text="", thought=False))]).complete(CASE, "s", "u")
    assert exc.value.kind == "api_error"


def test_missing_api_key_gives_actionable_message(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(SystemExit, match=r"OPENAI_API_KEY is set in \.env"):
        models.make_model("gpt-5.6-luna")


def test_out_of_credit_errors_stop_immediately_instead_of_retrying():
    body = {"error": {"message": "You have no credits remaining. Add credits to continue using the API.",
                      "type": "insufficient_quota", "param": None, "code": "credit_balance_exhausted"}}
    with pytest.raises(FatalError, match="out of credits"):
        make_openai([(429, body)] * 6).complete(CASE, "s", "u")

    spend_cap = {"type": "error", "error": {"type": "rate_limit_error", "message": "You have reached your API usage limits",
                                            "details": {"error_code": "enforced_spend_limit_reached"}}}
    with pytest.raises(FatalError, match="out of credits"):
        make([(429, spend_cap)] * 6).complete(CASE, "s", "u")

    with pytest.raises(FatalError, match="out of credits"):
        make_gemini([gem_error(429, "RESOURCE_EXHAUSTED")[0:1] + ({"error": {
            "code": 429, "message": "Your prepayment credits are depleted.", "status": "RESOURCE_EXHAUSTED"}},)] * 6
        ).complete(CASE, "s", "u")


def test_ordinary_gemini_rate_limit_is_still_retried():
    rate_limited = (429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message":
                    "You exceeded your current quota, please check your plan and billing details."}})
    c = make_gemini([rate_limited, (200, gem_response())]).complete(CASE, "s", "u")
    assert c.status == "ok" and c.attempts == 2
