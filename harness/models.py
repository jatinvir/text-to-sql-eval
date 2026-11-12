"""Model providers.

A provider turns (case, system prompt, user prompt) into a Completion. It must keep
plumbing failures (rate limits, outages, a different model answering) out of the
Completion: those raise InfraError and land in errors.jsonl, never in the score.

Every vendor gets the same system prompt, the same user message, the same JSON schema for
the answer, and the same output-token budget. What differs is only how each API spells
those things - that translation is all a provider class does.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field

import yaml

from . import pricing
from .paths import MODELS_YAML
from .prompt import OUTPUT_SCHEMA
from .testset import Case


@dataclass
class Completion:
    sql: str | None
    # "ok" | "refusal" | "truncated" (hit the output-token limit) | "unparseable"
    status: str
    raw_text: str = ""
    stop_reason: str | None = None
    served_model: str | None = None
    request_id: str | None = None
    # Normalized across vendors: input_tokens excludes cache reads; output_tokens includes
    # reasoning/thinking tokens (all vendors bill them as output); reasoning_tokens is the
    # part of output_tokens spent thinking, when the API reports it.
    usage: dict = field(default_factory=dict)
    cost_usd: float | None = None
    latency_s: float | None = None  # the final, successful request only - no backoff sleeps
    attempts: int = 1


class InfraError(Exception):
    """This attempt produced no scorable output for reasons outside the model's control."""

    def __init__(self, kind: str, message: str, attempts: int = 1):
        super().__init__(message)
        self.kind = kind
        self.attempts = attempts


class FatalError(Exception):
    """Configuration or credential problem that would fail every case - stop the run."""


# Where each provider's SDK looks for credentials (Anthropic also accepts an `ant auth login`
# profile; Gemini also accepts GOOGLE_API_KEY). Local servers need no key.
KEY_ENV_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}


def load_roster() -> dict[str, dict]:
    with open(MODELS_YAML) as f:
        return yaml.safe_load(f)["models"]


def make_model(name: str):
    roster = load_roster()
    if name not in roster:
        raise SystemExit(f"unknown model {name!r}; models.yaml defines: {', '.join(roster)}")
    cfg = roster[name]
    providers = {
        "anthropic": AnthropicModel,
        "openai": OpenAIModel,
        "gemini": GeminiModel,
        "local": LocalModel,
        "oracle": OracleModel,
        "null": NullModel,
    }
    provider = providers.get(cfg.get("provider"))
    if provider is None:
        raise SystemExit(f"model {name!r}: unknown provider {cfg.get('provider')!r}")
    try:
        return provider(name, cfg)
    except FatalError as e:
        raise SystemExit(f"could not set up {name!r}: {e}") from e
    except Exception as e:  # SDK clients raise on construction when credentials are missing
        env_var = KEY_ENV_VARS.get(cfg["provider"], "the API key")
        raise SystemExit(
            f"could not create a client for {name!r}: {type(e).__name__}: {e}\n"
            f"Check that {env_var} is set in .env (see .env.example), then run `python -m harness models`."
        ) from e


class OracleModel:
    """Answers with the gold SQL. Must score 100%; anything less is a harness bug."""

    def __init__(self, name: str, cfg: dict):
        self.name, self.config = name, cfg

    def complete(self, case: Case, system: str, user: str) -> Completion:
        return Completion(sql=case.gold_sql, status="ok", served_model="oracle", cost_usd=0.0, latency_s=0.0)


class NullModel:
    """Answers every question with an empty result. Must score ~0%; a pass means some case
    can be satisfied without answering the question."""

    SQL = "SELECT NULL WHERE 0"

    def __init__(self, name: str, cfg: dict):
        self.name, self.config = name, cfg

    def complete(self, case: Case, system: str, user: str) -> Completion:
        return Completion(sql=self.SQL, status="ok", served_model="null", cost_usd=0.0, latency_s=0.0)


# ---------------------------------------------------------------------------
# API providers
# ---------------------------------------------------------------------------

RETRY, FATAL = "retry", "fatal"

# Out-of-credit and spend-cap errors arrive as 429s that look like rate limits but never
# clear by waiting. OpenAI: insufficient_quota / credit_balance_exhausted; Anthropic:
# enforced_spend_limit_reached. Deliberately not "quota" or "billing": Gemini's ordinary
# per-minute rate-limit message mentions both.
_OUT_OF_CREDIT = re.compile(
    r"insufficient_quota|credit_balance_exhausted|enforced_spend_limit_reached|no credits"
    r"|credit balance|credits? (?:are |have been )?(?:depleted|exhausted)",
    re.I,
)


class APIModel:
    """Retry loop, timing, served-model check and answer parsing shared by every vendor.

    Subclasses implement:
        _call(system, user)          -> raw SDK response (raise on HTTP/transport errors)
        _classify(exc)               -> RETRY | FATAL | None (None re-raises: a harness bug)
        _is_timeout(exc)             -> bool
        _to_completion(resp, latency, attempts) -> Completion
    """

    def __init__(self, name: str, cfg: dict):
        self.name, self.config = name, cfg
        self.model = cfg["model"]
        self.max_tokens = int(cfg.get("max_tokens", 16000))
        self.max_attempts = int(cfg.get("max_attempts", 6))
        self.case_deadline_s = float(cfg.get("case_deadline_s", 900))
        self.request_timeout_s = float(cfg.get("request_timeout_s", 300))

    def complete(self, case: Case, system: str, user: str) -> Completion:
        started = time.monotonic()
        attempt = 0
        while True:
            attempt += 1
            t0 = time.monotonic()
            try:
                resp = self._call(system, user)
                latency = time.monotonic() - t0
                break
            except Exception as e:
                verdict = self._classify(e)
                if verdict == RETRY and _OUT_OF_CREDIT.search(f"{getattr(e, 'body', '')} {e}"):
                    raise FatalError(
                        f"out of credits or over the spend limit for {self.model} - add credits or raise the "
                        f"limit in that provider's billing settings, then resume. {type(e).__name__}: {e}"
                    ) from e
                if verdict == FATAL:
                    raise FatalError(f"{type(e).__name__}: {e}") from e
                if verdict != RETRY:
                    # Not a known transient or config error (e.g. the SDK couldn't validate a
                    # response). Record it; the runner stops the run if these keep happening.
                    raise InfraError("provider_error", f"{type(e).__name__}: {e}", attempts=attempt) from e
                last_error = e
            elapsed = time.monotonic() - started
            if attempt >= self.max_attempts or elapsed > self.case_deadline_s:
                kind = "timeout" if self._is_timeout(last_error) else "api_error"
                raise InfraError(kind, f"{type(last_error).__name__}: {last_error}", attempts=attempt)
            time.sleep(min(self._backoff(attempt, last_error), max(1.0, self.case_deadline_s - elapsed)))
        try:
            return self._to_completion(resp, latency, attempt)
        except InfraError:
            raise
        except Exception as e:  # a response shape this provider class doesn't understand
            raise InfraError(
                "response_parse_error", f"{type(e).__name__}: {e}; response: {repr(resp)[:2000]}", attempts=attempt
            ) from e

    @staticmethod
    def _retry_for_status(status: int) -> str:
        # 408 timeout, 409 conflict, 429 rate limit, 5xx server side: transient.
        # Other 4xx (bad request, auth, permission, unknown model) would fail every case.
        return RETRY if status >= 500 or status in (408, 409, 429) else FATAL

    @staticmethod
    def _backoff(attempt: int, error: Exception) -> float:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None) or {}
        retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
        if retry_after:
            try:
                return float(retry_after) + random.uniform(0, 1)
            except ValueError:
                pass
        return min(60.0, 2.0 ** attempt) * random.uniform(0.5, 1.5)

    # Version suffixes a vendor may append when an ID resolves to a snapshot:
    # -20251001 (Anthropic), -2026-08-01 (OpenAI), -001 / -09-2025 (Google).
    _SNAPSHOT_SUFFIX = re.compile(r"-(\d{8}|\d{4}-\d{2}-\d{2}|\d{3}|\d{2}-\d{4})")

    def _check_served(self, served: str | None, attempts: int) -> str:
        served = (served or "").removeprefix("models/")
        # Accept the exact ID or that ID plus a snapshot suffix - never a sibling model that
        # merely shares a prefix (claude-fable-5 vs claude-fable-5-1, gpt-5.6 vs gpt-5.6-luna).
        suffix = served[len(self.model):] if served.startswith(self.model) else None
        if served != self.model and not (suffix and self._SNAPSHOT_SUFFIX.fullmatch(suffix)):
            raise InfraError("served_model_mismatch", f"asked for {self.model}, served by {served or 'unknown'}", attempts)
        return served

    def _finish(self, *, text: str, refusal: bool, truncated: bool, usage: dict, **common) -> Completion:
        common.update(raw_text=text, usage=usage, cost_usd=pricing.cost_usd(self.model, usage))
        if refusal:
            return Completion(sql=None, status="refusal", **common)
        if truncated:
            return Completion(sql=None, status="truncated", **common)
        try:
            sql = json.loads(text)["sql"]
            if not isinstance(sql, str):
                raise TypeError("sql is not a string")
        except (ValueError, KeyError, TypeError):
            return Completion(sql=None, status="unparseable", **common)
        return Completion(sql=sql, status="ok", **common)


class AnthropicModel(APIModel):
    """Messages API with JSON-schema structured output."""

    def __init__(self, name: str, cfg: dict, client=None):
        super().__init__(name, cfg)
        import anthropic

        self._sdk = anthropic
        # SDK retries off so every attempt is visible to the harness.
        self.client = client or anthropic.Anthropic(max_retries=0, timeout=self.request_timeout_s)

    def request_params(self, system: str, user: str) -> dict:
        output_config: dict = {"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}}
        if self.config.get("effort"):
            output_config["effort"] = self.config["effort"]
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
        }
        if self.config.get("thinking"):
            params["thinking"] = self.config["thinking"]
        return params

    def _call(self, system, user):
        return self.client.messages.create(**self.request_params(system, user))

    def _classify(self, e):
        a = self._sdk
        if isinstance(e, a.APIStatusError):
            return self._retry_for_status(e.status_code)
        if isinstance(e, a.APIConnectionError):  # includes APITimeoutError
            return RETRY
        return None

    def _is_timeout(self, e):
        return isinstance(e, self._sdk.APITimeoutError)

    def _to_completion(self, resp, latency, attempts):
        served = self._check_served(resp.model, attempts)
        u = resp.usage
        usage = {
            "input_tokens": u.input_tokens or 0,
            "output_tokens": u.output_tokens or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", None) or 0,
        }
        return self._finish(
            text="".join(b.text for b in resp.content if b.type == "text"),
            refusal=resp.stop_reason == "refusal",
            truncated=resp.stop_reason == "max_tokens",
            usage=usage,
            stop_reason=resp.stop_reason,
            served_model=served,
            request_id=getattr(resp, "_request_id", None),
            latency_s=latency,
            attempts=attempts,
        )


class OpenAIModel(APIModel):
    """Responses API with strict JSON-schema output. `store=False`: nothing is kept server-side."""

    def __init__(self, name: str, cfg: dict, client=None):
        super().__init__(name, cfg)
        import openai

        self._sdk = openai
        self.client = client or openai.OpenAI(max_retries=0, timeout=self.request_timeout_s)

    def request_params(self, system: str, user: str) -> dict:
        params = {
            "model": self.model,
            "instructions": system,
            "input": user,
            "max_output_tokens": self.max_tokens,
            "text": {"format": {"type": "json_schema", "name": "sql_answer", "schema": OUTPUT_SCHEMA, "strict": True}},
            "store": False,
        }
        if self.config.get("reasoning_effort"):
            params["reasoning"] = {"effort": self.config["reasoning_effort"]}
        return params

    def _call(self, system, user):
        return self.client.responses.create(**self.request_params(system, user))

    def _classify(self, e):
        o = self._sdk
        if isinstance(e, o.APIStatusError):
            return self._retry_for_status(e.status_code)
        if isinstance(e, o.APIConnectionError):  # includes APITimeoutError
            return RETRY
        return None

    def _is_timeout(self, e):
        return isinstance(e, self._sdk.APITimeoutError)

    def _to_completion(self, resp, latency, attempts):
        served = self._check_served(resp.model, attempts)
        reason = resp.incomplete_details.reason if resp.incomplete_details else None
        if resp.status == "failed" or resp.error:
            raise InfraError("api_error", f"response failed: {resp.error}", attempts)
        if resp.status == "incomplete" and reason not in ("max_output_tokens", "content_filter"):
            raise InfraError("api_error", f"response incomplete: {reason}", attempts)
        refused = any(
            getattr(part, "type", None) == "refusal"
            for item in resp.output or []
            if getattr(item, "type", None) == "message"
            for part in item.content or []
        )
        u = resp.usage
        cached = (u.input_tokens_details.cached_tokens or 0) if u and u.input_tokens_details else 0
        usage = {
            "input_tokens": (u.input_tokens or 0) - cached if u else 0,
            "output_tokens": (u.output_tokens or 0) if u else 0,  # includes reasoning tokens
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0,
            "reasoning_tokens": (u.output_tokens_details.reasoning_tokens or 0) if u and u.output_tokens_details else 0,
        }
        return self._finish(
            text=resp.output_text or "",
            refusal=refused or reason == "content_filter",
            truncated=reason == "max_output_tokens",
            usage=usage,
            stop_reason=reason or resp.status,
            served_model=served,
            request_id=getattr(resp, "_request_id", None),
            latency_s=latency,
            attempts=attempts,
        )


class GeminiModel(APIModel):
    """Gemini API (google-genai `models.generate_content`) with a JSON response schema."""

    # Finish reasons meaning the model or its safety layer declined to answer.
    BLOCKED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "LANGUAGE"}

    def __init__(self, name: str, cfg: dict, client=None):
        super().__init__(name, cfg)
        import httpx
        from google import genai
        from google.genai import errors, types

        self._httpx, self._errors, self._types = httpx, errors, types
        # google-genai does not retry unless retry_options is set, which is what we want.
        self.client = client or genai.Client(http_options=types.HttpOptions(timeout=int(self.request_timeout_s * 1000)))

    def request_config(self, system: str):
        t = self._types
        kwargs = dict(
            system_instruction=system,
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json",
            response_json_schema=OUTPUT_SCHEMA,
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),  # no tools
        )
        if self.config.get("thinking_level"):
            kwargs["thinking_config"] = t.ThinkingConfig(thinking_level=self.config["thinking_level"].upper())
        return t.GenerateContentConfig(**kwargs)

    def _call(self, system, user):
        return self.client.models.generate_content(model=self.model, contents=user, config=self.request_config(system))

    def _classify(self, e):
        if isinstance(e, self._errors.APIError):
            return self._retry_for_status(e.code or 500)
        if isinstance(e, self._httpx.TransportError):  # connection errors and timeouts
            return RETRY
        return None

    def _is_timeout(self, e):
        return isinstance(e, self._httpx.TimeoutException)

    def _to_completion(self, resp, latency, attempts):
        served = self._check_served(resp.model_version, attempts)
        candidate = resp.candidates[0] if resp.candidates else None
        finish = getattr(candidate.finish_reason, "value", candidate.finish_reason) if candidate else None
        blocked_prompt = bool(resp.prompt_feedback and resp.prompt_feedback.block_reason)
        parts = (candidate.content.parts or []) if candidate and candidate.content else []
        text = "".join(p.text for p in parts if p.text and not p.thought)
        if not text and not blocked_prompt and finish not in self.BLOCKED | {"MAX_TOKENS"}:
            # Empty response with no refusal or truncation behind it (e.g. finish OTHER):
            # nothing the model said can be scored, so retry it rather than grade it.
            raise InfraError("api_error", f"empty response (finish_reason={finish})", attempts)
        u = resp.usage_metadata
        cached = (u.cached_content_token_count or 0) if u else 0
        thoughts = (u.thoughts_token_count or 0) if u else 0
        usage = {
            "input_tokens": (u.prompt_token_count or 0) - cached if u else 0,
            "output_tokens": ((u.candidates_token_count or 0) + thoughts) if u else 0,
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0,
            "reasoning_tokens": thoughts,
        }
        return self._finish(
            text=text,
            refusal=blocked_prompt or finish in self.BLOCKED,
            truncated=finish == "MAX_TOKENS",
            usage=usage,
            stop_reason=finish or ("PROMPT_BLOCKED" if blocked_prompt else None),
            served_model=served,
            request_id=resp.response_id,
            latency_s=latency,
            attempts=attempts,
        )


class LocalModel(APIModel):
    """Open-weight model behind an OpenAI-compatible Chat Completions server (Ollama, vLLM,
    llama.cpp server, LM Studio). Same system prompt, user message, JSON schema and output
    budget as the hosted vendors; only the transport differs.

    Pinning: a local tag such as `qwen3:8b` can be re-pulled to different weights, so for
    Ollama the roster records the weights' `digest` and every run checks it before the first
    request. That digest is part of the model config, so it lands in the run manifest.
    """

    DEFAULT_BASE_URL = "http://localhost:11434/v1"  # Ollama

    def __init__(self, name: str, cfg: dict, client=None, http=None):
        super().__init__(name, cfg)
        import httpx
        import openai

        self._sdk = openai
        self.base_url = cfg.get("base_url", self.DEFAULT_BASE_URL).rstrip("/")
        self.client = client or openai.OpenAI(
            base_url=self.base_url, api_key="local", max_retries=0, timeout=self.request_timeout_s,
        )
        if cfg.get("digest"):
            self._check_digest(http or httpx.Client(timeout=30))

    def _check_digest(self, http) -> None:
        """Ollama-only: confirm the tag still points at the pinned weights."""
        native = self.base_url.removesuffix("/v1")
        try:
            tags = http.get(f"{native}/api/tags").raise_for_status().json()
        except Exception as e:
            raise FatalError(f"cannot reach the model server at {native} ({type(e).__name__}: {e}); "
                             f"is it running? (`ollama serve`)") from e
        local = {m.get("model") or m.get("name"): m.get("digest", "") for m in tags.get("models", [])}
        if self.model not in local:
            raise FatalError(f"{self.model} is not pulled on {native} (`ollama pull {self.model}`)")
        if not local[self.model].startswith(self.config["digest"]):
            raise FatalError(f"{self.model} has digest {local[self.model][:12]}, roster pins "
                             f"{self.config['digest'][:12]}: the tag now points at different weights. "
                             f"Add a new roster name instead of editing this one.")

    def request_params(self, system: str, user: str) -> dict:
        params = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "sql_answer", "schema": OUTPUT_SCHEMA, "strict": True}},
        }
        if self.config.get("reasoning_effort"):
            params["reasoning_effort"] = self.config["reasoning_effort"]
        if "temperature" in self.config:
            params["temperature"] = self.config["temperature"]
        return params

    def _call(self, system, user):
        return self.client.chat.completions.create(**self.request_params(system, user))

    def _classify(self, e):
        o = self._sdk
        if isinstance(e, o.APIStatusError):
            return self._retry_for_status(e.status_code)
        if isinstance(e, o.APIConnectionError):  # includes APITimeoutError; also "server not running"
            return RETRY
        return None

    def _is_timeout(self, e):
        return isinstance(e, self._sdk.APITimeoutError)

    def _to_completion(self, resp, latency, attempts):
        served = self._check_served(resp.model, attempts)
        choice = resp.choices[0]
        msg = choice.message
        u = resp.usage
        # Thinking models put their reasoning outside `content` (Ollama `reasoning`, vLLM
        # `reasoning_content`), so `content` is the answer; completion_tokens counts both.
        usage = {
            "input_tokens": (u.prompt_tokens or 0) if u else 0,
            "output_tokens": (u.completion_tokens or 0) if u else 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        completion = self._finish(
            text=msg.content or "",
            refusal=bool(getattr(msg, "refusal", None)) or choice.finish_reason == "content_filter",
            truncated=choice.finish_reason == "length",
            usage=usage,
            stop_reason=choice.finish_reason,
            served_model=served,
            request_id=resp.id,
            latency_s=latency,
            attempts=attempts,
        )
        # No per-token charge. Hardware and electricity aren't counted (see NOTES.md).
        completion.cost_usd = 0.0
        return completion
