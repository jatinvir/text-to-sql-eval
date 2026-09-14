"""Per-token prices used to cost each response from the API's own usage counts.

USD per million tokens, standard (non-batch) tier. Sources and dates:
  Anthropic  anthropic.com/pricing                     as of 2026-06-24
  OpenAI     developers.openai.com/api/docs/pricing    as of 2026-09-14
  Google     ai.google.dev/gemini-api/docs/pricing     as of 2026-09-14 (page dated 2026-09-11)
Check the pages before quoting numbers externally. Each row's cost is computed and stored
at run time, so a later price change doesn't silently rewrite old results.

`output` applies to all output tokens, including reasoning/thinking tokens - every vendor
bills those as output. `cache_read` is the discounted rate for cached input tokens;
`cache_write` is what writing to the cache costs (Anthropic charges a premium; OpenAI and
Gemini implicit caching charge the normal input rate). The harness doesn't enable caching,
but vendors may cache automatically, so the rates are kept to keep cost correct.
"""

from __future__ import annotations

PRICES_PER_MTOK: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-fable-5-1": {"input": 10.00, "output": 50.00, "cache_read": 0.25, "cache_write": 12.50},
    "claude-opus-5": {"input": 5.00, "output": 25.00, "cache_read": 0.50, "cache_write": 6.25},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00, "cache_read": 0.50, "cache_write": 6.25},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write": 2.50},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_read": 0.10, "cache_write": 1.25},
    # OpenAI
    "gpt-6-astra": {"input": 10.00, "output": 50.00, "cache_read": 1.00, "cache_write": 10.00},
    "gpt-5.6-sol": {"input": 4.00, "output": 20.00, "cache_read": 0.40, "cache_write": 4.00},  # promotional through 2026-11-21
    "gpt-5.6-terra": {"input": 2.00, "output": 12.00, "cache_read": 0.20, "cache_write": 2.00},
    "gpt-5.6-luna": {"input": 0.20, "output": 1.20, "cache_read": 0.02, "cache_write": 0.20},
    # Google (prompts <= 200k tokens, which every prompt here is)
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00, "cache_read": 0.20, "cache_write": 2.00},
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75, "cache_read": 0.075, "cache_write": 0.75},  # doubles 2027-01-01
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50, "cache_read": 0.30, "cache_write": 0.30},  # no caching
}


def cost_usd(model: str, usage: dict) -> float | None:
    """Cost of one response. Returns None for models without a price entry rather than
    guessing - an unknown cost must not look like a cheap one."""
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    total = (
        usage.get("input_tokens", 0) * prices["input"]
        + usage.get("output_tokens", 0) * prices["output"]
        + usage.get("cache_read_input_tokens", 0) * prices["cache_read"]
        + usage.get("cache_creation_input_tokens", 0) * prices["cache_write"]
    )
    return total / 1_000_000
