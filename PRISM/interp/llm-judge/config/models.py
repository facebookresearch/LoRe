"""Judge model registry: short alias -> provider + concrete model id + temperature.

`--models <alias> ...` on the CLI selects from these. The default run uses all
three families (one round each). Each model_id must be a concrete string; any alias
left at UNSET_MODEL_ID is rejected before the run makes a single API call, so an
unfinished config fails fast instead of burning quota on a nonexistent model.

`temperature` is the per-family sampling temperature. The only band valid across
all three families is [0.0, 1.0] (Anthropic caps at 1.0), so 0.0 is the shared
low-variance setting for the families that accept it. A value of None means the
parameter is omitted from the request entirely -- required for reasoning models
that reject an explicit temperature. A global `--temperature` on the CLI overrides
every family's value here.
"""

# Sentinel meaning "no model id has been chosen for this alias yet".
UNSET_MODEL_ID = "<unset>"

JUDGE_MODELS: dict[str, dict] = {
    # None => omit temperature: this model has the parameter deprecated and 400s if sent.
    "claude": {"provider": "anthropic", "model_id": "claude-opus-4-8", "temperature": None},
    "gemini": {"provider": "google", "model_id": "gemini-3.5-flash", "temperature": 0.0},
    # None => omit temperature: this reasoning model rejects the parameter.
    "chatgpt": {"provider": "openai", "model_id": "gpt-5.6-sol", "temperature": None},
}

# Default set of aliases when --models is not passed: one round per LLM family.
DEFAULT_MODELS: list[str] = ["claude", "gemini", "chatgpt"]
