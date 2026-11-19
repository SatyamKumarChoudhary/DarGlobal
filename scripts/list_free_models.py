"""Print the currently live free models on OpenRouter.

Usage: ``uv run python scripts/list_free_models.py`` (or ``make free-models``).
No API key required; the models endpoint is public.
"""

import json
import urllib.request

MODELS_URL = "https://openrouter.ai/api/v1/models"


def main() -> None:
    """Fetch the model list and print free instruct models, largest first."""
    with urllib.request.urlopen(MODELS_URL, timeout=30) as response:  # noqa: S310
        payload = json.load(response)
    free_models = []
    for model in payload.get("data", []):
        model_id = model.get("id", "")
        pricing = model.get("pricing", {})
        if not model_id.endswith(":free"):
            continue
        if float(pricing.get("prompt", "1") or "1") != 0.0:
            continue
        free_models.append(
            (
                model_id,
                int(model.get("context_length") or 0),
                (model.get("name") or "")[:60],
            )
        )
    free_models.sort(key=lambda entry: -entry[1])
    print(f"{'MODEL ID':<60} {'CTX':>8}  NAME")
    for model_id, context_length, name in free_models:
        print(f"{model_id:<60} {context_length:>8}  {name}")
    print(f"\n{len(free_models)} free models live right now.")
    print("Pick one for LLM_PRIMARY_MODEL and 2-3 for LLM_FALLBACK_MODELS in .env.")


if __name__ == "__main__":
    main()
