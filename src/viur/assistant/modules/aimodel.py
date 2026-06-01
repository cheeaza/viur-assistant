import logging
import typing as t

from viur.core import access, errors, exposed
from viur.toolkit import as_json_response

from viur.assistant.config import CONFIG
from viur.assistant.providers import ModelInfo, get_provider
from viur.core.prototypes import List


_PROVIDERS: list[str] = ["gemini", "openai", "anthropic"]

_API_KEYS: dict[str, t.Callable[[], str | None]] = {
    "gemini": lambda: CONFIG.api_gemini_key,
    "openai": lambda: CONFIG.api_openai_key,
    "anthropic": lambda: CONFIG.api_anthropic_key,
}


class Aimodel(List):
    kindName = "aimodel"

    adminInfo = {
        "name": "KI-Modelle",
        "icon": "cpu",
        "moduleGroup": "configuration",
        "sortIndex": 411,
        "columns": [
            "model_id",
            "name",
            "provider",
            "deprecated",
        ],
        "actions": ["load_from_api", "load_from_api_gemini", "load_from_api_openai", "load_from_api_anthropic"],
        "customActions": {
            "load_from_api": {
                "name": "Alle Modelle synchronisieren",
                "icon": "cloud-arrow-down",
                "access": ["root"],
                "action": "fetch",
                "url": "/vi/{{module}}/loadFromApi",
                "confirm": "Alle Provider synchronisieren?",
                "success": "Modelle synchronisiert.",
                "enabled": "True",
            },
            "load_from_api_gemini": {
                "name": "Gemini synchronisieren",
                "icon": "cloud-arrow-down",
                "access": ["root"],
                "action": "fetch",
                "url": "/vi/{{module}}/loadFromApi?provider=gemini",
                "confirm": "Gemini-Modelle synchronisieren?",
                "success": "Gemini-Modelle synchronisiert.",
                "enabled": "True",
            },
            "load_from_api_openai": {
                "name": "OpenAI synchronisieren",
                "icon": "cloud-arrow-down",
                "access": ["root"],
                "action": "fetch",
                "url": "/vi/{{module}}/loadFromApi?provider=openai",
                "confirm": "OpenAI-Modelle synchronisieren?",
                "success": "OpenAI-Modelle synchronisiert.",
                "enabled": "True",
            },
            "load_from_api_anthropic": {
                "name": "Anthropic synchronisieren",
                "icon": "cloud-arrow-down",
                "access": ["root"],
                "action": "fetch",
                "url": "/vi/{{module}}/loadFromApi?provider=anthropic",
                "confirm": "Anthropic-Modelle synchronisieren?",
                "success": "Anthropic-Modelle synchronisiert.",
                "enabled": "True",
            },
        },
    }

    roles = {
        "admin": "*",
    }

    def listFilter(self, query):
        if self.render.kind == "json.vi" and (superquery := super().listFilter(query)):
            return superquery
        return None  # not public

    def _sync_provider(self, provider: str, models: list[ModelInfo]) -> dict:
        """Upserts models for one provider, marks removed ones as deprecated."""
        seen_ids: set[str] = set()
        created = updated = 0

        for m in models:
            seen_ids.add(m["model_id"])
            existing = self.editSkel().all().filter("model_id =", m["model_id"]).getSkel()

            if existing:
                changed = False
                for field in ("name", "description", "input_token_limit", "output_token_limit"):
                    if existing[field] != m[field]:
                        existing[field] = m[field]
                        changed = True
                if existing["deprecated"]:
                    existing["deprecated"] = False
                    changed = True
                if changed:
                    existing.write()
                    updated += 1
            else:
                skel = self.addSkel()
                skel["model_id"] = m["model_id"]
                skel["name"] = m["name"]
                skel["provider"] = provider
                skel["deprecated"] = False
                skel["description"] = m["description"]
                skel["input_token_limit"] = m["input_token_limit"]
                skel["output_token_limit"] = m["output_token_limit"]
                skel.write()
                created += 1

        deprecated_now = 0
        for skel in self.editSkel().all().filter("provider =", provider).fetch(1000):
            if skel["model_id"] not in seen_ids and not skel["deprecated"]:
                skel["deprecated"] = True
                skel.write()
                deprecated_now += 1

        return {
            "created": created,
            "updated": updated,
            "deprecated": deprecated_now,
            "total_from_api": len(seen_ids),
        }

    @exposed
    @access("root")
    @as_json_response
    def loadFromApi(self, provider: str = "") -> dict:
        """Sync available models from all configured providers into the aimodel module.

        :param provider: If given, only this provider is synced (e.g. ``"gemini"`` or ``"openai"``).
                         Omit or pass an empty string to sync all providers.
        """
        if provider and provider not in _PROVIDERS:
            raise errors.BadRequest(f"Unknown provider {provider!r}. Valid: {sorted(_PROVIDERS)}")

        results = {}
        for p in _PROVIDERS:
            if provider and p != provider:
                continue
            api_key = _API_KEYS[p]()
            if not api_key:
                logging.info(f"aimodel sync: skipping {p!r} (no API key configured)")
                continue
            try:
                models = get_provider(p).list_models()
            except Exception as exc:
                logging.exception(f"Failed to list {p!r} models")
                results[p] = {"error": str(exc)}
                continue

            stats = self._sync_provider(p, models)
            logging.info(f"aimodel sync [{p}]: {stats}")
            results[p] = stats

        return results


Aimodel.json = True
