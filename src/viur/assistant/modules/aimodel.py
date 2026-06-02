"""
Aimodel module – admin management of AI model entries.

Provides the :class:`Aimodel` List module which stores one entry per AI model
(e.g. ``gemini-2.5-flash``, ``gpt-4o``, ``claude-3-7-sonnet-20250219``).  Entries
are either created manually in the admin or synced from the provider APIs via the
:meth:`~Aimodel.loadFromApi` endpoint.

**Provider registration**

:data:`_PROVIDERS` lists all supported provider identifiers.  To add a new provider:

1. Implement :meth:`~viur.assistant.providers.base.BaseProvider.list_models` in the
   corresponding provider class.
2. Add the provider name to :data:`_PROVIDERS`.
3. Add the API-key accessor to :data:`_API_KEYS`.
4. Add a ``customAction`` entry to :attr:`Aimodel.adminInfo`.
5. Add the provider value to :class:`~viur.assistant.skeletons.aimodel.AimodelSkel`'s
   ``provider`` SelectBone.
"""

import logging
import typing as t

from viur.core import access, errors, exposed
from viur.toolkit import as_json_response

from viur.assistant.config import CONFIG
from viur.assistant.providers import ModelInfo, get_provider
from viur.core.prototypes import List


_PROVIDERS: list[str] = ["gemini", "openai", "anthropic"]
"""Ordered list of all supported provider identifiers used during full sync."""

_API_KEYS: dict[str, t.Callable[[], str | None]] = {
    "gemini": lambda: CONFIG.api_gemini_key,
    "openai": lambda: CONFIG.api_openai_key,
    "anthropic": lambda: CONFIG.api_anthropic_key,
}
"""Lazy accessors for each provider's API key.  Evaluated at sync time so that
keys configured after import are picked up correctly."""


class Aimodel(List):
    """Admin module for managing AI model entries.

    Each entry represents one model available from a specific provider.  Entries
    can be created manually or populated automatically via :meth:`loadFromApi`,
    which calls :meth:`~viur.assistant.providers.base.BaseProvider.list_models`
    for each configured provider and upserts the results into the Datastore.

    Models that were previously synced but are no longer returned by the provider
    API are automatically marked as ``deprecated=True``.  Deprecated models remain
    selectable in the admin but are hidden from the availability check in
    :class:`~viur.assistant.modules.assistant.Assistant`.
    """

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
        """Restrict list access to the vi-admin JSON renderer only.

        The ``aimodel`` kind is internal configuration data and must not be
        exposed via public HTML or JSON renderers.  Only authenticated vi-admin
        requests (``json.vi``) pass through.
        """
        if self.render.kind == "json.vi" and (superquery := super().listFilter(query)):
            return superquery
        return None  # not public

    def _sync_provider(self, provider: str, models: list[ModelInfo]) -> dict:
        """Upsert model entries for one provider and deprecate removed ones.

        Iterates over *models* (fetched from the provider API) and either updates
        an existing Datastore entry or creates a new one.  After processing all
        returned models, any entry for *provider* whose ``model_id`` was not in the
        API response is marked ``deprecated=True``.

        :param provider: Provider identifier (e.g. ``"gemini"``).
        :param models: List of :class:`~viur.assistant.providers.base.ModelInfo`
            dicts returned by
            :meth:`~viur.assistant.providers.base.BaseProvider.list_models`.
        :returns: Stats dict with keys ``created``, ``updated``, ``deprecated``,
            and ``total_from_api``.
        """
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
        """Sync available models from provider APIs into the aimodel kind.

        Iterates over all entries in :data:`_PROVIDERS` (or only *provider* if
        given), skips any provider whose API key is not configured, and calls
        :meth:`~viur.assistant.providers.base.BaseProvider.list_models` followed
        by :meth:`_sync_provider`.

        :param provider: If non-empty, only this provider is synced
            (e.g. ``"gemini"``, ``"openai"``, or ``"anthropic"``).
            Omit or pass an empty string to sync all providers.
        :returns: Dict mapping provider name to a stats dict
            (``created``, ``updated``, ``deprecated``, ``total_from_api``)
            or ``{"error": "<message>"}`` on failure.
        :raises BadRequest: If *provider* is not a known provider identifier.
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
