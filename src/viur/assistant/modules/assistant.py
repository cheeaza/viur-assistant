"""
Assistant module for viur-assistant.

Provides the :class:`Assistant` singleton, which exposes LLM-powered endpoints
for script generation, image description, and text translation. All AI calls are
routed through the provider abstraction in :mod:`viur.assistant.providers`, so the
underlying model (Anthropic, Gemini, OpenAI, …) is determined entirely by the
``ai_model`` relation configured in the singleton skeleton – no code changes needed
when switching providers.

Public endpoints (all require ``admin`` access unless noted):

* ``POST /assistant/generate_script`` – Generate Python code for a ViUR backend.
* ``POST /assistant/translate``       – Translate text into a target language.
* ``POST /assistant/describe_image``  – Generate an HTML ``alt`` attribute for an image.
"""

import base64
import io
import json
import os
import typing as t

import PIL
from viur.core import conf, current, db, errors, exposed, utils
from viur.core.decorators import access, force_post
from viur.core.prototypes import List, Singleton, Tree

from viur.assistant.config import ASSISTANT_LOGGER, CONFIG
from viur.assistant.providers import BaseProvider, get_provider

logger = ASSISTANT_LOGGER.getChild(__name__)


class Assistant(Singleton):
    """LLM-powered utility singleton for ViUR admin tooling.

    This singleton acts as the backend for AI-assisted features in the ViUR admin
    interface. Every endpoint delegates the actual model call to a
    :class:`~viur.assistant.providers.base.BaseProvider` instance that is resolved
    at request time from the ``ai_model`` relation stored in the singleton skeleton.

    **Provider selection**

    The provider (Anthropic / Gemini / OpenAI / …) and the concrete model are
    determined by the ``ai_model`` RelationalBone in the skeleton, which points to
    an entry in the ``aimodel`` kind.  The resolved ``provider`` and ``model_id``
    fields from that entry are used to instantiate the correct
    :class:`~viur.assistant.providers.base.BaseProvider` via
    :func:`~viur.assistant.providers.get_provider`.

    **Configuration**

    Runtime settings (temperature, max tokens, system prompt, …) come from the
    singleton skeleton itself.  API keys are read from
    :data:`viur.assistant.config.CONFIG` which is populated in the project's
    ``main.py``.

    .. note::
       Configure only what you actually use – every endpoint checks that a model
       is configured and raises :class:`~viur.core.errors.InternalServerError`
       with a descriptive message otherwise.
    """

    kindName: t.Final[str] = "viur-assistant"

    # ── Internal helpers ─────────────────────────────────────────────

    def _get_provider_and_model(self, skel) -> tuple[BaseProvider, str]:
        """Resolve the AI provider and model ID from the skeleton's ``ai_model`` relation.

        Reads ``ai_model.dest.provider`` and ``ai_model.dest.model_id`` from the
        singleton skeleton.  Both fields must be present in the ``refKeys`` of the
        ``ai_model`` :class:`~viur.core.bones.RelationalBone`.

        :param skel: The populated singleton skeleton returned by :meth:`getContents`.
        :returns: A ``(provider, model_id)`` tuple ready for use in a
            :meth:`~viur.assistant.providers.base.BaseProvider.complete` call.
        :raises InternalServerError: If no ``ai_model`` is configured or
            ``model_id`` is empty.
        """
        dest = ((skel.get("ai_model") or {}).get("dest") or {})
        model_id = dest.get("model_id") or ""
        if not model_id:
            raise errors.InternalServerError(descr="No AI model configured in assistant settings.")
        return get_provider(dest.get("provider", "openai")), model_id

    # ── Public endpoints ─────────────────────────────────────────────

    @exposed
    @access("admin")
    @force_post
    def generate_script(
        self,
        *,
        prompt: str,
        modules_to_include: list[str] = None,
        enable_caching: bool = False,
        max_thinking_tokens: int = 0,
    ):
        """Generate a ViUR backend script from a natural-language prompt.

        Builds a structured message for the configured LLM, optionally enriching
        the context with serialised ViUR module structures, and returns the model
        response as a JSON object ``{"code": "<generated code>"}``.

        :param prompt: Natural-language instruction for the code to generate.
        :param modules_to_include: Optional list of ViUR module names whose skeleton
            structures are injected into the prompt as JSON, giving the model
            knowledge of available bones and their types.
        :param enable_caching: When ``True``, marks the documentation section of
            the system prompt with an ephemeral cache-control header (Anthropic only).
        :param max_thinking_tokens: Budget for extended reasoning steps.  If > 0
            and the provider supports thinking
            (:meth:`~viur.assistant.providers.base.BaseProvider.supports_thinking`),
            this many tokens are reserved for internal chain-of-thought before the
            visible response.  Capped by ``skel["max_thinking_tokens"]``.
        :returns: JSON response ``{"code": "<result>"}`` with
            ``Content-Type: application/json``.
        :raises InternalServerError: If the skeleton is missing, no model is
            configured, or the LLM call fails.
        """
        if not (skel := self.getContents()):
            raise errors.InternalServerError(descr="Configuration missing")

        messages = []

        if modules_to_include is not None and (structures := self.get_viur_structures(modules_to_include)):
            messages.append({
                "role": "user",
                "content": json.dumps({"module_structures": structures}, indent=2),
            })

        messages.append({"role": "user", "content": prompt})

        effective_thinking_tokens = 0
        if max_thinking_tokens > 0:
            effective_thinking_tokens = (
                min(max_thinking_tokens, skel["max_thinking_tokens"])
                if skel["max_thinking_tokens"]
                else max_thinking_tokens
            )

        provider, model_id = self._get_provider_and_model(skel)
        logger.debug(f"generate_script: {provider.__class__.__name__}, model={model_id}")
        try:
            result = provider.complete(
                model=model_id,
                messages=messages,
                temperature=skel["temperature"],
                max_tokens=skel["max_tokens"],
                system_prompt=skel["system_prompt"],
                max_thinking_tokens=effective_thinking_tokens,
                enable_caching=enable_caching,
            )
        except Exception as e:
            logger.exception(e)
            raise errors.InternalServerError(descr=str(e))

        current.request.get().response.headers["Content-Type"] = "application/json"
        return json.dumps({"code": result})  # TODO: parse real "code" value

    def get_viur_structures(self, modules_to_include: t.Iterable[str]) -> dict[str, dict]:
        """Collect ViUR module structures for use as LLM context.

        For each named module, the skeleton structure is extracted via
        :meth:`~viur.core.prototypes.List.structure`.  :class:`~viur.core.prototypes.Tree`
        modules return a nested dict with ``"node"`` and ``"leaf"`` keys.
        Modules that are not found in ``conf.main_app.vi`` are silently skipped.

        :param modules_to_include: Iterable of ViUR module names to look up.
        :returns: Dict mapping module names to their structure dicts.
        :raises ValueError: If a module is found but is neither a
            :class:`~viur.core.prototypes.List` nor a
            :class:`~viur.core.prototypes.Tree`.
        """
        structures_from_viur = {}
        for module_name in modules_to_include:
            module = getattr(conf.main_app.vi, module_name, None)
            if not module:
                continue
            if isinstance(module, List):
                if module_name not in structures_from_viur:
                    structures_from_viur[module_name] = module.structure()
            elif isinstance(module, Tree):
                if module_name not in structures_from_viur:
                    structures_from_viur[module_name] = {
                        "node": module.structure(skelType="node"),
                        "leaf": module.structure(skelType="leaf"),
                    }
            else:
                raise ValueError(
                    f"The ViUR-module must be of type 'Tree' or 'List'. "
                    f"{module!r} is (currently) unsupported."
                )
        return structures_from_viur

    @exposed
    @access("admin")
    @force_post
    def translate(
        self,
        *,
        text: str,
        language: str,
        characteristic: t.Optional[str] = None,
    ):
        """Translate text into a target language using the configured LLM.

        Builds a translation prompt incorporating optional style characteristics
        from :attr:`~viur.assistant.config.AssistantConfig.translate_language_characteristics`
        and returns the translated text as-is (HTML tags are preserved).

        :param text: The source text to translate.
        :param language: BCP-47 language code for the target language, e.g.
            ``"de"``, ``"en"``, ``"de-DE-x-simple-language"``.  Resolved to a
            human-readable name via
            :attr:`~viur.assistant.config.AssistantConfig.language_map`; falls
            back to the raw code if not mapped.
        :param characteristic: Optional style key from
            :attr:`~viur.assistant.config.AssistantConfig.translate_language_characteristics`,
            e.g. ``"simple"``.  Its rules are merged with the always-active ``"*"``
            base rules.
        :returns: Translated text rendered for the current renderer
            (plain text for HTML, JSON-encoded string for JSON renderer).
        :raises InternalServerError: If the skeleton is missing, no model is
            configured, or the LLM call fails.
        """
        if not (skel := self.getContents()):
            raise errors.InternalServerError(descr="Configuration missing")

        characteristics = [
            *CONFIG.translate_language_characteristics.get("*", []),
            *CONFIG.translate_language_characteristics.get(characteristic, []),
        ]

        provider, model_id = self._get_provider_and_model(skel)
        try:
            result = provider.complete(
                model=model_id,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Translate the following text into {CONFIG.language_map.get(language, language)}"
                        f" ({'. '.join(characteristics)})"
                        f" and only return the translation, keep HTML-tags (if there are any):\n\n{text}\n"
                    ),
                }],
            )
        except Exception as e:
            logger.exception(e)
            raise errors.InternalServerError(descr=str(e))

        return self.render_text(result)

    @exposed
    @access("admin", "file-view")
    @force_post
    def describe_image(
        self,
        filekey: db.Key | str,
        prompt: str = "",
        context: str = "",
        language: str | None = None,
    ):
        """Generate an HTML ``alt`` attribute description for an image file.

        Reads the image from the ViUR file module, resizes it to the configured
        target pixel count (see
        :attr:`~viur.assistant.config.AssistantConfig.describe_image_pixel_default`)
        to reduce token cost, encodes it as base64 JPEG, and sends it to the
        configured vision-capable LLM.

        The provider must support vision
        (:meth:`~viur.assistant.providers.base.BaseProvider.supports_vision`);
        an :class:`~viur.core.errors.InternalServerError` is raised otherwise.

        :param filekey: Key of the
            :class:`~viur.core.modules.file.FileLeafSkel` entry to describe.
        :param prompt: Optional hint for the model, e.g. ``"Focus on the product,
            ignore the background."``.
        :param context: Optional additional context injected alongside the prompt,
            e.g. a product name or category.
        :param language: BCP-47 language code for the generated description.
            Falls back to :func:`~viur.core.current.language.get` if not specified.
        :returns: Plain-text ``alt`` attribute value rendered for the current
            renderer.
        :raises InternalServerError: If the skeleton is missing, no model is
            configured, the provider does not support vision, or the LLM call fails.
        :raises NotFound: If the file referenced by ``filekey`` does not exist.
        """
        if not (skel := self.getContents()):
            raise errors.InternalServerError(descr="Configuration missing")

        provider, model_id = self._get_provider_and_model(skel)
        if not provider.supports_vision():
            raise errors.InternalServerError(
                descr=f"{provider.__class__.__name__} does not support vision."
            )

        if language is None:
            language = current.language.get()

        blob, mime = conf.main_app.file.read(key=filekey)
        if not blob:
            raise errors.NotFound(f"File not found with {filekey=!r}")

        resized_image_bytes = self._get_resized_image_bytes(
            image=blob,
            target_pixel_count=CONFIG.describe_image_pixel_default,
            jpeg_quality=CONFIG.describe_image_jpeg_quality_default,
        )
        base64_image = base64.b64encode(resized_image_bytes).decode("utf-8")

        context_prompt = ""
        if context or prompt:
            context_prompt = (
                f"Use the following data as additional information to describe the image:\n"
                f" {prompt}\n\n{context}"
            )

        try:
            result = provider.complete(
                model=model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Analyze the image and generate an appropriate HTML alt attribute"
                                f" in language: {CONFIG.language_map.get(language, language)}."
                                f" Provide only the plain text for the alt attributes without quotes and label.\n\n"
                                f"{context_prompt}\n"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                # "low" = 85 tokens, resize to < 512×512 px on the provider side.
                                # See: https://platform.openai.com/docs/guides/images?api-mode=chat#calculating-costs
                                "detail": "low",
                            },
                        },
                    ],
                }],
            )
        except Exception as e:
            logger.exception(e)
            raise errors.InternalServerError(descr=str(e))

        return self.render_text(result)

    # ── Private utilities ────────────────────────────────────────────

    def _get_resized_image_bytes(
        self,
        image: t.IO[bytes] | str | bytes | os.PathLike[str] | os.PathLike[bytes],
        target_pixel_count: int,
        jpeg_quality: int = 50,
    ) -> bytes:
        """Resize an image proportionally and return it as a JPEG byte string.

        The image is scaled so that ``width × height ≈ target_pixel_count`` while
        preserving the original aspect ratio.  If the computed dimensions would
        exceed the original size, the image is returned unscaled (no upscaling).

        PNG, SVG, and WEBP images are converted to RGB JPEG before resizing.

        :param image: Source image as a file-like object or raw bytes.
        :param target_pixel_count: Desired total pixel count (``width × height``).
        :param jpeg_quality: JPEG compression quality from 0 (lowest) to 100
            (highest). Defaults to 50 as a balance between file size and detail
            sufficient for AI-based image analysis.
        :returns: JPEG-encoded image bytes.
        :raises ValueError: If ``jpeg_quality`` is outside the 0–100 range or
            ``image`` is neither file-like nor bytes.
        """
        if not (0 <= jpeg_quality <= 100):
            raise ValueError("jpeg_quality must be between 0 and 100")

        if isinstance(image, bytes):
            image = io.BytesIO(image)
        if not isinstance(image, (io.TextIOBase, io.BufferedIOBase, io.RawIOBase, io.IOBase)):
            raise ValueError("image must be file-like or bytes")

        pillow_image = PIL.Image.open(image)
        if pillow_image.format in ["PNG", "SVG", "WEBP"]:  # TODO: ???
            jpeg_image = io.BytesIO()
            pillow_image.convert("RGB").save(jpeg_image, "JPEG")
            jpeg_image.seek(0)
            pillow_image = PIL.Image.open(jpeg_image)

        original_img_total_pixels = pillow_image.width * pillow_image.height
        side_ratio_to_n_pixels = (target_pixel_count / original_img_total_pixels) ** 0.5
        new_width = round(pillow_image.width * side_ratio_to_n_pixels)
        new_height = round(pillow_image.height * side_ratio_to_n_pixels)

        if new_height > pillow_image.height or new_width > pillow_image.width:
            resized_img = pillow_image
        else:
            resized_img = pillow_image.resize(
                (new_width, new_height),
                PIL.Image.Resampling.LANCZOS,
            )

        result_bio = io.BytesIO()
        resized_img.save(result_bio, "jpeg", quality=jpeg_quality)
        result_bio.seek(0)
        return result_bio.read()

    def render_text(self, text: str) -> t.Any:
        """Render a plain-text string for the current ViUR renderer.

        Sets the appropriate ``Content-Type`` header and returns the text in the
        format expected by the active renderer:

        * **HTML renderer** – returns the text as-is with ``text/html`` content type.
        * **JSON renderer** – returns the text JSON-encoded with
          ``application/json`` content type.

        :param text: The text to render.
        :returns: The text, optionally JSON-encoded.
        :raises NotImplemented: If the current renderer is neither HTML nor JSON.
        """
        if utils.string.is_prefix(self.render.kind, "html"):
            current.request.get().response.headers["Content-Type"] = "text/html; charset=utf-8"
            return text
        elif utils.string.is_prefix(self.render.kind, "json"):
            current.request.get().response.headers["Content-Type"] = "application/json; charset=utf-8"
            return json.dumps(text)
        raise errors.NotImplemented(f"{self.render.kind} rendering not implemented")


Assistant.json = True
Assistant.html = True

# Enforce AssistantSkel is loaded and initialized anywhere
from viur.assistant.skeletons.assistant import AssistantSkel  # noqa
