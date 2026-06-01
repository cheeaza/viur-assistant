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
    """
    Provides LLM-powered utilities within a ViUR module context.

    This class includes various AI-assisted features such as script generation,
    image description, and language-aware prompting. It is designed to be used
    in the context of a ViUR admin and integrates with services like
    OpenAI, Anthropic and Google Gemini.

    Responsibilities:
      - Generating backend scripts from user instructions and data models.
      - Producing accessible image descriptions (e.g., for alt attributes).
      - Translate text.
      - Managing prompt and API settings.

    Configuration can be made in
     - this singleton skel itself.
     - The backend configuration `CONFIG`.

    .. note::
       The methods in this module assumes a properly configured environment
       with API keys and AI model settings. However, you only need to
       configure what is actually used.
       Error handling is included for common failure modes such as
       misconfiguration or service unavailability.
    """
    kindName: t.Final[str] = "viur-assistant"

    def _get_provider_and_model(self, skel) -> tuple[BaseProvider, str]:
        """Resolve provider and model_id from the skel's ai_model relation."""
        dest = ((skel.get("ai_model") or {}).get("dest") or {})
        model_id = dest.get("model_id") or ""
        if not model_id:
            raise errors.InternalServerError(descr="No AI model configured in assistant settings.")
        return get_provider(dest.get("provider", "openai")), model_id

    @exposed
    @access("admin")
    @force_post
    def generate_script(
        self,
        *,
        prompt: str,
        modules_to_include: list[str] = None,
        enable_caching: bool = False,
        max_thinking_tokens: int = 0
    ):
        """
        Generates a script based on a user prompt and optional module structures using a language model.

        This method builds a structured prompt for the LLM and optionally includes
        application-specific module metadata to enrich the generation context. Additional configuration
        such as caching behavior and token budgeting for "thinking steps" can be provided.

        :param prompt: The main user instruction or query that guides the script generation.
        :param modules_to_include: Optional list of module names whose structures
            should be included as part of the model context.
            These are injected into the LLM prompt as JSON.
        :param enable_caching: If set to True, instructs the system to use ephemeral caching for the
            scriptor documentation prompt section (provider-dependent).
        :param max_thinking_tokens: If greater than 0, enables the model's "thinking" feature with a
            token budget for intermediate reasoning or planning steps.
        :return: A JSON-encoded string of the model's response, typically containing the generated script.

        :raises InternalServerError:
          - If configuration (`skel`) is missing.
          - If the LLM request fails due to connection or model errors.
        """
        if not (skel := self.getContents()):
            raise errors.InternalServerError(descr="Configuration missing")

        messages = []

        # add module structures
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
        """
        Collect and return ViUR module structures for a given list of module names.

        For each named module, its structure is extracted if it is of type ``List`` or ``Tree``.
        ``Tree`` structures will return separate entries for ``node`` and ``leaf`` skeletons.

        :param modules_to_include: List of ViUR module names to retrieve structures for.
        :return: A dictionary mapping module names to their respective structure definitions.
            For ``Tree`` modules, nested keys ``"node"`` and ``"leaf"`` are returned.

        :raises ValueError: If a module exists but is not a supported type (i.e., not ``List`` or ``Tree``).

        .. note::
           Modules that are missing or not found are silently skipped.
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
                        "leaf": module.structure(skelType="leaf")
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
        """
        Translate a given text into a target language, optionally using a specific style.

        :param text: The source text to translate.
        :param language: The target language code (e.g. ``"de"``, ``"en"``, ``"de-x-simple"``).
        :param characteristic: Optional translation style (e.g. ``"simplified"``, ``"formal"``, etc.)
            as defined in ``CONFIG.translate_language_characteristics``.
        :return: Translated text as a plain string. HTML tags from the original text are preserved.

        :raises InternalServerError: If configuration is missing.
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
        """
        Generate an HTML ``alt`` attribute description for a given image.

        :param filekey: Key identifying the image file in the ViUR file module.
        :param prompt: Optional user-defined hint for how the image should be interpreted.
        :param context: Optional additional background information.
        :param language: Target language code. Falls back to the current session language.
        :return: A plain-text string suitable for use in an HTML ``alt`` attribute.

        :raises InternalServerError: If required configuration is missing or the provider
            does not support vision.
        :raises NotFound: If the referenced image file could not be loaded.
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
                                "detail": "low",
                                # "low" = 85 Tokens, "high" = calculated differently
                                # "low" = resize (on OpenAI's side) to < 512x512px
                                # https://platform.openai.com/docs/guides/images?api-mode=chat#calculating-costs
                            },
                        },
                    ],
                }],
            )
        except Exception as e:
            logger.exception(e)
            raise errors.InternalServerError(descr=str(e))

        return self.render_text(result)

    def _get_resized_image_bytes(
        self,
        image: t.IO[bytes] | str | bytes | "os.PathLike[str]" | "os.PathLike[bytes]",
        target_pixel_count: int,
        jpeg_quality: int = 50,
    ):
        """
        Resize an image to approximately match a target total pixel count and return it as a JPEG byte stream.

        :param image: Input image as file-like object, byte string, or file path.
        :param target_pixel_count: Desired total pixel count (width × height).
        :param jpeg_quality: JPEG compression quality (0–100). Default is 50.
        :return: Resized JPEG image as bytes.

        :raises ValueError: If ``jpeg_quality`` is out of range or the image input is invalid.
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
                PIL.Image.Resampling.LANCZOS
            )

        result_bio = io.BytesIO()
        resized_img.save(result_bio, "jpeg", quality=jpeg_quality)
        result_bio.seek(0)
        return result_bio.read()

    def render_text(self, text: str) -> t.Any:
        """
        Render the given text for the current renderer.

        :param text: The text to render.
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
