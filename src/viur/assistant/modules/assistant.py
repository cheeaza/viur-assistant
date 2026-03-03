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
from viur.assistant.llm.provider import LLMProviderInterface, LLMRequest, LLMMessage

logger = ASSISTANT_LOGGER.getChild(__name__)


class Assistant(Singleton):
    """
    Provides LLM-powered utilities within a ViUR module context.

    This class includes various AI-assisted features such as script generation,
    image description, and language-aware prompting. It is designed to be used
    in the context of a ViUR admin and integrates with services like
    OpenAI and Anthropic.

    Responsibilities:
      - Generating backend scripts from user instructions and data models.
      - Producing accessible image descriptions (e.g., for alt attributes).
      - Translate text.
      - Managing prompt and API settings.

    Integrated Services:
      - OpenAI (e.g., GPT) for image description generation and translations.
      - Anthropic Claude for structured script generation with reasoning capabilities.

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

    def _get_provider(self, provider_type: str | None = None) -> LLMProviderInterface:
        """
        Returns the LLM provider instance based on the configuration or specified type.
        """
        if not provider_type:
            if skel := self.getContents():
                provider_type = skel["provider"]
            else:
                provider_type = CONFIG.default_provider

        return LLMProviderInterface.create(provider_type, CONFIG)

    @exposed
    @access("admin")
    @force_post
    def chat(
        self,
        *,
        prompt: str,
        context: str,
        enable_caching: bool = False,
        max_thinking_tokens: int = 0
    ):
        """
        Sends a simple text request to an LLM and returns the response.

        This method provides a generic chat endpoint that sends a user request together
        with a system instruction to the configured LLM. Depending on the model, the
        response can be plain text or a JSON string.

        :param prompt: The user text (user string) sent to the LLM.
        :param context: Additional contextual information to provide to the LLM.
        :param enable_caching: Optional; enables model-specific caching if supported.
        :param max_thinking_tokens: Optional; budget for additional “thinking” tokens if supported.
        :return: Text or JSON string as the LLM response.

        :raises InternalServerError:
          - If the configuration is missing.
          - If the LLM request fails.
        """
        raise NotImplementedError("Not yet implemented")

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
            scriptor documentation prompt section.
        :param max_thinking_tokens: If greater than 0, enables the model's "thinking" feature with a
            token budget for intermediate reasoning or planning steps.
        :return: A JSON-encoded string of the model's response, typically containing the generated script.

        :raises InternalServerError:
          - If configuration (`skel`) is missing.
          - If the LLM request fails due to connection or model errors.

        .. note::
         - Requires a valid configuration in the current context.
         - The actual parsing of the generated code (e.g., extracting specific script content)
           is currently marked as a TODO and has to be discussed.
        """
        if not (skel := self.getContents()):
            raise errors.InternalServerError(descr="Configuration missing")

        provider = self._get_provider()
        provider_type = skel["provider"]

        request = LLMRequest(
            model=skel[f"model"],
            temperature=skel[f"temperature"],
            system_prompt=skel[f"system_prompt"],
        )

        provider.prepare_script_request(
            request,
            skel=skel,
            max_thinking_tokens=max_thinking_tokens,
            enable_caching=enable_caching
        )

        # add module structures
        user_content = []
        if modules_to_include is not None and (structures := self.get_viur_structures(modules_to_include)):
            user_content.append({
                "type": "text",
                "text": json.dumps({
                    "module_structures": structures
                }, indent=2)
            })

        # finally, append user prompt
        user_content.append({
            "type": "text",
            "text": prompt
        })

        request.messages = [LLMMessage(role="user", content=user_content)]

        try:
            response = provider.generate_response(request)
        except Exception as e:
            logger.exception(e)
            raise errors.InternalServerError(descr=str(e))

        logger.debug(f"{response=}")
        current.request.get().response.headers["Content-Type"] = "application/json"

        return provider.parse_script_response(response)

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

        This method sends the input text to the LLM with instructions to
        translate it into the requested language, optionally applying predefined translation
        characteristics such as simplification.

        :param text: The source text to translate.
        :param language: The target language code (e.g. ``"de"``, ``"en"``, ``"de-x-simple"``).
        :param characteristic: Optional translation style (e.g. ``"simplified"``, ``"formal"``, etc.)
            as defined in ``CONFIG.translate_language_characteristics``.
        :return: Translated text as a plain string. HTML tags from the original text are preserved.

        :raises InternalServerError: If configuration is missing.

        .. note::
           - The translation style is determined by merging base rules (`*`) and the selected characteristic.
           - The returned translation contains only the translated text, with no explanation or additional formatting.
        """
        if not (skel := self.getContents()):
            raise errors.InternalServerError(descr="Configuration missing")

        characteristics = [
            *CONFIG.translate_language_characteristics.get("*", []),
            *CONFIG.translate_language_characteristics.get(characteristic, []),
        ]

        provider = self._get_provider()
        provider_type = skel["provider"]

        request = LLMRequest(
            model=skel[f"{provider_type}_model"],
            messages=[LLMMessage(
                role="user",
                content=(
                    f"Translate the following text into {CONFIG.language_map.get(language, language)}"
                    f" ({'. '.join(characteristics)})"
                    f" and only return the translation, keep HTML-tags (if there are any):\n\n{text}\n"
                )
            )]
        )

        provider.prepare_translation_request(request)

        response = provider.generate_response(request)
        content = provider.parse_translation_response(response)

        return self.render_text(content)

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
        Generate an HTML ``alt`` attribute description for a given image using LLM.

        This method reads an image via its filekey, resizes it to a configured pixel target,
        and sends it along with optional prompt and context data to a model.
        The model returns a plain-text alternative description for accessibility purposes,
        formatted as an HTML ``alt`` text in the specified language.

        :param filekey: Key identifying the image file in the ViUR file module.
            (Key of the :class:`file skeleton <viur.core.modules.file.FileLeafSkel>`).
        :param prompt: Optional user-defined hint or instruction for how the image should be interpreted.
        :param context: Optional additional background information to support a better description.
        :param language: Target language code for the generated description (e.g., ``"en"``, ``"de-x-simple"``).
            Falls back to the current session language if not specified.
        :return: A plain-text string suitable for use in an HTML ``alt`` attribute (no quotes or labels).

        :raises InternalServerError: If required configuration is missing.
        :raises NotFound: If the referenced image file could not be loaded.

        .. note::
          - The image is resized and converted to JPEG before being sent to the model.
          - This function uses a preconfigured model, pixel count, and JPEG quality settings.
        """
        if not (skel := self.getContents()):
            raise errors.InternalServerError(descr="Configuration missing")

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

        provider = self._get_provider()
        provider_type = skel["provider"]

        content_text = (
            f"Analyze the image and generate an appropriate HTML alt attribute"
            f" in language: {CONFIG.language_map.get(language, language)}."
            f" Provide only the plain text for the alt attributes without quotes and label.\n\n"
            f"{context_prompt}\n"
        )

        content = provider.build_image_description_content(
            text=content_text,
            base64_image=base64_image
        )

        request = LLMRequest(
            model=skel[f"{provider_type}_model"],
            messages=[LLMMessage(role="user", content=content)]
        )

        provider.prepare_image_description_request(request)

        response = provider.generate_response(request)
        content = provider.parse_image_description_response(response)

        return self.render_text(content)

    def _get_resized_image_bytes(
        self,
        image: t.IO[bytes] | str | bytes | "os.PathLike[str]" | "os.PathLike[bytes]",
        target_pixel_count: int,
        jpeg_quality: int = 50,
    ):
        """
        Resize an image to approximately match a target total pixel count and return it as a JPEG byte stream.

        The image is scaled proportionally to meet the target pixel count (width × height),
        while preserving the original aspect ratio. If the resized dimensions would exceed
        the original image size, no upscaling is performed. The result is encoded as a JPEG.

        :param image: Input image, provided as a file-like object, byte string, file path, or raw bytes.
        :param target_pixel_count: Desired total number of pixels for the resized image.
            Used to compute the new dimensions.
        :param jpeg_quality: JPEG compression quality (0 to 100).
            Higher values yield better image quality at the cost of file size. Default is 50.
        :return: The resized and JPEG-compressed image as a byte stream.

        :raises ValueError: If `jpeg_quality` is not in the 0–100 range or the image input is invalid.

        .. note::
         - Images in PNG, SVG, or WEBP format are automatically converted to RGB JPEG.
         - This function is intended for preprocessing images before passing them to
           AI models, balancing detail and data size.
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
        Render the give text as usual for the current renderer.

        If the current renderer is the JSON renderer, JSON string is returned
        and the content-type header is also set to JSON.

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
