import logging
import typing as t

from viur.core.config import ConfigType

ASSISTANT_LOGGER: logging.Logger = logging.getLogger("viur.assistant")


class AssistantConfig(ConfigType):
    """Configuration for viur-assistant plugin."""

    api_openai_key: str = None
    """API Key for OpenAi"""

    api_anthropic_key: str = None
    """API Key for Anthropic"""

    api_gemini_key: str = None
    """API Key for Gemini"""

    default_provider: t.Literal["openai", "anthropic", "gemini"] = "openai"
    """Default LLM provider to use"""

    language_map: t.Dict[str, str] = {
        "de": "German",
        "de-DE-x-simple-language": "Deutsch, einfache Sprache",
        "en": "English",
    }
    """
    Maps language codes to human-readable language names or specific language variants.

    This dictionary allows mapping short language identifiers (e.g., BCP 47 language tags)
    to more descriptive names. It supports both standard language codes (like "de" for German)
    and extended tags (like "de-DE-x-simple-language" for simplified German).

    Useful for precise language definitions and for differentiating
    between standard and stylistic variants of a language.
    """

    translate_language_characteristics: dict[str, t.Iterable[str]] = {
        "*": [
            "not vulgar",
        ],
        "simple": [
            "simplified",
            "short sentences",
            "one information per sentence",
            "simple words",
            "avoid negative language",
            "sentences should be active, not passive",
            "avoid subjunctive clause",
            "one sentence per line",
            "split compound nouns with hyphens",
        ]
    }
    """
    Dictionary for defining style characteristics of translations.

    Each key represents a named translation style (e.g., "simple"), and its value
    is a list of stylistic rules or constraints that should be followed when generating
    a translation in that style.

    The special key "*" defines rules that are *always* applied in addition to any
    specific style. These are base-level constraints that apply globally, regardless of
    the selected style.

    Example:
    - The "simple" characteristic focuses on clarity and accessibility,
      encouraging short, active-voice sentences with simple vocabulary and structure.
    - The "*" characteristic enforces universally that the output must not be vulgar.

    This structure allows combining a base set of rules with additional style-specific ones.
    """

    describe_image_jpeg_quality_default = 50
    """
    Default JPEG compression quality used when resizing and encoding images.

    A value of 50 strikes a balance between file size and visual clarity,
    ensuring efficient transmission and sufficient detail for AI-based image analysis.
    Valid range is 0 (lowest quality) to 100 (best quality).
    """

    describe_image_pixel_default = 100_000
    """
    Default target pixel count (width × height) used when resizing an image
    for AI-based description. Balances sufficient detail with low token cost.
    Used by `_get_resized_image_bytes` if no other value is provided.
    """

    # TODO: make client set-able?
    '''
    describe_image_pixel_min = 1_000
    """
    Minimum allowed pixel count when resizing an image.
    Prevents extremely small images that may lack meaningful visual information
    for AI-based interpretation.
    """

    describe_image_pixel_max = 200_000
    """
    Maximum allowed pixel count when resizing an image.
    Limits image resolution to avoid high token usage and ensure efficient
    processing when generating AI-based descriptions.
    """

    '''


CONFIG: t.Final[AssistantConfig] = AssistantConfig(
    strict_mode=True,
)
"""
The viur-assistent config instance.
"""
