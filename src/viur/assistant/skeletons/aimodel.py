from viur.core.bones import *
from viur.core.skeleton import Skeleton


class AimodelSkel(Skeleton):
    model_id = StringBone(
        descr="Model ID",
        required=True,
        unique=UniqueValue(UniqueLockMethod.SameValue, False, "Model ID muss eindeutig sein"),
        indexed=True,
        searchable=True,
        params={"tooltip": "Technische Modell-ID, z. B. 'gemini-2.5-flash'."},
    )

    name = StringBone(
        descr="Anzeigename",
        required=True,
        searchable=True,
        params={"tooltip": "Wird im Agenten-Picker angezeigt."},
    )

    provider = SelectBone(
        descr="Provider",
        values={
            "anthropic": "Anthropic",
            "gemini": "Gemini",
            "openai": "OpenAI",
        },
        defaultValue="gemini",
        required=True,
    )

    deprecated = BooleanBone(
        descr="Deprecated",
        defaultValue=False,
        params={"tooltip": "Wird beim API-Sync gesetzt, wenn das Modell nicht mehr verfügbar ist."},
    )

    description = TextBone(
        descr="Beschreibung",
        readOnly=True,
        params={"tooltip": "Aus der Provider-API übernommen."},
    )

    input_token_limit = NumericBone(
        descr="Input Token Limit",
        readOnly=True,
    )

    output_token_limit = NumericBone(
        descr="Output Token Limit",
        readOnly=True,
    )
