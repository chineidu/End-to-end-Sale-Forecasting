from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict  # type: ignore
from pydantic.alias_generators import to_camel


def round_probability(value: float) -> float:
    """Round a float value to two decimal places.

    Returns:
        float: Rounded value.
    """
    if isinstance(value, float):
        return round(value, 2)
    return value


def strip_string(text: str) -> str:
    """Strip whitespace from beginning and end of a string.

    Parameters
    ----------
    text : str
        The input string to be stripped.

    Returns
    -------
    str
        The string with leading and trailing whitespace removed.
    """
    return text.strip()


class BaseSchema(BaseModel):
    """Base schema class that inherits from Pydantic BaseModel.

    This class provides common configuration for all schema classes including
    camelCase alias generation, population by field name, and attribute mapping.
    """

    model_config: ConfigDict = ConfigDict(  # type: ignore
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
    )


Float = Annotated[float, BeforeValidator(round_probability)]
String = Annotated[str, BeforeValidator(strip_string)]
