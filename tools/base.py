import inspect
import logging
from typing import Callable, Dict, get_type_hints

logger = logging.getLogger(__name__)

TOOL_MARKER = "_is_tool"

_TYPE_MAP: Dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def tool(func: Callable) -> Callable:
    """
    Декоратор для автоматической регистрации функции как инструмента LLM.
    Генерирует OpenAI JSON Schema на основе подсказок типов и docstring.
    """
    setattr(func, TOOL_MARKER, True)

    name = func.__name__
    doc = func.__doc__ or ""
    description = doc.strip().split("\n")[0] if doc else "No description provided."

    type_hints = get_type_hints(func)
    type_hints.pop("return", None)

    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    sig = inspect.signature(func)
    for param_name, param in sig.parameters.items():
        p_type = type_hints.get(param_name, str)
        json_type = _TYPE_MAP.get(p_type, "string")

        param_desc = ""
        for line in doc.split("\n"):
            if f"{param_name}:" in line or f"{param_name} (" in line:
                param_desc = (
                    line.split(":", 1)[-1].strip() if ":" in line else line.strip()
                )
                break

        parameters["properties"][param_name] = {
            "type": json_type,
            "description": param_desc or f"Parameter {param_name}",
        }

        if param.default is inspect.Parameter.empty:
            parameters["required"].append(param_name)

    func.openai_schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }

    return func
