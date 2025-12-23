from pathlib import Path
from typing import Literal, Any
import os
import re
import yaml
from pydantic import BaseModel, ValidationError


class PromptMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

    model_config = {
        "extra": "forbid"
    }


class PromptConfig(BaseModel):
    system: PromptMessage
    user: PromptMessage

    model_config = {
        "extra": "forbid"
    }


_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        match = _ENV_VAR_PATTERN.fullmatch(value)
        if match:
            return os.getenv(match.group(1))
        return value

    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]

    return value


def load_prompt_config(path: Path) -> PromptConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    expanded = _expand_env_vars(raw)

    try:
        return PromptConfig.model_validate(expanded)
    except ValidationError as e:
        raise RuntimeError(f"Invalid prompt configuration:\n{e}") from e
