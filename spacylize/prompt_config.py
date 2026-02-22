"""Prompt configuration loading module.

This module provides functionality to load and validate prompt configurations
from YAML files for LLM data generation tasks.
"""

from pathlib import Path
from typing import Literal, Any, Optional, List
import os
import re
import yaml
from pydantic import BaseModel, ValidationError, Field, field_validator


class PromptMessage(BaseModel):
    """A single prompt message with role and content.

    Attributes:
        role: The role of the message (system, user, or assistant).
        content: The text content of the message.
    """

    role: Literal["system", "user", "assistant"]
    content: str

    model_config = {"extra": "forbid"}


class PromptConfig(BaseModel):
    """Configuration model for prompt templates.

    Defines the structure for prompt configurations including system
    and user messages for LLM interactions.

    Attributes:
        system: System prompt message for setting LLM behavior.
        user: User prompt message for the main task instruction.
    """

    system: PromptMessage
    user: PromptMessage

    model_config = {"extra": "forbid"}


# Structured configuration models for template-based prompt generation


class NERExample(BaseModel):
    """Example for NER few-shot learning.

    Attributes:
        text: Example text with inline [ENTITY](LABEL) annotations.
        explanation: Optional explanation of what this example demonstrates.
    """

    text: str
    explanation: Optional[str] = None


class NERStructuredConfig(BaseModel):
    """Structured configuration for NER tasks.

    Users specify high-level parameters and templates generate the prompts.

    Attributes:
        task: Task type identifier (must be "ner").
        entities: List of entity labels to include in generated text.
        domain: Description of the domain/topic for generated text.
        tone: Writing style (e.g., "professional", "casual", "technical").
        length: Expected length of generated text (e.g., "2-3 sentences").
        language: ISO language code (e.g., "en", "de", "es").
        temperature: LLM temperature for generation (0.0-1.0).
        constraints: Additional rules or constraints for generation.
        examples: Few-shot examples to guide the LLM.
    """

    task: Literal["ner"] = "ner"
    entities: List[str] = Field(..., min_length=1)
    domain: str
    tone: Optional[str] = "professional"
    length: Optional[str] = "1-2 paragraphs"
    language: Optional[str] = "en"
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0)
    constraints: Optional[List[str]] = []
    examples: Optional[List[NERExample]] = []

    @field_validator("entities")
    @classmethod
    def validate_entities(cls, v):
        """Ensure at least one entity is specified."""
        if not v:
            raise ValueError("At least one entity required")
        return v


class TextCatCategory(BaseModel):
    """Category definition for text classification.

    Attributes:
        name: Category name/label.
        description: Description of what this category includes.
    """

    name: str
    description: str


class TextCatExample(BaseModel):
    """Example for TextCat few-shot learning.

    Attributes:
        text: Example text to classify.
        category: The correct category for this text.
    """

    text: str
    category: str


class TextCatStructuredConfig(BaseModel):
    """Structured configuration for text classification tasks.

    Users specify high-level parameters and templates generate the prompts.

    Attributes:
        task: Task type identifier (must be "textcat").
        categories: List of category definitions.
        domain: Description of the domain/topic for generated text.
        tone: Writing style (e.g., "professional", "casual", "marketing").
        length: Expected length of generated text (e.g., "2-3 sentences").
        language: ISO language code (e.g., "en", "de", "es").
        temperature: LLM temperature for generation (0.0-1.0).
        constraints: Additional rules or constraints for generation.
        examples: Few-shot examples to guide the LLM.
    """

    task: Literal["textcat"] = "textcat"
    categories: List[TextCatCategory] = Field(..., min_length=2)
    domain: str
    tone: Optional[str] = "professional"
    length: Optional[str] = "2-3 sentences"
    language: Optional[str] = "en"
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0)
    constraints: Optional[List[str]] = []
    examples: Optional[List[TextCatExample]] = []

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, v):
        """Ensure at least two categories are specified."""
        if len(v) < 2:
            raise ValueError("At least two categories required")
        return v


StructuredConfig = NERStructuredConfig | TextCatStructuredConfig


_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variable references in configuration values.

    Replaces ${VAR_NAME} patterns with their environment variable values.
    Supports nested dictionaries and lists.

    Args:
        value: Configuration value to process (str, dict, list, or other).

    Returns:
        The value with environment variables expanded.
    """
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


def _render_structured_config(config: StructuredConfig) -> PromptConfig:
    """Render structured config to PromptConfig using templates.

    Args:
        config: Structured configuration (NER or TextCat).

    Returns:
        PromptConfig with rendered system and user prompts.

    Raises:
        ValueError: If the task type has no template.
    """
    from spacylize.templates import TemplateRegistry

    template_cls = TemplateRegistry.get_template(config.task)
    config_dict = config.model_dump()
    system_content, user_content = template_cls.render(config_dict)

    return PromptConfig(
        system=PromptMessage(role="system", content=system_content),
        user=PromptMessage(role="user", content=user_content),
    )


def load_prompt_config(
    path: Path, output_folder: Optional[Path] = None
) -> PromptConfig:
    """Load and render structured prompt configuration from YAML.

    This function loads a structured configuration file and uses Jinja2 templates
    to render the final system and user prompts. The structured format allows
    users to specify high-level parameters (entities, domain, tone, etc.) while
    templates handle the prompt engineering.

    Args:
        path: Path to the structured config YAML file.
        output_folder: Optional folder to write rendered prompts for user verification.

    Returns:
        PromptConfig with rendered system and user prompts.

    Raises:
        RuntimeError: If the configuration is invalid or missing required fields.
    """
    from loguru import logger

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    expanded = _expand_env_vars(raw)

    try:
        task = expanded.get("task")
        if not task:
            raise RuntimeError(
                "Missing 'task' field in config. Must be 'ner' or 'textcat'."
            )

        if task == "ner":
            structured_config = NERStructuredConfig.model_validate(expanded)
        elif task == "textcat":
            structured_config = TextCatStructuredConfig.model_validate(expanded)
        else:
            raise RuntimeError(f"Unsupported task: {task}. Must be 'ner' or 'textcat'.")

        prompt_config = _render_structured_config(structured_config)

        # Write rendered prompts to output folder for user verification
        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)

            system_file = output_folder / "system_prompt.txt"
            user_file = output_folder / "user_prompt.txt"

            system_file.write_text(prompt_config.system.content)
            user_file.write_text(prompt_config.user.content)

            logger.info(f"Rendered prompts saved to {output_folder}/")

        return prompt_config

    except ValidationError as e:
        raise RuntimeError(f"Invalid structured config:\n{e}") from e
