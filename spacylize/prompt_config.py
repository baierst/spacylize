"""Prompt configuration loading module.

This module provides functionality to load and validate prompt configurations
from YAML files for LLM data generation tasks.
"""

import random
from pathlib import Path
from typing import Literal, Optional, List, Union
import yaml
from pydantic import BaseModel, ValidationError, Field, field_validator

from spacylize.llm_config import expand_env_vars


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
        temperature: Optional LLM sampling temperature for generation.
        domain: The domain of the generated texts, used for the topic check.
    """

    system: PromptMessage
    user: PromptMessage
    temperature: Optional[float] = None
    domain: Optional[str] = None

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
        min_entities: Minimum number of entity labels per sample. Each sample
            asks for a random subset of at least this many labels.
        domain: Description of the domain/topic for generated text.
        tone: Writing style (e.g., "professional", "casual", "technical").
        length: Expected length of generated text (e.g., "2-3 sentences").
        domain, tone and length can also be lists; one value is picked per sample.
        language: ISO language code (e.g., "en", "de", "es").
        temperature: Optional LLM temperature (0.0-1.0). Omitted if None.
        constraints: Additional rules or constraints for generation.
        examples: Few-shot examples to guide the LLM.
    """

    task: Literal["ner"] = "ner"
    entities: List[str] = Field(..., min_length=1)
    min_entities: int = Field(1, ge=1)
    domain: Union[str, List[str]]
    tone: Union[str, List[str]]
    length: Union[str, List[str]]
    language: Optional[str] = "en"
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0)
    constraints: Optional[List[str]] = []
    examples: Optional[List[NERExample]] = []

    @field_validator("entities")
    @classmethod
    def validate_entities(cls, v):
        """Ensure at least one entity is specified."""
        if not v:
            raise ValueError("At least one entity required")
        return v

    @property
    def labels(self) -> List[str]:
        """Configured entity labels."""
        return self.entities


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
        domain, tone and length can also be lists; one value is picked per sample.
        language: ISO language code (e.g., "en", "de", "es").
        temperature: Optional LLM temperature (0.0-1.0). Omitted if None.
        constraints: Additional rules or constraints for generation.
        examples: Few-shot examples to guide the LLM.
    """

    task: Literal["textcat"] = "textcat"
    categories: List[TextCatCategory] = Field(..., min_length=2)
    domain: Union[str, List[str]]
    tone: Union[str, List[str]]
    length: Union[str, List[str]]
    language: Optional[str] = "en"
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0)
    constraints: Optional[List[str]] = []
    examples: Optional[List[TextCatExample]] = []

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, v):
        """Ensure at least two categories are specified."""
        if len(v) < 2:
            raise ValueError("At least two categories required")
        return v

    @property
    def labels(self) -> List[str]:
        """Configured category names."""
        return [category.name for category in self.categories]


StructuredConfig = NERStructuredConfig | TextCatStructuredConfig


def render_prompt(config: StructuredConfig, rng=None, index=0) -> PromptConfig:
    """Render the prompts for one sample, varied per sample.

    Picks one value for list-valued domain, tone and length. For NER, asks for a
    random subset of at least min_entities labels. For textcat, picks the
    category by index, so samples rotate through the categories.

    Args:
        config: Structured configuration (NER or TextCat).
        rng: Random number generator (random.Random). A new one if None.
        index: Index of the sample, used to rotate the textcat categories.

    Returns:
        PromptConfig with rendered system and user prompts.

    Raises:
        ValueError: If the task type has no template.
    """
    from spacylize.templates import TemplateRegistry

    rng = rng or random.Random()
    values = config.model_dump()
    for key in ("domain", "tone", "length"):
        if isinstance(values[key], list):
            values[key] = rng.choice(values[key])
    if config.task == "ner":
        n = len(config.entities)
        k = rng.randint(min(config.min_entities, n), n)
        values["entities"] = rng.sample(config.entities, k)
    else:
        values["target_category"] = values["categories"][index % len(config.categories)]

    template_cls = TemplateRegistry.get_template(config.task)
    system_content, user_content = template_cls.render(values)

    domain = config.domain  # all values, for the topic check
    return PromptConfig(
        system=PromptMessage(role="system", content=system_content),
        user=PromptMessage(role="user", content=user_content),
        temperature=config.temperature,
        domain=" or ".join(domain) if isinstance(domain, list) else domain,
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

    structured_config = load_structured_config(path)
    prompt_config = render_prompt(structured_config)

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


def load_structured_config(path: Path) -> StructuredConfig:
    """Load and validate a structured prompt configuration from YAML.

    Unlike load_prompt_config, this does not render the prompts, so it can be
    used to read the task and labels, e.g. for validating a dataset.

    Args:
        path: Path to the structured config YAML file.

    Returns:
        NERStructuredConfig or TextCatStructuredConfig.

    Raises:
        RuntimeError: If the configuration is invalid or missing required fields.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    expanded = expand_env_vars(raw)

    task = expanded.get("task")
    if not task:
        raise RuntimeError(
            "Missing 'task' field in config. Must be 'ner' or 'textcat'."
        )

    try:
        if task == "ner":
            return NERStructuredConfig.model_validate(expanded)
        if task == "textcat":
            return TextCatStructuredConfig.model_validate(expanded)
    except ValidationError as e:
        raise RuntimeError(f"Invalid structured config:\n{e}") from e

    raise RuntimeError(f"Unsupported task: {task}. Must be 'ner' or 'textcat'.")
