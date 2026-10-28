"""Base classes for prompt template system.

This module provides the foundation for task-specific prompt templating
using Jinja2 templates.
"""

from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from typing import Dict, Any


class PromptTemplate:
    """Base class for task-specific prompt templates.

    Subclasses must define SYSTEM_TEMPLATE_FILE and USER_TEMPLATE_FILE
    class attributes pointing to Jinja2 template files.
    """

    SYSTEM_TEMPLATE_FILE: str = ""
    USER_TEMPLATE_FILE: str = ""

    @classmethod
    def _get_template_dir(cls) -> Path:
        """Get the directory containing template files.

        Returns:
            Path to the templates directory.
        """
        return Path(__file__).parent

    @classmethod
    def render(cls, config: Dict[str, Any]) -> tuple[str, str]:
        """Render system and user prompts from template files.

        Args:
            config: Dictionary containing template variables from structured config.

        Returns:
            tuple: (system_prompt, user_prompt) as rendered strings.

        Raises:
            jinja2.TemplateNotFound: If template files are missing.
        """
        template_dir = cls._get_template_dir()
        env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        system_template = env.get_template(cls.SYSTEM_TEMPLATE_FILE)
        user_template = env.get_template(cls.USER_TEMPLATE_FILE)

        system_prompt = system_template.render(**config)
        user_prompt = user_template.render(**config)

        return system_prompt, user_prompt
