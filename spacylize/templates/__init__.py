"""Template registry and task-specific template classes.

This module provides the registry for mapping task types to their
corresponding prompt templates.
"""

from spacylize.templates.base import PromptTemplate


class NERTemplate(PromptTemplate):
    """Prompt template for NER (Named Entity Recognition) tasks."""

    SYSTEM_TEMPLATE_FILE = "ner_system.jinja2"
    USER_TEMPLATE_FILE = "ner_user.jinja2"


class TextCatTemplate(PromptTemplate):
    """Prompt template for text classification tasks."""

    SYSTEM_TEMPLATE_FILE = "textcat_system.jinja2"
    USER_TEMPLATE_FILE = "textcat_user.jinja2"


class TemplateRegistry:
    """Registry mapping task types to template classes.

    Provides centralized lookup for retrieving the appropriate
    template class for a given task type.
    """

    _TEMPLATES = {
        "ner": NERTemplate,
        "textcat": TextCatTemplate,
    }

    @classmethod
    def get_template(cls, task: str):
        """Get the template class for a given task type.

        Args:
            task: The task type (e.g., 'ner', 'textcat').

        Returns:
            PromptTemplate subclass for the specified task.

        Raises:
            ValueError: If the task type is not supported.
        """
        if task not in cls._TEMPLATES:
            supported = ", ".join(cls._TEMPLATES.keys())
            raise ValueError(
                f"No template for task '{task}'. Supported: {supported}"
            )
        return cls._TEMPLATES[task]
