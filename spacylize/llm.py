"""LLM client wrapper for unified access to various language models.

This module provides a unified interface for interacting with different LLM
providers (OpenAI, Anthropic, Ollama, etc.) via the LiteLLM library.
"""

from typing import Optional
import dotenv
from litellm import completion

dotenv.load_dotenv()


class LLMClient:
    """Client for interacting with various LLM providers via LiteLLM.

    Supports OpenAI, Anthropic, Ollama, and other LiteLLM-compatible providers.

    Attributes:
        model: The model identifier (e.g., 'gpt-4o-mini', 'anthropic/claude-opus-4-5').
        api_key: API key for authentication.
        api_base: Optional custom API base URL for local or custom deployments.
        max_tokens: Maximum number of tokens to generate.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 1024,
    ):
        """Initialize the LLM client.

        Args:
            model: Model identifier for LiteLLM (e.g., 'gpt-4o-mini').
            api_key: API key for the LLM provider. Defaults to None.
            api_base: Custom API base URL for local or custom deployments.
            max_tokens: Maximum number of tokens to generate. Defaults to 1024.
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using the configured LLM.

        Args:
            prompt: The user prompt to send to the LLM.
            system_prompt: Optional system prompt to set context or instructions.

        Returns:
            str: The generated text response from the LLM.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        completion_kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }

        if self.api_key:
            completion_kwargs["api_key"] = self.api_key

        if self.api_base:
            completion_kwargs["api_base"] = self.api_base

        response = completion(**completion_kwargs)

        return response["choices"][0]["message"]["content"]


# # OpenAI
# llm_openai = LLMClient(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
# print(llm_openai.generate("Generate a random sentence."))
#
# # Anthropic
# llm_anthropic = LLMClient(
#     model="anthropic/claude-opus-4-5-20251101",
#     api_key=os.getenv("ANTHROPIC_API_KEY")
# )
# print(llm_anthropic.generate("Generate a random sentence."))
#
# # Local Ollama
# llm_local = LLMClient(
#     model="ollama/guanaco-7b",
#     api_base="http://localhost:11434"  # optional if default
# )
# print(llm_local.generate("Generate a random sentence."))
