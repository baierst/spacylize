from typing import Optional
import dotenv
from litellm import completion

dotenv.load_dotenv()


class LLMClient:

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.api_key = api_key or ""
        self.api_base = api_base
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = completion(
            model=self.model,
            messages=messages,
            api_key=self.api_key,
            api_base=self.api_base,
            max_tokens=self.max_tokens,
        )

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
