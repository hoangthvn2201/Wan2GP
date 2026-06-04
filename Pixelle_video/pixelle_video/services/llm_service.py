# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LLM (Large Language Model) Service - Direct OpenAI SDK implementation

Supports structured output via response_type parameter (Pydantic model).
"""

import json
import re
from typing import Optional, Type, TypeVar, Union

from openai import AsyncOpenAI
from pydantic import BaseModel
from loguru import logger


T = TypeVar("T", bound=BaseModel)

# Matches reasoning/"thinking" blocks that some models (Qwen3, DeepSeek-R1)
# leak into the message content, e.g. "<think>...</think>{json}".
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLMService:
    """
    LLM (Large Language Model) service
    
    Direct implementation using OpenAI SDK. No capability layer needed.
    
    Supports all OpenAI SDK compatible providers:
    - OpenAI (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
    - Alibaba Qwen (qwen-max, qwen-plus, qwen-turbo)
    - Anthropic Claude (claude-sonnet-4-5, claude-opus-4, claude-haiku-4)
    - DeepSeek (deepseek-chat)
    - Moonshot Kimi (moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k)
    - Ollama (llama3.2, qwen2.5, mistral, codellama) - FREE & LOCAL!
    - Any custom provider with OpenAI-compatible API
    
    Usage:
        # Direct call
        answer = await pixelle_video.llm("Explain atomic habits")
        
        # With parameters
        answer = await pixelle_video.llm(
            prompt="Explain atomic habits in 3 sentences",
            temperature=0.7,
            max_tokens=2000
        )
    """
    
    def __init__(self, config: dict):
        """
        Initialize LLM service
        
        Args:
            config: Full application config dict (kept for backward compatibility)
        """
        # Note: We no longer cache config here to support hot reload
        # Config is read dynamically from config_manager in _get_config_value()
        self._client: Optional[AsyncOpenAI] = None
    
    def _get_config_value(self, key: str, default=None):
        """
        Get config value dynamically from config_manager (supports hot reload)
        
        Args:
            key: Config key name
            default: Default value if not found
        
        Returns:
            Config value
        """
        from Pixelle_video.pixelle_video.config import config_manager
        return getattr(config_manager.config.llm, key, default)
    
    def _create_client(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> AsyncOpenAI:
        """
        Create OpenAI client
        
        Args:
            api_key: API key (optional, uses config if not provided)
            base_url: Base URL (optional, uses config if not provided)
        
        Returns:
            AsyncOpenAI client instance
        """
        # Get API key (priority: parameter > config)
        final_api_key = (
            api_key
            or self._get_config_value("api_key")
            or "dummy-key"  # Ollama doesn't need real key
        )
        
        # Get base URL (priority: parameter > config)
        final_base_url = (
            base_url
            or self._get_config_value("base_url")
        )
        
        # Create client
        client_kwargs = {"api_key": final_api_key}
        if final_base_url:
            client_kwargs["base_url"] = final_base_url
        
        return AsyncOpenAI(**client_kwargs)

    def _inject_thinking_control(self, kwargs: dict) -> dict:
        """
        Inject reasoning-model thinking control into the request kwargs.

        When `llm.enable_thinking` is configured (True/False), pass it through
        `extra_body.chat_template_kwargs.enable_thinking` so OpenAI-compatible
        servers (vLLM/SGLang serving Qwen3, DeepSeek-R1, etc.) toggle their
        chain-of-thought. When it is None (default), nothing is injected, so
        providers that don't understand the flag (e.g. OpenAI) are untouched.

        A caller-supplied `extra_body` is respected and never overwritten.
        """
        enable_thinking = self._get_config_value("enable_thinking", None)
        if enable_thinking is None:
            return kwargs

        kwargs = dict(kwargs)
        extra_body = dict(kwargs.get("extra_body") or {})
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        # Don't clobber an explicit caller override.
        chat_template_kwargs.setdefault("enable_thinking", enable_thinking)
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        kwargs["extra_body"] = extra_body
        return kwargs

    def _extract_content(self, response, model: str, max_tokens: int) -> str:
        """
        Extract assistant text from a chat completion, with clear diagnostics.

        Reasoning models can return an empty/`None` `message.content` when they
        spend the entire token budget on chain-of-thought (`finish_reason ==
        "length"`) or when the answer is emitted into a separate
        `reasoning_content` field. Returning `None` here causes a cryptic
        `len(None)` crash downstream, so we raise an actionable error instead.

        Returns:
            Non-empty assistant content (with any leaked <think> block stripped).

        Raises:
            ValueError: If the model returned no usable content.
        """
        if not response.choices:
            raise ValueError(f"LLM returned no choices (model={model}).")

        choice = response.choices[0]
        message = choice.message
        content = (message.content or "").strip()
        finish_reason = getattr(choice, "finish_reason", None)

        if content:
            return _THINK_RE.sub("", content).strip() or self._raise_empty(
                model, finish_reason, message
            )

        return self._raise_empty(model, finish_reason, message)

    def _raise_empty(self, model: str, finish_reason, message) -> str:
        """Raise a descriptive error when no usable content was returned."""
        # vLLM/SGLang call it reasoning_content; Ollama's OpenAI-compat layer
        # calls it reasoning. Extra fields land in model_extra on the SDK object.
        reasoning = getattr(message, "reasoning_content", None) or getattr(
            message, "reasoning", None
        )
        if reasoning is None and getattr(message, "model_extra", None):
            reasoning = message.model_extra.get("reasoning_content") or message.model_extra.get("reasoning")

        if finish_reason == "length":
            hint = (
                "the model hit max_tokens before emitting an answer. This is "
                "common with reasoning models (Qwen3, DeepSeek-R1, MiniMax-M3) "
                "that spend the token budget on chain-of-thought. Increase "
                "max_tokens, or set llm.enable_thinking=false in your config "
                "to disable thinking."
            )
        elif reasoning:
            hint = (
                "the model spent the whole token budget on reasoning and "
                "emitted no final answer. Increase max_tokens, or set "
                "llm.enable_thinking=false in your config (note: the flag is "
                "only honored by vLLM/SGLang-style servers, not by Ollama)."
            )
        else:
            hint = (
                "the provider returned empty content. Check the model name and "
                "that the endpoint is OpenAI-compatible. If this is a reasoning "
                "model, the token budget may have been consumed by hidden "
                "chain-of-thought - try a larger max_tokens."
            )

        raise ValueError(
            f"LLM returned no content (model={model}, finish_reason={finish_reason}): {hint}"
        )

    async def _chat_with_retry(
        self,
        client: AsyncOpenAI,
        model: str,
        messages: list,
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> str:
        """
        Run a chat completion, retrying once with a larger token budget when
        the model returns empty content.

        Reasoning models (MiniMax-M3, Qwen3, DeepSeek-R1, ...) emit hidden
        chain-of-thought that counts against max_tokens. A small budget (e.g.
        the 50 tokens historically used for title generation) is then entirely
        consumed by thinking and `message.content` comes back empty. One retry
        with a boosted budget recovers transparently.
        """
        request_kwargs = self._inject_thinking_control(kwargs)

        async def _attempt(budget: int) -> str:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=budget,
                **request_kwargs,
            )
            return self._extract_content(response, model, budget)

        try:
            return await _attempt(max_tokens)
        except ValueError as e:
            if "LLM returned no content" not in str(e):
                raise
            boosted = max(min(max_tokens * 8, 8192), 2000)
            if boosted <= max_tokens:
                raise
            logger.warning(
                f"LLM returned no content with max_tokens={max_tokens} "
                f"(reasoning model spent the budget on chain-of-thought?). "
                f"Retrying once with max_tokens={boosted}..."
            )
            return await _attempt(boosted)

    async def __call__(
        self,
        prompt: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_type: Optional[Type[T]] = None,
        **kwargs
    ) -> Union[str, T]:
        """
        Generate text using LLM
        
        Args:
            prompt: The prompt to generate from
            api_key: API key (optional, uses config if not provided)
            base_url: Base URL (optional, uses config if not provided)
            model: Model name (optional, uses config if not provided)
            temperature: Sampling temperature (0.0-2.0). Lower is more deterministic.
            max_tokens: Maximum tokens to generate
            response_type: Optional Pydantic model class for structured output.
                          If provided, returns parsed model instance instead of string.
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Generated text (str) or parsed Pydantic model instance (if response_type provided)
        
        Examples:
            # Basic text generation
            answer = await pixelle_video.llm("Explain atomic habits")
            
            # Structured output with Pydantic model
            class MovieReview(BaseModel):
                title: str
                rating: int
                summary: str
            
            review = await pixelle_video.llm(
                prompt="Review the movie Inception",
                response_type=MovieReview
            )
            print(review.title)  # Structured access
        """
        # Create client (new instance each time to support parameter overrides)
        client = self._create_client(api_key=api_key, base_url=base_url)
        
        # Get model (priority: parameter > config)
        final_model = (
            model
            or self._get_config_value("model")
            or "gpt-3.5-turbo"  # Default fallback
        )
        
        logger.debug(f"LLM call: model={final_model}, base_url={client.base_url}, response_type={response_type}")
        
        try:
            if response_type is not None:
                # Structured output mode - try beta.chat.completions.parse first
                return await self._call_with_structured_output(
                    client=client,
                    model=final_model,
                    prompt=prompt,
                    response_type=response_type,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            else:
                # Standard text output mode (retries once with a larger budget
                # if a reasoning model returns empty content)
                result = await self._chat_with_retry(
                    client=client,
                    model=final_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                logger.debug(f"LLM response length: {len(result)} chars")

                return result
        
        except Exception as e:
            logger.error(f"LLM call error (model={final_model}, base_url={client.base_url}): {e}")
            raise
    
    async def _call_with_structured_output(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: str,
        response_type: Type[T],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> T:
        """
        Call LLM with structured output support
        
        Uses JSON schema instruction appended to prompt for maximum compatibility
        across all OpenAI-compatible providers (Qwen, DeepSeek, etc.).
        
        Args:
            client: OpenAI client
            model: Model name
            prompt: The prompt
            response_type: Pydantic model class
            temperature: Sampling temperature
            max_tokens: Max tokens
            **kwargs: Additional parameters
        
        Returns:
            Parsed Pydantic model instance
        """
        # Build JSON schema instruction and append to prompt
        json_schema_instruction = self._get_json_schema_instruction(response_type)
        enhanced_prompt = f"{prompt}\n\n{json_schema_instruction}"

        # Call LLM with enhanced prompt (retries once with a larger budget
        # if a reasoning model returns empty content)
        content = await self._chat_with_retry(
            client=client,
            model=model,
            messages=[{"role": "user", "content": enhanced_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        logger.debug(f"Structured output response length: {len(content)} chars")

        # Parse JSON from response content
        return self._parse_response_as_model(content, response_type)
    
    def _get_json_schema_instruction(self, response_type: Type[T]) -> str:
        """
        Generate JSON schema instruction for LLM fallback mode
        
        Args:
            response_type: Pydantic model class
        
        Returns:
            Formatted instruction string with JSON schema
        """
        try:
            # Get JSON schema from Pydantic model
            schema = response_type.model_json_schema()
            schema_str = json.dumps(schema, indent=2, ensure_ascii=False)
            
            return f"""## IMPORTANT: JSON Output Format Required
You MUST respond with ONLY a valid JSON object (no markdown, no extra text).
The JSON must strictly follow this schema:

```json
{schema_str}
```

Output ONLY the JSON object, nothing else."""
        except Exception as e:
            logger.warning(f"Failed to generate JSON schema: {e}")
            return """## IMPORTANT: JSON Output Format Required
You MUST respond with ONLY a valid JSON object (no markdown, no extra text)."""
    
    def _parse_response_as_model(self, content: str, response_type: Type[T]) -> T:
        """
        Parse LLM response content as Pydantic model
        
        Args:
            content: Raw LLM response text
            response_type: Target Pydantic model class
        
        Returns:
            Parsed model instance
        """
        # Try direct JSON parsing first
        try:
            data = json.loads(content)
            return response_type.model_validate(data)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code block
        json_pattern = r'```(?:json)?\s*([\s\S]+?)\s*```'
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return response_type.model_validate(data)
            except json.JSONDecodeError:
                pass
        
        # Try to find any JSON object in the text
        brace_start = content.find('{')
        brace_end = content.rfind('}')
        if brace_start != -1 and brace_end > brace_start:
            try:
                json_str = content[brace_start:brace_end + 1]
                data = json.loads(json_str)
                return response_type.model_validate(data)
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Failed to parse LLM response as {response_type.__name__}: {content[:200]}...")
    
    @property
    def active(self) -> str:
        """
        Get active model name
        
        Returns:
            Active model name
        
        Example:
            print(f"Using model: {pixelle_video.llm.active}")
        """
        return self._get_config_value("model", "gpt-3.5-turbo")
    
    def __repr__(self) -> str:
        """String representation"""
        model = self.active
        base_url = self._get_config_value("base_url", "default")
        return f"<LLMService model={model!r} base_url={base_url!r}>"

