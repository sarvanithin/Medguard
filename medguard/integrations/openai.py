"""
OpenAI-compatible LLM caller.

Works with:
  - OpenAI API
  - Azure OpenAI (set base_url + api_key_env)
  - Ollama (base_url=http://localhost:11434/v1, any api_key)
  - Any OpenAI-compatible endpoint
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator

import structlog

from medguard.config import LLMConfig

log = structlog.get_logger(__name__)


class OpenAICaller:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._api_key = os.environ.get(config.api_key_env, "")
        self._base_url = config.base_url or "https://api.openai.com/v1"

    async def call(self, prompt: str) -> str:
        try:
            return await self._call_with_sdk(prompt)
        except ImportError:
            return await self._call_with_httpx(prompt)

    async def call_stream(self, prompt: str) -> AsyncIterator[str]:
        try:
            async for token in self._stream_with_sdk(prompt):
                yield token
        except ImportError:
            async for token in self._stream_with_httpx(prompt):
                yield token

    async def _call_with_sdk(self, prompt: str) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout_seconds,
        )
        return response.choices[0].message.content or ""

    async def _stream_with_sdk(self, prompt: str) -> AsyncIterator[str]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = await client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    async def _call_with_httpx(self, prompt: str) -> str:
        import httpx

        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            r = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "max_tokens": self.config.max_tokens,
                },
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]

    async def _stream_with_httpx(self, prompt: str) -> AsyncIterator[str]:
        import json

        import httpx

        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "max_tokens": self.config.max_tokens,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta and delta["content"]:
                                yield delta["content"]
                        except (json.JSONDecodeError, KeyError):
                            pass
