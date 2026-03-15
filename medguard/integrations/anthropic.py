"""
Anthropic API LLM caller.

Uses the anthropic SDK if installed (pip install medguard[anthropic]).
Falls back to raw httpx if SDK is not available.
Supports both full response and streaming.
"""
from __future__ import annotations

import os
from typing import AsyncIterator

import structlog

from medguard.config import LLMConfig

log = structlog.get_logger(__name__)


class AnthropicCaller:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._api_key = os.environ.get(config.api_key_env, "")

    async def call(self, prompt: str) -> str:
        """Send prompt, return full response text."""
        try:
            return await self._call_with_sdk(prompt)
        except ImportError:
            return await self._call_with_httpx(prompt)

    async def call_stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream response tokens."""
        try:
            async for token in self._stream_with_sdk(prompt):
                yield token
        except ImportError:
            async for token in self._stream_with_httpx(prompt):
                yield token

    async def _call_with_sdk(self, prompt: str) -> str:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        message = await client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=self.config.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    async def _stream_with_sdk(self, prompt: str) -> AsyncIterator[str]:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        async with client.messages.stream(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=self.config.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def _call_with_httpx(self, prompt: str) -> str:
        import json
        import httpx

        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": self.config.system_prompt,
            "messages": [{"role": "user", "content": prompt}],
        }
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
            return data["content"][0]["text"]

    async def _stream_with_httpx(self, prompt: str) -> AsyncIterator[str]:
        import json
        import httpx

        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": self.config.system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if data.get("type") == "content_block_delta":
                                delta = data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    yield delta.get("text", "")
                        except json.JSONDecodeError:
                            pass
