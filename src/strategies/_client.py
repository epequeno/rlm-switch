"""
Shared OpenAI-compatible client factory.

Prefers OpenRouter if OPENROUTER_API_KEY is set, falls back to OPENAI_API_KEY.

OpenRouter model names use provider-prefixed format: "openai/gpt-4o-mini"
OpenAI-direct model names are unprefixed: "gpt-4o-mini"
"""

from __future__ import annotations
import os
from openai import OpenAI


def make_client() -> OpenAI:
    """Return an OpenAI-compatible client, preferring OpenRouter."""
    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key:
        return OpenAI(
            api_key=or_key,
            base_url="https://openrouter.ai/api/v1",
        )

    oa_key = os.getenv("OPENAI_API_KEY")
    if oa_key:
        return OpenAI(api_key=oa_key)

    raise RuntimeError(
        "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY in your environment."
    )
