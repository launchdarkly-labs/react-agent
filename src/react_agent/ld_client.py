"""LaunchDarkly AI Config wiring for the ReAct agent."""

from __future__ import annotations

import os

import ldclient
from dotenv import load_dotenv
from ldai.client import LDAIClient
from ldai.models import (
    AIAgentConfig,
    AIAgentConfigDefault,
    ModelConfig,
    ProviderConfig,
)
from ldclient import Context as LDContext
from ldclient.config import Config as LDClientConfig

from react_agent import prompts

load_dotenv()

AI_CONFIG_KEY = "react-agent"

# Fallback used when LD is unreachable / LD_SDK_KEY is unset.
FALLBACK = AIAgentConfigDefault(
    enabled=True,
    model=ModelConfig(name="claude-sonnet-4-5-20250929", custom={"max_search_results": 10}),
    provider=ProviderConfig(name="anthropic"),
    instructions=prompts.SYSTEM_PROMPT,
)


def _init_client() -> LDAIClient:
    sdk_key = os.environ.get("LD_SDK_KEY")
    if sdk_key:
        ldclient.set_config(LDClientConfig(sdk_key))
    else:
        ldclient.set_config(LDClientConfig(sdk_key="offline", offline=True))
    return LDAIClient(ldclient.get())


ai_client = _init_client()
_context = LDContext.builder("anonymous-react-agent").kind("user").anonymous(True).build()


def get_agent_config(system_time: str) -> AIAgentConfig:
    """Resolve the ReAct agent's AI Config for the current request."""
    return ai_client.agent_config(
        AI_CONFIG_KEY, _context, FALLBACK, {"system_time": system_time}
    )
