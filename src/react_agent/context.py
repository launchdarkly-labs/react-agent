"""Configuration passed via LangGraph Runtime.

All model/prompt/tool-parameter configuration is sourced from LaunchDarkly at
request time (see ``react_agent.ld_client``); this dataclass now only exists to
satisfy LangGraph's ``context_schema`` requirement.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True)
class Context:
    """Empty runtime context. Reserved for future request-scoped identity."""
