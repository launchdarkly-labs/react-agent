"""Default prompts used by the agent.

The ``{{ system_time }}`` placeholder is Mustache, not a Python format string.
The LaunchDarkly AI SDK interpolates Mustache variables on both the
LD-served path (variation instructions in the UI are stored as Mustache) and
the fallback path (``FALLBACK.instructions`` here flows through the same
interpolator when the SDK returns the fallback). Leaving a Python-style
``{system_time}`` here would ship the literal brace string to the model when
LaunchDarkly is unreachable.
"""

SYSTEM_PROMPT = """You are a helpful AI assistant.

System time: {{ system_time }}"""
