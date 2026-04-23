# Offline evaluations

`react-agent-seed.csv` is a starter dataset for running the `react-agent` AI
Config through LaunchDarkly's offline eval Playground. It covers three
shapes: direct-answer prompts, search-required prompts, and one prompt that
verifies `{{ system_time }}` was interpolated into the instructions.

## Recommended flow

Follow the [Offline Evals tutorial](https://docs.launchdarkly.com/guides/ai-configs/offline-evaluations):

1. Open **AI Configs > `react-agent`** in the `react-agent-demo` project and
   open the **Playground**.
2. Upload `react-agent-seed.csv` as a dataset.
3. Run the `default` variation (Anthropic `claude-sonnet-4-5`) against the
   dataset as the baseline.
4. Clone `default` into a second variation that points at a **different
   model family** (for example, `openai/gpt-4o` or `openai/gpt-4o-mini`) and
   run it against the same dataset.
5. Attach the built-in **Accuracy** judge with a pass threshold of **0.85**
   and compare the two runs side-by-side. Promote the winner to fallthrough
   via `/aiconfig-targeting` only if it beats the baseline on Accuracy and
   does not regress on Relevance or Toxicity.

Agent mode does not support UI-attached auto-judges on live traffic today;
offline eval is the supported pre-ship regression gate. Add programmatic
`ai_client.create_judge(...)` calls in application code only if continuous
live scoring is needed after the rollout is stable.
