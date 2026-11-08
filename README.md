# haystack-atr

[Agent Threat Rules](https://github.com/Agent-Threat-Rule/agent-threat-rules) (ATR) for [Haystack](https://haystack.deepset.ai/) agents and pipelines. It checks tool calls before they run, tool results before the model reads them, and chat messages in a pipeline, using a local rule engine ([pyatr](https://pypi.org/project/pyatr/)): no network, no API key, and the same input gives the same result on every run.

It plugs into the extension points Haystack 3 already has: the `before_tool` and `after_tool` Agent hooks, and the confirmation strategies of `ConfirmationHook`.

## Install

Not on PyPI yet. Install from this repository (Haystack 3.4 or later):

```sh
pip install "haystack-atr @ git+https://github.com/Agent-Threat-Rule/haystack-atr"
```

## Use with an Agent

```python
from haystack.components.agents import Agent
from haystack.hooks.human_in_the_loop import ConfirmationHook
from haystack_atr import ATRToolCallStrategy, ATRToolResultHook

agent = Agent(
    chat_generator=...,
    tools=[...],
    hooks={
        "before_tool": [ConfirmationHook({"*": ATRToolCallStrategy()})],
        "after_tool": [ATRToolResultHook()],
    },
)
```

- `ATRToolCallStrategy` checks each pending tool call (tool name and arguments). A rejected call does not run; the model receives a tool result that names the rules that matched. It is a confirmation strategy, so it can share a `ConfirmationHook` with human approval for specific tools (`{"delete_file": ..., "*": ATRToolCallStrategy()}`). A tool with its own entry uses that strategy instead of the `"*"` ATR entry, so its calls are not checked by ATR.
- `ATRToolResultHook` checks the results of the tools that just ran. A result that is blocked is replaced with an error result naming the rules, so the model never reads the original text.

## Use in a pipeline

`ATRMessageScanner` scans user messages and tool results and returns `messages`, `findings` and `blocked`. Route on `blocked` with a `ConditionalRouter` to stop before a generator. System and assistant messages are not scanned.

## Lanes and blocking

ATR rules carry a maturity, and a lane selects which ones run: `enforce` (stable rules), `alert` (stable and test) and `hunt` (all).

- `lane` (default `alert`) chooses the rules that are evaluated. Every match is logged and passed to `on_finding`.
- `block_lane` (default `enforce`) chooses the rules that can block: a high or critical match from a rule in that lane. `None` records only. `block_lane` cannot be wider than `lane`.

The default blocks only on stable rules, and that lane is narrow. In our tests it blocked a tool result telling the model to send `~/.ssh/id_rsa` to a webhook, but none of three tool calls: a reverse shell, a `curl | sh` call, and a settings write that disables the agent's permission prompts. `block_lane="alert"` blocked all four. The alert lane also blocks ordinary content: with pyatr 0.5.0 it blocked 112 of 1,500 benign texts (7.5%) sampled from ATR's benign corpus and checked as tool results. Most of the rules that fired look for credential file paths such as `~/.ssh/id_rsa` or `.aws/credentials`, shell metacharacters or `eval()`. Start with `block_lane=None` and an `on_finding` callback, look at what your own traffic produces, then choose.

## License

MIT
