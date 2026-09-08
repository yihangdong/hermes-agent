"""Stage-A proposal-only entrypoint: the tool surface stays empty under hostile
ambient configuration, and every failure fails closed.

Nothing here touches a real provider, a real tool, the network, the shell, a
browser, or any credential store. The agent is a recording double, and the one
place the *accepted* toolset resolver is exercised for real is given a fake
registry that echoes the requested tool names back as schemas, so the assertion
is about which tools got selected, never about running one.

Two of these tests are positive controls rather than assertions about the new
module. ``test_absent_selection_really_does_mean_every_toolset`` and
``test_ambient_kanban_marker_really_does_widen_an_empty_selection`` establish
that the failure modes this module defends against are real in the accepted
source. Without them the rest of the file could pass against a Hermes where
``enabled_toolsets=[]`` and ``enabled_toolsets=None`` happened to behave the
same, and the module would look proven while defending against nothing.
"""

from __future__ import annotations

import io
from types import MappingProxyType
from typing import Any

import pytest

from agent import stagea_text_only as sto
from agent.stagea_text_only import (
    STAGE_A_SYSTEM_PROMPT,
    STAGE_A_TEXT_ONLY_POLICY,
    StageATextOnlyRefusal,
    StageAUpstream,
    assert_text_only_surface,
    run_stage_a_proposal,
    stage_a_agent_kwargs,
)

UPSTREAM = StageAUpstream(
    base_url="http://127.0.0.1:1/synthetic-not-contacted",
    model="synthetic-model",
    provider="synthetic-provider",
    max_tokens=256,
)

ACTION_CAPABLE_TOOL = {
    "type": "function",
    "function": {"name": "terminal", "description": "run a shell command"},
}


class RecordingAgent:
    """A compliant agent double. ``overrides`` softens exactly one field."""

    def __init__(self, **overrides):
        self.tools = []
        self.valid_tool_names = set()
        self.enabled_toolsets = []
        self.disabled_toolsets = []
        self.max_iterations = 1
        self.max_tokens = UPSTREAM.max_tokens
        self.skip_context_files = True
        self.load_soul_identity = False
        self._memory_store = None
        self._memory_manager = None
        self._tool_snapshot_generation = 0
        self.closed = False
        self.conversations = []
        self.response = "PROPOSAL: do the thing, then have someone check it."
        for name, value in overrides.items():
            setattr(self, name, value)

    def run_conversation(self, prompt, system_message=None, **kwargs):
        self.conversations.append((prompt, system_message, kwargs))
        return {"final_response": self.response}

    def close(self):
        self.closed = True


class Factory:
    """Records construction kwargs and hands back a prepared agent."""

    def __init__(self, agent=None):
        self.agent = agent if agent is not None else RecordingAgent()
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.agent


@pytest.fixture()
def armed(monkeypatch):
    """Arm the opt-in and clear every marker the entrypoint refuses."""
    monkeypatch.setenv(sto.OPT_IN_ENV, sto.OPT_IN_VALUE)
    for key in sto.APPROVAL_BYPASS_ENV_KEYS + sto.AMBIENT_TOOL_SCOPE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Positive controls: the failure modes being defended against are real
# ---------------------------------------------------------------------------


def _fake_registry_definitions(names, quiet=False):
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in sorted(names)
    ]


def _selected_tool_names(monkeypatch, enabled_toolsets):
    """Run the accepted resolver with a fake registry; return selected names.

    ``quiet_mode=False`` deliberately bypasses the memoisation in
    ``get_tool_definitions`` so each call re-resolves against the fake.
    """
    import model_tools

    monkeypatch.setattr(
        model_tools.registry, "get_definitions", _fake_registry_definitions
    )
    defs = model_tools._compute_tool_definitions(
        enabled_toolsets=enabled_toolsets, quiet_mode=False
    )
    return {d["function"]["name"] for d in defs}


def test_empty_selection_and_absent_selection_are_not_the_same_thing(monkeypatch):
    """``[]`` selects nothing; ``None`` selects everything. Load-bearing.

    The whole policy rests on spelling the empty selection as an empty list.
    If the accepted resolver ever collapsed the two, every other assertion in
    this file would still pass while the surface silently reopened.
    """
    empty = _selected_tool_names(monkeypatch, [])
    absent = _selected_tool_names(monkeypatch, None)

    assert empty == set()
    assert len(absent) > 50, "expected the absent selection to mean every toolset"
    assert {"terminal", "write_file"} <= absent


def test_ambient_kanban_marker_widens_an_empty_selection(monkeypatch):
    """An ambient env marker adds tools the caller never asked for.

    This is exactly why the module verifies the resolved surface instead of
    trusting the kwargs it passed: the widening happens inside the resolver,
    downstream of ``enabled_toolsets=[]``.
    """
    monkeypatch.setenv("HERMES_KANBAN_TASK", "synthetic-task-id")
    widened = _selected_tool_names(monkeypatch, [])

    assert widened, "ambient marker should have re-populated an empty selection"
    assert any(name.startswith("kanban") for name in widened), sorted(widened)


# ---------------------------------------------------------------------------
# Default off
# ---------------------------------------------------------------------------


def test_refuses_without_the_opt_in(monkeypatch):
    monkeypatch.delenv(sto.OPT_IN_ENV, raising=False)
    factory = Factory()

    with pytest.raises(StageATextOnlyRefusal) as excinfo:
        run_stage_a_proposal("propose something", UPSTREAM, agent_factory=factory)

    assert excinfo.value.code == "OPT_IN_ABSENT"
    assert factory.calls == [], "nothing may be constructed before the opt-in check"


@pytest.mark.parametrize("value", ["", "0", "2", "true", "yes", "on", " 1", "1 "])
def test_refuses_near_miss_opt_in_values(monkeypatch, value):
    """The opt-in is matched literally, so no truthiness parser can arm it."""
    monkeypatch.setenv(sto.OPT_IN_ENV, value)
    factory = Factory()

    with pytest.raises(StageATextOnlyRefusal) as excinfo:
        run_stage_a_proposal("propose something", UPSTREAM, agent_factory=factory)

    assert excinfo.value.code == "OPT_IN_ABSENT"
    assert factory.calls == []


# ---------------------------------------------------------------------------
# Ambient state is refused, not merged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", sto.APPROVAL_BYPASS_ENV_KEYS)
@pytest.mark.parametrize("value", ["1", "0", "anything"])
def test_refuses_ambient_approval_bypass(armed, monkeypatch, key, value):
    """Presence is refused, not interpreted — including ``=0``.

    ``tools/approval.py`` freezes ``HERMES_YOLO_MODE`` at import time, so a
    control that merely cleared it at call time would depend on import order.
    """
    monkeypatch.setenv(key, value)
    factory = Factory()

    with pytest.raises(StageATextOnlyRefusal) as excinfo:
        run_stage_a_proposal("propose something", UPSTREAM, agent_factory=factory)

    assert excinfo.value.code == "AMBIENT_APPROVAL_BYPASS"
    assert key in excinfo.value.detail
    assert factory.calls == [], "no agent may be built with a bypass marker present"


@pytest.mark.parametrize("key", sto.AMBIENT_TOOL_SCOPE_ENV_KEYS)
def test_refuses_ambient_dispatcher_worker_markers(armed, monkeypatch, key):
    monkeypatch.setenv(key, "synthetic")
    factory = Factory()

    with pytest.raises(StageATextOnlyRefusal) as excinfo:
        run_stage_a_proposal("propose something", UPSTREAM, agent_factory=factory)

    assert excinfo.value.code == "AMBIENT_TOOL_SCOPE"
    assert key in excinfo.value.detail
    assert factory.calls == []


def test_kanban_marker_family_is_bound_to_the_accepted_constant():
    """Drift guard: the refused family is the accepted family, not a copy."""
    from agent.delegation_context import KANBAN_ENV_KEYS

    assert sto.AMBIENT_TOOL_SCOPE_ENV_KEYS == tuple(KANBAN_ENV_KEYS)
    assert "HERMES_KANBAN_TASK" in sto.AMBIENT_TOOL_SCOPE_ENV_KEYS


def test_refused_markers_are_scrubbed_for_the_call_and_restored_after():
    env = {
        "HERMES_YOLO_MODE": "1",
        "HERMES_KANBAN_TASK": "t-1",
        "UNRELATED": "keep-me",
    }
    seen = {}

    with sto._scrubbed_environment(env):
        seen = dict(env)

    assert "HERMES_YOLO_MODE" not in seen
    assert "HERMES_KANBAN_TASK" not in seen
    assert seen["UNRELATED"] == "keep-me"
    assert env == {
        "HERMES_YOLO_MODE": "1",
        "HERMES_KANBAN_TASK": "t-1",
        "UNRELATED": "keep-me",
    }


def test_construction_cannot_observe_the_refused_markers(armed):
    """End to end: whatever builds the agent sees a scrubbed environment."""
    import os

    observed = {}

    def observing_factory(**kwargs):
        observed.update(
            {
                key: os.environ.get(key)
                for key in sto.APPROVAL_BYPASS_ENV_KEYS
                + sto.AMBIENT_TOOL_SCOPE_ENV_KEYS
            }
        )
        return RecordingAgent()

    run_stage_a_proposal("propose something", UPSTREAM, agent_factory=observing_factory)

    assert observed, "the factory should have been called"
    assert set(observed.values()) == {None}


# ---------------------------------------------------------------------------
# The frozen policy
# ---------------------------------------------------------------------------


def test_policy_spells_the_empty_selection_as_an_empty_list():
    kwargs = stage_a_agent_kwargs(UPSTREAM)

    assert kwargs["enabled_toolsets"] is not None
    assert kwargs["enabled_toolsets"] == []
    assert isinstance(kwargs["enabled_toolsets"], list)


def test_policy_closes_context_memory_and_bounds():
    kwargs = stage_a_agent_kwargs(UPSTREAM)

    assert kwargs["skip_context_files"] is True
    assert kwargs["load_soul_identity"] is False
    assert kwargs["skip_memory"] is True
    assert kwargs["skip_background_review"] is True
    assert kwargs["max_iterations"] == 1
    assert kwargs["max_tokens"] == UPSTREAM.max_tokens


def test_policy_is_frozen_and_not_shared_between_runs():
    # A read-only proxy has no ``__setitem__`` at all, so an in-place softening
    # of the policy is a TypeError rather than a silent success.
    assert isinstance(STAGE_A_TEXT_ONLY_POLICY, MappingProxyType)
    assert not hasattr(STAGE_A_TEXT_ONLY_POLICY, "__setitem__")

    first = stage_a_agent_kwargs(UPSTREAM)
    first["enabled_toolsets"].append("terminal")

    assert stage_a_agent_kwargs(UPSTREAM)["enabled_toolsets"] == []


def test_no_credential_is_sourced_or_carried():
    kwargs = stage_a_agent_kwargs(UPSTREAM)

    assert kwargs["api_key"] is None
    assert not hasattr(UPSTREAM, "api_key")
    # Every string that reaches the constructor came from the caller's typed
    # upstream. Nothing was read from config.yaml, the environment, or a
    # keychain, and no string constant is injected by this module.
    supplied = {UPSTREAM.base_url, UPSTREAM.model, UPSTREAM.provider}
    assert {v for v in kwargs.values() if isinstance(v, str)} <= supplied


def test_every_policy_key_is_a_real_accepted_agent_parameter():
    """The policy must bind to the accepted constructor, not to hope.

    A misspelled or removed knob would otherwise be a ``TypeError`` at the
    first real Stage-A run, long after this suite went green against a double.
    """
    import inspect

    from run_agent import AIAgent

    parameters = inspect.signature(AIAgent.__init__).parameters

    for name in stage_a_agent_kwargs(UPSTREAM):
        assert name in parameters, f"AIAgent.__init__ has no {name!r} parameter"


def test_default_factory_targets_the_accepted_agent():
    """The un-injected path constructs the accepted agent and nothing else."""
    import inspect

    source = inspect.getsource(sto._default_agent_factory)

    assert "from run_agent import AIAgent" in source
    assert "AIAgent(**kwargs)" in source


def test_source_supplies_no_endpoint_or_model_default():
    """F1: source binds the shape, runtime supplies the value."""
    for field in ("base_url", "model", "provider", "max_tokens"):
        assert field not in STAGE_A_TEXT_ONLY_POLICY


@pytest.mark.parametrize(
    "override",
    [
        {"base_url": ""},
        {"base_url": "   "},
        {"model": ""},
        {"provider": ""},
        {"max_tokens": 0},
        {"max_tokens": -1},
        {"max_tokens": True},
        {"max_tokens": "256"},
    ],
)
def test_incomplete_upstream_fails_closed(armed, override):
    fields: dict[str, Any] = {
        "base_url": UPSTREAM.base_url,
        "model": UPSTREAM.model,
        "provider": UPSTREAM.provider,
        "max_tokens": UPSTREAM.max_tokens,
    }
    fields.update(override)
    upstream = StageAUpstream(**fields)
    factory = Factory()

    with pytest.raises(StageATextOnlyRefusal) as excinfo:
        run_stage_a_proposal("propose something", upstream, agent_factory=factory)

    assert excinfo.value.code == "UPSTREAM_INCOMPLETE"
    assert factory.calls == []


@pytest.mark.parametrize("prompt", ["", "   ", "\n", None])
def test_empty_prompt_fails_closed(armed, prompt):
    factory = Factory()

    with pytest.raises(StageATextOnlyRefusal) as excinfo:
        run_stage_a_proposal(prompt, UPSTREAM, agent_factory=factory)

    assert excinfo.value.code == "PROMPT_EMPTY"
    assert factory.calls == []


# ---------------------------------------------------------------------------
# The resolved surface is verified, not assumed
# ---------------------------------------------------------------------------


def test_verifier_accepts_the_compliant_shape():
    """Positive control for the mutation table below."""
    assert_text_only_surface(RecordingAgent(), UPSTREAM) is None


@pytest.mark.parametrize(
    ("override", "code"),
    [
        ({"tools": [ACTION_CAPABLE_TOOL]}, "TOOL_SURFACE_PRESENT"),
        ({"valid_tool_names": {"terminal"}}, "TOOL_SURFACE_PRESENT"),
        ({"enabled_toolsets": None}, "POLICY_NOT_APPLIED"),
        ({"enabled_toolsets": ["hermes-cli"]}, "POLICY_NOT_APPLIED"),
        ({"max_iterations": 25}, "POLICY_NOT_APPLIED"),
        ({"max_tokens": 999999}, "POLICY_NOT_APPLIED"),
        ({"skip_context_files": False}, "CONTEXT_INHERITANCE"),
        ({"load_soul_identity": True}, "CONTEXT_INHERITANCE"),
        ({"_memory_store": object()}, "MEMORY_INHERITANCE"),
        ({"_memory_manager": object()}, "MEMORY_INHERITANCE"),
    ],
)
def test_verifier_refuses_every_softened_field(override, code):
    with pytest.raises(StageATextOnlyRefusal) as excinfo:
        assert_text_only_surface(RecordingAgent(**override), UPSTREAM)

    assert excinfo.value.code == code


@pytest.mark.parametrize(
    ("override", "code"),
    [
        ({"tools": [ACTION_CAPABLE_TOOL]}, "TOOL_SURFACE_PRESENT"),
        ({"valid_tool_names": {"browser_navigate"}}, "TOOL_SURFACE_PRESENT"),
        ({"skip_context_files": False}, "CONTEXT_INHERITANCE"),
        ({"_memory_store": object()}, "MEMORY_INHERITANCE"),
        ({"max_iterations": 25}, "POLICY_NOT_APPLIED"),
    ],
)
def test_a_noncompliant_agent_never_reaches_the_model(armed, override, code):
    """A factory that ignores the policy still cannot get a model call."""
    agent = RecordingAgent(**override)
    factory = Factory(agent)

    with pytest.raises(StageATextOnlyRefusal) as excinfo:
        run_stage_a_proposal("propose something", UPSTREAM, agent_factory=factory)

    assert excinfo.value.code == code
    assert factory.calls, "the refusal must come from the resolved agent, not the args"
    assert agent.conversations == [], "no model call may happen after a refusal"
    assert agent.closed is True, "the refused agent must still be released"


def test_the_accepted_late_binding_rebuild_keeps_the_surface_empty():
    """The surface is verified once at construction — the rebuild must agree.

    ``tools.mcp_tool.refresh_agent_mcp_tools`` re-derives an already-built
    agent's tool snapshot from the live registry: MCP servers that connect
    after the build, ``/reload-mcp``, the late-binding thread, the
    between-turns prologue. It reuses the agent's own ``enabled_toolsets``, so
    the empty selection has to survive it. If it did not, an agent verified at
    construction could regain tools before its single model call.
    """
    from tools.mcp_tool import refresh_agent_mcp_tools

    agent = RecordingAgent()

    added = refresh_agent_mcp_tools(agent, quiet_mode=True)

    assert added == set()
    assert agent.tools == []
    assert agent.valid_tool_names == set()
    assert_text_only_surface(agent, UPSTREAM) is None


def test_the_late_binding_rebuild_would_have_repopulated_an_absent_selection():
    """Positive control for the test above: the rebuild really does rebuild."""
    from tools.mcp_tool import refresh_agent_mcp_tools

    agent = RecordingAgent(enabled_toolsets=None)

    refresh_agent_mcp_tools(agent, quiet_mode=True)

    assert agent.valid_tool_names, "an absent selection should rebuild to a full surface"
    with pytest.raises(StageATextOnlyRefusal) as excinfo:
        assert_text_only_surface(agent, UPSTREAM)
    assert excinfo.value.code == "TOOL_SURFACE_PRESENT"


def test_action_capable_tool_names_are_reported_on_refusal(armed):
    agent = RecordingAgent(tools=[ACTION_CAPABLE_TOOL])

    with pytest.raises(StageATextOnlyRefusal) as excinfo:
        run_stage_a_proposal("propose", UPSTREAM, agent_factory=Factory(agent))

    assert "terminal" in excinfo.value.detail


# ---------------------------------------------------------------------------
# The accepted path
# ---------------------------------------------------------------------------


def test_successful_run_is_text_only_and_carries_the_frozen_policy(armed):
    factory = Factory()

    text = run_stage_a_proposal("propose something", UPSTREAM, agent_factory=factory)

    assert text == factory.agent.response
    (kwargs,) = factory.calls
    assert kwargs["enabled_toolsets"] == []
    assert kwargs["skip_context_files"] is True
    assert kwargs["load_soul_identity"] is False
    assert kwargs["skip_memory"] is True
    assert kwargs["api_key"] is None
    assert kwargs["base_url"] == UPSTREAM.base_url
    assert kwargs["model"] == UPSTREAM.model

    (prompt, system_message, _), = factory.agent.conversations
    assert prompt == "propose something"
    assert system_message == STAGE_A_SYSTEM_PROMPT
    assert factory.agent.closed is True


def test_system_prompt_is_constant_and_states_the_no_action_contract():
    assert "no tools" in STAGE_A_SYSTEM_PROMPT
    assert "proposal" in STAGE_A_SYSTEM_PROMPT.lower()


def test_caller_cannot_inject_its_own_system_message(armed):
    """There is no seam for a second instruction source."""
    import inspect

    signature = inspect.signature(run_stage_a_proposal)

    assert "system_message" not in signature.parameters
    assert "ephemeral_system_prompt" not in stage_a_agent_kwargs(UPSTREAM)


@pytest.mark.parametrize("response", ["", "   ", None])
def test_a_run_without_text_fails_closed(armed, response):
    agent = RecordingAgent(response=response)

    with pytest.raises(StageATextOnlyRefusal) as excinfo:
        run_stage_a_proposal("propose", UPSTREAM, agent_factory=Factory(agent))

    assert excinfo.value.code == "NO_PROPOSAL_TEXT"


# ---------------------------------------------------------------------------
# Process entrypoint
# ---------------------------------------------------------------------------


CLI_ARGS = [
    "--base-url",
    UPSTREAM.base_url,
    "--model",
    UPSTREAM.model,
    "--provider",
    UPSTREAM.provider,
    "--max-tokens",
    str(UPSTREAM.max_tokens),
]


def test_main_reads_the_prompt_from_stdin_and_writes_only_text(armed, monkeypatch):
    factory = Factory()
    monkeypatch.setattr(sto, "_default_agent_factory", factory)
    out = io.StringIO()

    code = sto.main(CLI_ARGS, stdin=io.StringIO("propose something"), stdout=out)

    assert code == 0
    assert out.getvalue() == factory.agent.response + "\n"
    (prompt, _, _), = factory.agent.conversations
    assert prompt == "propose something"


@pytest.mark.parametrize(
    "argv",
    [
        [],
        CLI_ARGS + ["--toolsets", "all"],
        CLI_ARGS + ["extra-positional"],
        CLI_ARGS[:-1],
        ["--base-url=x"] + CLI_ARGS[2:],
        CLI_ARGS + ["--model", "second"],
        ["--model", UPSTREAM.model],
        CLI_ARGS[:-1] + ["not-a-number"],
    ],
)
def test_main_refuses_anything_outside_its_closed_argv(armed, monkeypatch, argv):
    factory = Factory()
    monkeypatch.setattr(sto, "_default_agent_factory", factory)
    out = io.StringIO()

    code = sto.main(argv, stdin=io.StringIO("propose something"), stdout=out)

    assert code == 2
    assert out.getvalue() == ""
    assert factory.calls == []


def test_main_refuses_without_the_opt_in(monkeypatch):
    monkeypatch.delenv(sto.OPT_IN_ENV, raising=False)
    factory = Factory()
    monkeypatch.setattr(sto, "_default_agent_factory", factory)
    out = io.StringIO()

    code = sto.main(CLI_ARGS, stdin=io.StringIO("propose something"), stdout=out)

    assert code == 2
    assert out.getvalue() == ""
    assert factory.calls == []


def test_importing_the_module_arms_nothing(monkeypatch):
    """A default-off entrypoint must do nothing at import."""
    import importlib
    import os

    monkeypatch.delenv(sto.OPT_IN_ENV, raising=False)
    importlib.reload(sto)

    assert sto.OPT_IN_ENV not in os.environ
    assert sto.STAGE_A_TEXT_ONLY_POLICY["enabled_toolsets"] == ()


def test_refusal_codes_are_a_closed_vocabulary():
    with pytest.raises(ValueError):
        StageATextOnlyRefusal("SOMETHING_NEW", "")
