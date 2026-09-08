"""Stage-A proposal-only (text-only) execution entrypoint.

Stage A (AI-Org #197/#198) asks Hermes for a *proposal*: text, and nothing
else.  The accepted bare oneshot path cannot supply that guarantee.  Its own
module docstring states the opposite contract --- ``hermes_cli/oneshot.py``
says toolsets come from "whatever the user has configured for ``cli``",
"Rules / memory / AGENTS.md / preloaded skills = same as a normal chat turn",
and "Approvals = auto-bypassed (``HERMES_YOLO_MODE=1`` is set for the call)".
Every one of those is ambient host state, so a Stage-A request routed through
it inherits whatever the host happens to be configured with.

This module is the smallest execution entrypoint that closes that gap without
touching the accepted bridge, the CLI, or any other accepted module.  It does
not change ``hermes -z`` and does not change ordinary Hermes behaviour in any
way: it is a *separate* entrypoint that a Stage-A caller invokes instead.

Three independent controls, all fail-closed:

1. **Default off.**  Nothing here runs unless ``HERMES_STAGEA_TEXT_ONLY_V1``
   is exactly ``"1"``.  Importing the module has no effect.

2. **Ambient state is refused, not merged.**  Approval-bypass markers and
   dispatcher-worker identity markers present in the caller's environment
   cause a refusal *before* an agent is built.  Refusing rather than clearing
   is deliberate: ``tools/approval.py`` freezes ``HERMES_YOLO_MODE`` into
   ``_YOLO_MODE_FROZEN`` at *import* time, so clearing it at call time cannot
   be relied on to undo a bypass in a process that already imported that
   module.  A control that can be defeated by import order is not a control.
   The markers are additionally scrubbed for the duration of the call so that
   anything constructed inside the call cannot observe them either.

3. **The resulting surface is verified, not assumed.**  Passing the right
   constructor arguments is a request; it is not proof.  After the agent is
   built and before a single model call is made, :func:`assert_text_only_surface`
   re-reads the agent's *actual* resolved state and refuses unless the tool
   surface is empty and the context/memory/bound policy actually took effect.
   This is what catches re-enablement paths that do not go through our
   arguments at all --- for example ``model_tools._compute_tool_definitions``
   appends the ``kanban`` toolset when ``HERMES_KANBAN_TASK`` marks a
   dispatcher-owned worker, *even when the caller passed an empty
   ``enabled_toolsets``*.

What the policy binds (see :data:`STAGE_A_TEXT_ONLY_POLICY`):

* ``enabled_toolsets=[]`` --- an empty *selection*, not an absent one.
  ``model_tools._compute_tool_definitions`` branches on
  ``enabled_toolsets is not None``; ``None`` means "start with everything".
  ``[]`` means "start with nothing", which is the only spelling that yields a
  zero-tool surface.
* ``skip_context_files=True`` + ``load_soul_identity=False`` --- together
  these are the exact pair ``agent/system_prompt.py`` gates SOUL.md on
  (``if agent.load_soul_identity or not agent.skip_context_files``) and the
  gate AGENTS.md / project rules are read behind (``if not
  agent.skip_context_files``).  Both must hold or context files come back.
* ``skip_memory=True`` --- no built-in memory store and no external memory
  provider, so MEMORY.md and the user profile stay out of the prompt.
* ``skip_background_review=True`` --- no auxiliary review turn.
* ``max_iterations=1`` --- ``agent/conversation_loop.py`` loops
  ``while api_call_count < agent.max_iterations``, so this is exactly one
  model-side exchange with no agentic continuation.
* ``api_key=None`` --- the Stage-A child is credentialless (#198 Phase B item
  4).  This module never reads a credential from the environment, a config
  file, or a keychain, and never accepts one on argv.

What this module deliberately does **not** decide: the upstream endpoint,
provider label, model version and output-token bound.  Planner ruled on #198
(``5581901721``, F1) that source binds a *typed, default-off* upstream
identity and that concrete endpoint/model selection is a later runtime and
admission fact.  So :class:`StageAUpstream` names those fields and requires
them at call time, with no defaults and no fallback to ambient config or
environment.  Missing selection fails closed.

Usage is either the importable entrypoint::

    from agent.stagea_text_only import StageAUpstream, run_stage_a_proposal
    text = run_stage_a_proposal("...", StageAUpstream(...))

or, for a subprocess-shaped caller, ``python -m agent.stagea_text_only``
with the prompt on stdin and the runtime selection on argv.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Sequence

# Single source of truth for the dispatcher-worker identity family.  Imported
# rather than re-listed so this control cannot drift away from the family the
# accepted code actually keys off.  The module's own imports are stdlib only.
from agent.delegation_context import KANBAN_ENV_KEYS

__all__ = [
    "OPT_IN_ENV",
    "OPT_IN_VALUE",
    "APPROVAL_BYPASS_ENV_KEYS",
    "AMBIENT_TOOL_SCOPE_ENV_KEYS",
    "STAGE_A_TEXT_ONLY_POLICY",
    "STAGE_A_SYSTEM_PROMPT",
    "StageAUpstream",
    "StageATextOnlyRefusal",
    "assert_text_only_surface",
    "stage_a_agent_kwargs",
    "run_stage_a_proposal",
    "main",
]

# ---------------------------------------------------------------------------
# Opt-in
# ---------------------------------------------------------------------------

#: Exact environment variable that arms this entrypoint.  Absent or any other
#: value => refusal.  There is no config-file, argv or default-on route in.
OPT_IN_ENV = "HERMES_STAGEA_TEXT_ONLY_V1"

#: Exact accepted value.  Matched literally: no truthiness parsing, so
#: ``true`` / ``yes`` / ``2`` do not arm it either.
OPT_IN_VALUE = "1"

# ---------------------------------------------------------------------------
# Ambient state that must not be inherited
# ---------------------------------------------------------------------------

#: Approval-bypass markers.  These are exactly the two that the accepted bare
#: oneshot path sets on the caller's behalf (``hermes_cli/oneshot.py``:
#: ``HERMES_YOLO_MODE=1`` bypasses dangerous-command approval,
#: ``HERMES_ACCEPT_HOOKS=1`` auto-accepts shell-hook registration).  Presence
#: is refused, not interpreted --- see the module docstring on why clearing
#: ``HERMES_YOLO_MODE`` at call time is not a reliable control.
APPROVAL_BYPASS_ENV_KEYS: tuple[str, ...] = (
    "HERMES_YOLO_MODE",
    "HERMES_ACCEPT_HOOKS",
)

#: Ambient markers that widen the *tool* surface independently of the toolset
#: arguments.  ``HERMES_KANBAN_TASK`` in a dispatcher-owned worker makes
#: ``model_tools._compute_tool_definitions`` append the ``kanban`` toolset to
#: an explicitly empty ``enabled_toolsets``.
AMBIENT_TOOL_SCOPE_ENV_KEYS: tuple[str, ...] = tuple(KANBAN_ENV_KEYS)

# ---------------------------------------------------------------------------
# Frozen execution policy
# ---------------------------------------------------------------------------

#: The constructor policy, frozen so callers cannot soften it in place.  Every
#: entry is a security-relevant choice; see the module docstring for the exact
#: accepted-source gate each one corresponds to.  Sequences are stored as
#: tuples and materialised as fresh lists per call by
#: :func:`stage_a_agent_kwargs`.
STAGE_A_TEXT_ONLY_POLICY: Mapping[str, Any] = MappingProxyType(
    {
        "enabled_toolsets": (),
        "disabled_toolsets": (),
        "skip_context_files": True,
        "load_soul_identity": False,
        "skip_memory": True,
        "skip_background_review": True,
        "max_iterations": 1,
        "quiet_mode": True,
        "api_key": None,
    }
)

#: The only *caller-supplied* system message a Stage-A run may carry.
#: Deliberately a constant with no injection seam --- a caller-chosen system
#: message would be another ambient-instruction source.  ``system_prompt.py``
#: appends it to Hermes' own base identity and guidance rather than replacing
#: them; what this policy removes from that prompt is SOUL.md, the AGENTS.md
#: chain, project rules, the skills index, the memory snapshot and the user
#: profile (see ``skip_context_files`` / ``load_soul_identity`` / ``skip_memory``).
STAGE_A_SYSTEM_PROMPT = (
    "You are answering a Stage-A request in proposal-only mode.\n"
    "\n"
    "You have no tools and no ability to act. You cannot read or write files, "
    "run commands, browse, call the network, or cause any side effect. Do not "
    "claim to have done any of those things, and do not ask for permission to "
    "do them.\n"
    "\n"
    "Reply with plain text only: your analysis and your proposed action for a "
    "separate reviewer to evaluate. The proposal is the deliverable."
)


class StageATextOnlyRefusal(RuntimeError):
    """Stage-A refused to run, or refused to keep running.

    ``code`` is drawn from a closed vocabulary so a caller can branch on the
    reason without parsing prose.  The prompt is never interpolated into the
    message, and neither is the upstream base URL, provider or model; a
    numeric bound may appear so the caller can see which one was violated.
    """

    #: Closed refusal vocabulary.
    CODES: tuple[str, ...] = (
        "OPT_IN_ABSENT",
        "AMBIENT_APPROVAL_BYPASS",
        "AMBIENT_TOOL_SCOPE",
        "UPSTREAM_INCOMPLETE",
        "PROMPT_EMPTY",
        "TOOL_SURFACE_PRESENT",
        "CONTEXT_INHERITANCE",
        "MEMORY_INHERITANCE",
        "POLICY_NOT_APPLIED",
        "NO_PROPOSAL_TEXT",
        "ARGV_REJECTED",
    )

    def __init__(self, code: str, detail: str = "") -> None:
        if code not in self.CODES:
            raise ValueError(f"unknown Stage-A refusal code: {code!r}")
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class StageAUpstream:
    """Typed, default-off upstream identity for one Stage-A run.

    Every field is required at call time and none of them has a source-level
    default.  This module never resolves them from ``config.yaml``, the
    environment, a model alias or a provider catalog: doing so would be
    exactly the ambient inheritance Stage A is trying to exclude, and #198
    ``5581901721`` (F1) reserves concrete endpoint/model selection for a later
    runtime/admission gate.

    There is no credential field.  The Stage-A child is credentialless (#198
    Phase B item 4) and this module always constructs with ``api_key=None``.
    """

    base_url: str
    model: str
    provider: str
    max_tokens: int

    def validate(self) -> None:
        """Raise ``UPSTREAM_INCOMPLETE`` unless every field is usable."""
        for field_name in ("base_url", "model", "provider"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise StageATextOnlyRefusal(
                    "UPSTREAM_INCOMPLETE",
                    f"{field_name} must be a non-empty string",
                )
        # bool is an int subclass; a True here would silently become 1 token.
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise StageATextOnlyRefusal(
                "UPSTREAM_INCOMPLETE", "max_tokens must be an int"
            )
        if self.max_tokens <= 0:
            raise StageATextOnlyRefusal(
                "UPSTREAM_INCOMPLETE", "max_tokens must be positive"
            )


def stage_a_agent_kwargs(upstream: StageAUpstream) -> dict[str, Any]:
    """Materialise the exact constructor arguments for one Stage-A run.

    Sequence policy values become fresh lists so no two runs share a mutable
    object, and so ``enabled_toolsets`` reaches the agent as the empty *list*
    the accepted resolver expects.
    """
    upstream.validate()
    kwargs: dict[str, Any] = {}
    for key, value in STAGE_A_TEXT_ONLY_POLICY.items():
        kwargs[key] = list(value) if isinstance(value, tuple) else value
    kwargs["base_url"] = upstream.base_url
    kwargs["model"] = upstream.model
    kwargs["provider"] = upstream.provider
    kwargs["max_tokens"] = upstream.max_tokens
    return kwargs


def _tool_names(agent: Any) -> list[str]:
    """Best-effort names from a resolved tool list, for the refusal detail."""
    names: list[str] = []
    for entry in getattr(agent, "tools", None) or []:
        try:
            names.append(str(entry["function"]["name"]))
        except (KeyError, TypeError):
            names.append(repr(entry)[:40])
    return names


def assert_text_only_surface(agent: Any, upstream: StageAUpstream) -> None:
    """Refuse unless the *resolved* agent really is proposal-only.

    Read the agent's own post-construction state rather than trusting the
    arguments we passed.  Anything that re-enabled a tool, re-armed a context
    source or softened a bound --- through config, environment, plugin
    discovery, MCP registration or a code path that never saw our kwargs ---
    is caught here, before any model call.
    """
    tools = getattr(agent, "tools", None)
    valid_names = getattr(agent, "valid_tool_names", None)
    if tools:
        raise StageATextOnlyRefusal(
            "TOOL_SURFACE_PRESENT",
            f"{len(tools)} tool(s) resolved: {', '.join(sorted(_tool_names(agent)))}",
        )
    if valid_names:
        raise StageATextOnlyRefusal(
            "TOOL_SURFACE_PRESENT",
            f"tool names present: {', '.join(sorted(str(n) for n in valid_names))}",
        )

    enabled = getattr(agent, "enabled_toolsets", None)
    if enabled is None or list(enabled):
        raise StageATextOnlyRefusal(
            "POLICY_NOT_APPLIED",
            f"enabled_toolsets must be an empty selection, got {enabled!r}",
        )
    if getattr(agent, "max_iterations", None) != STAGE_A_TEXT_ONLY_POLICY["max_iterations"]:
        raise StageATextOnlyRefusal(
            "POLICY_NOT_APPLIED",
            f"max_iterations must be {STAGE_A_TEXT_ONLY_POLICY['max_iterations']}, "
            f"got {getattr(agent, 'max_iterations', None)!r}",
        )
    if getattr(agent, "max_tokens", None) != upstream.max_tokens:
        raise StageATextOnlyRefusal(
            "POLICY_NOT_APPLIED",
            f"max_tokens must be {upstream.max_tokens}, "
            f"got {getattr(agent, 'max_tokens', None)!r}",
        )

    if getattr(agent, "skip_context_files", None) is not True:
        raise StageATextOnlyRefusal(
            "CONTEXT_INHERITANCE", "skip_context_files is not True"
        )
    if getattr(agent, "load_soul_identity", None) is not False:
        raise StageATextOnlyRefusal(
            "CONTEXT_INHERITANCE", "load_soul_identity is not False"
        )
    # ``agent_init`` stores this one as ``bool(skip_background_review)``, so an
    # agent that quietly kept the auxiliary review turn is visible here.
    if getattr(agent, "skip_background_review", None) is not True:
        raise StageATextOnlyRefusal(
            "CONTEXT_INHERITANCE", "skip_background_review is not True"
        )

    if getattr(agent, "_memory_store", None) is not None:
        raise StageATextOnlyRefusal(
            "MEMORY_INHERITANCE", "a built-in memory store was created"
        )
    if getattr(agent, "_memory_manager", None) is not None:
        raise StageATextOnlyRefusal(
            "MEMORY_INHERITANCE", "an external memory provider was attached"
        )


def _require_opt_in(env: Mapping[str, str]) -> None:
    if env.get(OPT_IN_ENV) != OPT_IN_VALUE:
        raise StageATextOnlyRefusal(
            "OPT_IN_ABSENT",
            f"{OPT_IN_ENV} must be exactly {OPT_IN_VALUE!r}",
        )


def _reject_ambient_state(env: Mapping[str, str]) -> None:
    """Refuse when the caller's environment carries state we must not inherit.

    Presence, not truthiness: ``HERMES_YOLO_MODE=0`` is refused too.  Stage A
    declines to adjudicate what a bypass marker meant --- an operator that
    wants a Stage-A run launches it without the marker.
    """
    present = [k for k in APPROVAL_BYPASS_ENV_KEYS if env.get(k, "") != ""]
    if present:
        raise StageATextOnlyRefusal(
            "AMBIENT_APPROVAL_BYPASS",
            "approval-bypass markers present: " + ", ".join(present),
        )
    present = [k for k in AMBIENT_TOOL_SCOPE_ENV_KEYS if env.get(k, "") != ""]
    if present:
        raise StageATextOnlyRefusal(
            "AMBIENT_TOOL_SCOPE",
            "dispatcher-worker markers present: " + ", ".join(present),
        )


@contextmanager
def _scrubbed_environment(env: MutableMapping[str, str]) -> Iterator[None]:
    """Remove the refused keys for the duration of the call, then restore.

    Defence in depth behind :func:`_reject_ambient_state`: the refusal above
    already covers the caller's own environment, and this covers anything that
    sets a marker between the check and the construction, or that reads the
    environment lazily inside the construction.
    """
    removed: dict[str, str] = {}
    for key in APPROVAL_BYPASS_ENV_KEYS + AMBIENT_TOOL_SCOPE_ENV_KEYS:
        if key in env:
            removed[key] = env[key]
            del env[key]
    try:
        yield
    finally:
        for key, value in removed.items():
            env[key] = value


def _default_agent_factory(**kwargs: Any) -> Any:
    """Construct the accepted agent.  Imported lazily: importing this module
    must stay cheap and side-effect free, and ``run_agent`` pulls in the whole
    runtime."""
    from run_agent import AIAgent

    return AIAgent(**kwargs)


def run_stage_a_proposal(
    prompt: str,
    upstream: StageAUpstream,
    *,
    agent_factory: Callable[..., Any] | None = None,
    env: MutableMapping[str, str] | None = None,
) -> str:
    """Run one Stage-A request in proposal-only mode and return its text.

    Args:
        prompt: The Stage-A request text.
        upstream: Runtime-supplied endpoint/model/provider/bound.  No field is
            defaulted or resolved from ambient state.
        agent_factory: Injection seam for tests and for a caller that owns its
            own construction.  Defaults to the accepted ``AIAgent``.  A factory
            does not get to relax the policy: whatever it returns is still put
            through :func:`assert_text_only_surface`.
        env: Environment mapping to read and scrub.  Defaults to ``os.environ``.

    Returns:
        The model's final text.

    Raises:
        StageATextOnlyRefusal: on any failure of the opt-in, the ambient-state
            checks, the upstream validation or the resolved-surface
            verification.  Every refusal happens before or instead of a model
            call, never after a side effect.
    """
    environment = os.environ if env is None else env
    factory = _default_agent_factory if agent_factory is None else agent_factory

    _require_opt_in(environment)
    _reject_ambient_state(environment)
    upstream.validate()
    if not isinstance(prompt, str) or not prompt.strip():
        raise StageATextOnlyRefusal("PROMPT_EMPTY", "prompt must be non-empty text")

    kwargs = stage_a_agent_kwargs(upstream)

    with _scrubbed_environment(environment):
        agent = factory(**kwargs)
        try:
            assert_text_only_surface(agent, upstream)
            result = agent.run_conversation(
                prompt, system_message=STAGE_A_SYSTEM_PROMPT
            )
        finally:
            close = getattr(agent, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - cleanup must not mask the outcome
                    pass

    text = ""
    if isinstance(result, Mapping):
        text = str(result.get("final_response") or "")
    if not text.strip():
        raise StageATextOnlyRefusal(
            "NO_PROPOSAL_TEXT", "the run produced no final text"
        )
    return text


# ---------------------------------------------------------------------------
# Process entrypoint
# ---------------------------------------------------------------------------

#: Exactly the flags ``main`` accepts.  Anything else --- an unknown flag, a
#: repeat, a missing value, a positional argument, or the ``--flag=value``
#: spelling --- is refused.  A permissive parser here would be another way for
#: ambient argv to reach the run.
_CLI_FLAGS: tuple[str, ...] = ("--base-url", "--model", "--provider", "--max-tokens")


def _parse_argv(argv: Sequence[str]) -> StageAUpstream:
    seen: dict[str, str] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token not in _CLI_FLAGS:
            raise StageATextOnlyRefusal(
                "ARGV_REJECTED",
                f"unexpected argument {token!r}; accepted flags are "
                + " ".join(f"{flag} VALUE" for flag in _CLI_FLAGS),
            )
        if token in seen:
            raise StageATextOnlyRefusal("ARGV_REJECTED", f"repeated flag {token!r}")
        if index + 1 >= len(argv):
            raise StageATextOnlyRefusal("ARGV_REJECTED", f"{token!r} needs a value")
        seen[token] = argv[index + 1]
        index += 2

    missing = [flag for flag in _CLI_FLAGS if flag not in seen]
    if missing:
        raise StageATextOnlyRefusal(
            "ARGV_REJECTED", "missing required flags: " + " ".join(missing)
        )

    raw_max_tokens = seen["--max-tokens"]
    try:
        max_tokens = int(raw_max_tokens)
    except ValueError:
        raise StageATextOnlyRefusal(
            "ARGV_REJECTED", "--max-tokens must be an integer"
        ) from None

    return StageAUpstream(
        base_url=seen["--base-url"],
        model=seen["--model"],
        provider=seen["--provider"],
        max_tokens=max_tokens,
    )


def main(argv: Sequence[str] | None = None, *, stdin: Any = None, stdout: Any = None) -> int:
    """``python -m agent.stagea_text_only`` --- prompt on stdin, text on stdout.

    The prompt is read from stdin rather than argv so request content never
    lands in a process listing.  No credential is accepted on argv or read
    from the environment.

    Returns 0 on success and 2 on any refusal; the refusal code goes to stderr
    so a caller can branch on it without parsing the proposal.
    """
    args = sys.argv[1:] if argv is None else list(argv)
    in_stream = sys.stdin if stdin is None else stdin
    out_stream = sys.stdout if stdout is None else stdout
    try:
        upstream = _parse_argv(args)
        text = run_stage_a_proposal(in_stream.read(), upstream)
    except StageATextOnlyRefusal as refusal:
        sys.stderr.write(f"hermes stage-a text-only: refused {refusal}\n")
        return 2
    out_stream.write(text if text.endswith("\n") else text + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    raise SystemExit(main())
