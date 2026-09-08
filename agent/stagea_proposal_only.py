"""Stage-A proposal-only program identity (AI-Org #198, Phase A).

This module is a **separate program**, not a mode of ``hermes``.  It exists
because the Stage-A controller's trust boundary is *which executable it
pinned*, not a token the child presents: ``dyhano_stage_a.hermes_oneshot``
starts exactly ``(pinned_real_path, "-z")`` with a five-key environment and
no ``PATH``, so the only thing that distinguishes a capability-reduced run
from an ordinary one is the identity of the ``main`` that runs.

Nothing here inspects an environment variable, argv token, cwd, marker file
or config value to decide whether it is "armed".  There is no arming
predicate to present and therefore nothing an unrelated caller can spoof:
a caller either started *this* program or it did not.

What the program guarantees, in order:

1. **Exact argv.**  ``(-z,)`` and nothing else.  Any extra or different
   argument is refused before stdin is read.
2. **Closed input.**  stdin must be the canonical 14-key proposal request
   built by ``dyhano_stage_a.proposal.build_proposal_request``.  Unknown
   keys, a wrong schema version, or a resume/session/provider/credential
   token anywhere in the document is refused.
3. **A capability-reduced agent, verified after construction.**  The real
   ``run_agent.AIAgent`` factory is used unchanged; the program then reads
   the *resolved* state off the constructed object and refuses unless the
   surface really is empty.  Asserting the arguments it passed would prove
   nothing -- an empty ``enabled_toolsets`` list is honoured only because
   ``model_tools`` branches on ``is not None``, so passing ``None`` would
   silently mean *every* toolset.
4. **No egress during construction.**  Construction runs inside
   :class:`ConstructionEgressGuard`, which counts and denies outbound
   socket work.  If anything was attempted the program refuses instead of
   continuing, so a misconfigured deployment fails closed rather than
   quietly probing its endpoint.  See the class docstring for why this is
   a real guarantee and not a config hope.
5. **One framed line, or nothing.**  stdout carries exactly one
   ``DYHANO-STAGE-A-PROPOSAL-V1 <envelope>`` line of printable ASCII on
   success and nothing at all otherwise.  Every diagnostic goes to stderr,
   and library output produced during construction is captured so it can
   never be mistaken for a second proposal.

The program is a *producer* against the AI-Org contract; it is not a second
authority.  Admission stays where it already is: the controller revalidates
the returned bytes with the frozen ``dyhano_signed_grant`` parser and the
kernel's own path predicate (``dyhano_stage_a.proposal.validate_proposal``).
The checks here exist so this program never emits something it knows to be
malformed -- they do not, and must not, become an alternative grammar.
"""

# IMPORTANT: hermes_bootstrap must be the very first import -- UTF-8 stdio
# on Windows.  No-op on POSIX.  See hermes_bootstrap.py for full rationale.
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    # Graceful fallback when hermes_bootstrap isn't registered in the venv
    # yet -- happens during a partial ``hermes update``.  Missing bootstrap
    # means UTF-8 stdio setup is skipped on Windows; POSIX is unaffected.
    pass
else:
    # A Stage-A child is started with cwd set to a controller-chosen working
    # directory, so a same-named package there must not shadow Hermes modules.
    hermes_bootstrap.harden_import_path()

import contextlib
import io
import json
import socket
import sys


# ─────────────────────────────────────────────────────────────────────────
# The frozen cross-repository contract.
#
# These mirror the accepted AI-Org constants that bound this program's
# argv, stdin and stdout.  They are duplicated rather than imported because
# the Stage-A child runs with no PATH and no AI-Org code on sys.path; the
# controller is the authority for all of them and revalidates the result.
# ─────────────────────────────────────────────────────────────────────────

#: dyhano_stage_a.hermes_oneshot.FROZEN_ARGV_TAIL
FROZEN_ARGV_TAIL = ("-z",)

#: dyhano_stage_a.hermes_oneshot.PROPOSAL_FRAMING_PREFIX
FRAMING_PREFIX = "DYHANO-STAGE-A-PROPOSAL-V1 "

#: dyhano_stage_a.proposal.REQUEST_SCHEMA_VERSION
REQUEST_SCHEMA_VERSION = "dyhano-stage-a-proposal-request-v1"

#: dyhano_stage_a.proposal.REQUEST_FIELDS -- the complete, closed projection.
REQUEST_FIELDS = (
    "schema_version",
    "action_id",
    "decision_id",
    "action_intent",
    "kernel_version",
    "work_item_id",
    "contract_ref",
    "work_contract_fingerprint",
    "canonical_repo",
    "canonical_branch_ref",
    "expected_main_sha",
    "admitted_write_set",
    "envelope_schema_version",
    "max_envelope_bytes",
)

#: dyhano_stage_a.proposal.FORBIDDEN_REQUEST_TOKENS
FORBIDDEN_REQUEST_TOKENS = (
    "resume", "continue", "--session", "session_id", "chat", "history",
    "provider", "fallback", "api_key", "apikey", "token", "jwt", "secret",
    "password", "credential", "private_key", "client_secret", "bearer",
    "authorization", "netrc", "ssh", "keychain",
)

#: dyhano_stage_a.hermes_oneshot.MAX_REQUEST_BYTES / MAX_STDOUT_BYTES
MAX_REQUEST_BYTES = 262_144
MAX_STDOUT_BYTES = 262_144

# dyhano_signed_grant.envelope / .limits -- producer-side shape checks only.
ENVELOPE_TOP_KEYS = frozenset(("schema_version", "mutations"))
_CREATE_KEYS = frozenset(("op", "path", "content"))
_REPLACE_KEYS = frozenset(("op", "path", "base_blob_sha", "content"))
_DELETE_KEYS = frozenset(("op", "path", "base_blob_sha"))
_ENVELOPE_OP_KEYS = {"create": _CREATE_KEYS,
                     "replace": _REPLACE_KEYS,
                     "delete": _DELETE_KEYS}
_PATH_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/"
)
MAX_ENVELOPE_FILES = 32
MAX_FILE_CONTENT_BYTES = 65_536
MAX_ENVELOPE_PATH_CHARS = 200
MAX_ENVELOPE_PATH_SEGMENTS = 16


# ─────────────────────────────────────────────────────────────────────────
# The reduced capability policy and the one dedicated route.
#
# Deliberately literal.  The three route names below are not a provider or
# model *selector*: they name the single dedicated managed Stage-A route
# this program may speak on, fixed in source so the identity that reaches
# request shaping is bound before the first request instead of being left
# empty and filled in later by ambient routing.  There is still no
# endpoint, URL, header, method, proxy, credential or fallback here -- the
# deployment declares this route's endpoint and key under that provider
# name, and the program refuses whatever the factory resolved if it is
# anything other than that route.
# ─────────────────────────────────────────────────────────────────────────

#: The dedicated managed Stage-A route.  ``init_agent`` copies these into
#: ``agent.model`` / ``agent.provider`` / ``agent.api_mode`` and asks the
#: central router for *this* provider by name, so the resolved endpoint is
#: the dedicated route's own rather than an ``auto`` chain's pick whose
#: resolved model the accepted caller discards.
STAGEA_ROUTE_MODEL = "stagea-proposal-only"
STAGEA_ROUTE_PROVIDER = "stagea-local-relay"
STAGEA_ROUTE_API_MODE = "chat_completions"

#: Wire schemes an OpenAI-compatible Stage-A route may resolve to.  Not a
#: URL policy -- the deployment owns the endpoint -- only a refusal of a
#: resolved identity that is no HTTP endpoint at all (empty, or a virtual
#: ``moa://``-style facade).
STAGEA_ROUTE_URL_SCHEMES = ("http://", "https://")

#: An empty *selection*, never ``None``.  ``model_tools`` tests
#: ``enabled_toolsets is not None``, so ``None`` means "every toolset".
REDUCED_ENABLED_TOOLSETS = []

#: One bounded pass.  ``AIAgent`` defaults to ``sys.maxsize``.
REDUCED_MAX_ITERATIONS = 1

EXIT_OK = 0
EXIT_REFUSED = 2


class Refusal(Exception):
    """A deterministic, closed-family refusal.

    Carries a stable dotted ``code`` so the controller sees a classifiable
    stderr line rather than an arbitrary traceback.  Never carries input
    bytes, configuration values or credential material.
    """

    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _refuse(code):
    raise Refusal(code)


# ─────────────────────────────────────────────────────────────────────────
# Egress guard
# ─────────────────────────────────────────────────────────────────────────

class ConstructionEgressGuard:
    """Count and deny outbound socket work for the duration of a block.

    This is a *self-imposed* restriction, scoped to agent construction and
    fully restored on exit.  It changes no Hermes semantics outside the
    block and bypasses nothing: its only effect is to make this program
    refuse.

    It exists because "the reduced child performs no provider access" is
    otherwise a property of the deployment's configuration rather than of
    the program.  Accepted source reaches two endpoint-metadata probes
    during ``init_agent`` -- the tool-search context-length resolution and
    the Ollama ``num_ctx`` detection -- and both are wrapped in broad
    ``except Exception`` handlers, so a failed probe is indistinguishable
    from a probe that never happened.  Counting at the socket boundary is
    what makes the difference observable, and refusing on a non-zero count
    is what turns it into a guarantee this program can actually make.

    Attempts are counted, not merely blocked, so a caller can tell "nothing
    was tried" from "something was tried and swallowed".
    """

    __slots__ = ("attempts", "_saved")

    #: ``socket`` attributes replaced for the duration of the block.
    _MODULE_TARGETS = ("create_connection", "getaddrinfo")
    _SOCKET_TARGETS = ("connect", "connect_ex")

    def __init__(self):
        self.attempts = 0
        self._saved = ()

    def _deny(self, boundary):
        def _denied(*_args, **_kwargs):
            self.attempts += 1
            raise Refusal("construction.egress." + boundary)
        return _denied

    def __enter__(self):
        saved = []
        for name in self._MODULE_TARGETS:
            saved.append((socket, name, getattr(socket, name)))
            setattr(socket, name, self._deny(name))
        for name in self._SOCKET_TARGETS:
            saved.append((socket.socket, name, getattr(socket.socket, name)))
            setattr(socket.socket, name, self._deny(name))
        self._saved = tuple(saved)
        return self

    def __exit__(self, exc_type, exc, tb):
        for owner, name, original in reversed(self._saved):
            setattr(owner, name, original)
        self._saved = ()
        return False


# ─────────────────────────────────────────────────────────────────────────
# Request parsing
# ─────────────────────────────────────────────────────────────────────────

def parse_request(raw):
    """Parse and screen the controller's canonical request document.

    Bound-before-traversal: the byte length is proven before the parse, and
    the key set is proven before any value is read.
    """
    if not isinstance(raw, bytes):
        _refuse("request.type")
    if len(raw) == 0 or len(raw) > MAX_REQUEST_BYTES:
        _refuse("request.length")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        _refuse("request.encoding")
    try:
        document = json.loads(text)
    except ValueError:
        _refuse("request.json")
    if type(document) is not dict:
        _refuse("request.shape")
    if set(document) != set(REQUEST_FIELDS):
        # Exact closure in both directions: a missing field and an extra
        # field are equally refused, so a widened request cannot steer.
        _refuse("request.fields")
    if document["schema_version"] != REQUEST_SCHEMA_VERSION:
        _refuse("request.schema_version")
    _screen_request_hygiene(document)
    return document


def _screen_request_hygiene(document):
    """Refuse a request carrying a resume/provider/credential-shaped token.

    Mirrors ``dyhano_stage_a.proposal.assert_request_hygiene`` exactly: every
    key is screened, and so is every value except the one field the
    controller itself exempts.  ``admitted_write_set`` carries kernel
    contract truth -- path patterns already bounded by the envelope path
    grammar -- and screening its members would falsely refuse ordinary
    repository paths such as ``src/token.py`` or ``docs/provider-notes.md``.
    The exemption is value-scoped: that key is screened like every other, and
    every other field's scalar and list values keep being screened.
    """
    # dyhano_stage_a.proposal._UNSCREENED_REQUEST_FIELDS -- values only.
    unscreened_fields = frozenset(("admitted_write_set",))
    screened = list(document)
    for key, value in document.items():
        if key in unscreened_fields:
            continue
        if type(value) is str:
            screened.append(value)
        elif type(value) is list:
            screened.extend(item for item in value if type(item) is str)
    lowered = "\n".join(screened).lower()
    for token in FORBIDDEN_REQUEST_TOKENS:
        if token in lowered:
            _refuse("request.hygiene.forbidden_token")


# ─────────────────────────────────────────────────────────────────────────
# Envelope production
# ─────────────────────────────────────────────────────────────────────────

def canonical_dumps(value):
    """Byte-for-byte the canonicalization the frozen parser will re-derive.

    ``ensure_ascii=True`` is load-bearing twice over: it is what the
    accepted ``dyhano_signed_grant.canonical.canonical_dumps`` does, and it
    is what lets the result survive the controller's printable-ASCII
    framing check.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def validate_envelope_document(document, max_envelope_bytes,
                               expected_schema_version):
    """Prove an envelope well-formed and return its canonical bytes.

    Producer-side only.  The controller re-decides grammar, canonical byte
    identity, path closure and admission with the frozen parser and the
    kernel's own predicate; this refuses to *emit* something already known
    to be malformed rather than shipping it for the boundary to reject.

    ``expected_schema_version`` is required and deliberately carries no
    default: its only authority is the ``envelope_schema_version`` the
    controller put in this request, and a default would silently divorce
    the check from the request the envelope is answering.
    """
    if type(document) is not dict:
        _refuse("envelope.shape")
    if set(document) != ENVELOPE_TOP_KEYS:
        _refuse("envelope.top_keys")
    # Frozen ``parse_envelope_bytes`` proves the schema value before it reads
    # the mutations container, and refuses every other value outright, so
    # emitting one would spend the single bounded attempt on a certain
    # rejection at the controller's boundary.
    if type(expected_schema_version) is not str or not expected_schema_version:
        _refuse("envelope.schema_version.expected")
    schema_version = document["schema_version"]
    if type(schema_version) is not str:
        _refuse("envelope.schema_version.type")
    if schema_version != expected_schema_version:
        _refuse("envelope.schema_version.value")
    mutations = document["mutations"]
    if type(mutations) is not list:
        _refuse("envelope.mutations.type")
    if len(mutations) == 0 or len(mutations) > MAX_ENVELOPE_FILES:
        _refuse("envelope.mutations.length")

    seen_segments = []
    for mutation in mutations:
        if type(mutation) is not dict:
            _refuse("envelope.mutation.type")
        op = mutation.get("op")
        expected_keys = _ENVELOPE_OP_KEYS.get(op) if type(op) is str else None
        if expected_keys is None:
            _refuse("envelope.mutation.op")
        if set(mutation) != expected_keys:
            _refuse("envelope.mutation.keys")
        segments = _require_envelope_path(mutation["path"])
        if "content" in mutation:
            content = mutation["content"]
            if type(content) is not str:
                _refuse("envelope.mutation.content.type")
            if len(content.encode("utf-8")) > MAX_FILE_CONTENT_BYTES:
                _refuse("envelope.mutation.content.length")
        if "base_blob_sha" in mutation:
            # Present exactly for replace/delete -- the key closure says so.
            base_blob_sha = mutation["base_blob_sha"]
            if type(base_blob_sha) is not str:
                _refuse("envelope.mutation.base_blob_sha.type")
            # Frozen ``validation.require_sha40``: exactly 40 characters from
            # 0123456789abcdef.  Uppercase hex is a different string to the
            # parser, so refuse it here rather than emit a certain reject.
            if len(base_blob_sha) != 40:
                _refuse("envelope.mutation.base_blob_sha.length")
            if not all(c in "0123456789abcdef" for c in base_blob_sha):
                _refuse("envelope.mutation.base_blob_sha.grammar")
        seen_segments.append(segments)

    # Duplicate and prefix-collision closure: two mutations may not name the
    # same path, and no path may sit underneath another.
    for index, segments in enumerate(seen_segments):
        for other in seen_segments[index + 1:]:
            if len(segments) <= len(other):
                shorter, longer = segments, other
            else:
                shorter, longer = other, segments
            if longer[:len(shorter)] == shorter:
                _refuse("envelope.paths.collide")

    raw = canonical_dumps(document)
    if len(raw) == 0 or len(raw) > max_envelope_bytes:
        _refuse("envelope.length")
    return raw


def _require_envelope_path(value):
    if type(value) is not str:
        _refuse("envelope.path.type")
    if len(value) == 0 or len(value) > MAX_ENVELOPE_PATH_CHARS:
        _refuse("envelope.path.length")
    if not all(character in _PATH_CHARS for character in value):
        _refuse("envelope.path.grammar")
    segments = tuple(value.split("/"))
    if len(segments) > MAX_ENVELOPE_PATH_SEGMENTS:
        _refuse("envelope.path.depth")
    for segment in segments:
        if segment in ("", ".", ".."):
            _refuse("envelope.path.segment")
    if segments[0] == ".git":
        _refuse("envelope.path.git_dir")
    return segments


def frame_envelope(envelope_bytes):
    """Wrap canonical envelope bytes in the single framing line."""
    if type(envelope_bytes) is not bytes:
        _refuse("framing.type")
    if len(envelope_bytes) == 0:
        _refuse("framing.empty_payload")
    for byte in envelope_bytes:
        if byte < 0x20 or byte > 0x7E:
            _refuse("framing.non_printable")
    line = FRAMING_PREFIX + envelope_bytes.decode("ascii")
    # The parent bounds the whole stream, and truncation invalidates the
    # attempt before the exit code is even consulted -- so refuse here
    # rather than emit something the controller must throw away.
    if len(line.encode("ascii")) + 1 > MAX_STDOUT_BYTES:
        _refuse("framing.length")
    return line


# ─────────────────────────────────────────────────────────────────────────
# The reduced agent
# ─────────────────────────────────────────────────────────────────────────

def build_reduced_agent():
    """Construct the real ``AIAgent`` on the dedicated Stage-A route.

    The three route names are passed to the unchanged factory instead of
    being left empty.  Passing them is what makes the accepted
    no-explicit-credentials path ask the central router for *this* provider
    by name rather than run the ``auto`` chain and then discard the model
    it resolved -- which is how an empty model identity reached request
    shaping.  Nothing here supplies a URL, header, key or fallback: the
    deployment still owns the endpoint and credential, and
    :func:`verify_route_identity` refuses if what came back is not exactly
    this route.

    Returns ``(agent, attempted_egress)``.  ``run_agent`` is imported here
    rather than at module scope so importing this module stays cheap and
    side-effect-free.
    """
    guard = ConstructionEgressGuard()
    captured = io.StringIO()
    try:
        with guard, contextlib.redirect_stdout(captured):
            from run_agent import AIAgent
            agent = AIAgent(
                model=STAGEA_ROUTE_MODEL,
                provider=STAGEA_ROUTE_PROVIDER,
                api_mode=STAGEA_ROUTE_API_MODE,
                max_iterations=REDUCED_MAX_ITERATIONS,
                enabled_toolsets=REDUCED_ENABLED_TOOLSETS,
                skip_context_files=True,
                load_soul_identity=False,
                skip_memory=True,
                skip_background_review=True,
                quiet_mode=True,
            )
    finally:
        # Anything the factory printed belongs on stderr: stdout is the
        # framing channel and must carry the proposal line or nothing.
        noise = captured.getvalue()
        if noise:
            sys.stderr.write(noise)
    return agent, guard.attempts


def describe_resolved_surface(agent):
    """Read the reduced surface off the constructed object.

    Every value is read from the agent, never from the arguments that were
    passed to it -- that distinction is the whole point of the control.
    """
    return {
        "tool_count": len(getattr(agent, "tools", None) or []),
        "valid_tool_names": sorted(getattr(agent, "valid_tool_names", None) or []),
        "enabled_toolsets": getattr(agent, "enabled_toolsets", None),
        "enabled_toolsets_is_none": getattr(agent, "enabled_toolsets", None) is None,
        "max_iterations": getattr(agent, "max_iterations", None),
        "memory_enabled": bool(getattr(agent, "_memory_enabled", False)),
        "memory_manager_present": getattr(agent, "_memory_manager", None) is not None,
        "memory_store_present": getattr(agent, "_memory_store", None) is not None,
        "load_soul_identity": bool(getattr(agent, "load_soul_identity", False)),
        "skip_context_files": bool(getattr(agent, "skip_context_files", False)),
        "skip_background_review": bool(getattr(agent, "skip_background_review", False)),
    }


def verify_reduced_surface(agent):
    """Refuse unless the *resolved* surface really is capability-reduced."""
    surface = describe_resolved_surface(agent)
    if surface["enabled_toolsets_is_none"]:
        # ``None`` is not "no toolsets", it is "every toolset".
        _refuse("reduced.toolsets.absent_selection")
    if surface["enabled_toolsets"] != []:
        _refuse("reduced.toolsets.not_empty")
    if surface["tool_count"] != 0 or surface["valid_tool_names"]:
        _refuse("reduced.tools.not_empty")
    max_iterations = surface["max_iterations"]
    if type(max_iterations) is not int or isinstance(max_iterations, bool):
        _refuse("reduced.iterations.type")
    if max_iterations < 1 or max_iterations > REDUCED_MAX_ITERATIONS:
        _refuse("reduced.iterations.unbounded")
    if surface["memory_enabled"] or surface["memory_manager_present"] \
            or surface["memory_store_present"]:
        _refuse("reduced.memory.present")
    if surface["load_soul_identity"]:
        _refuse("reduced.soul.loaded")
    if not surface["skip_context_files"]:
        _refuse("reduced.context_files.loaded")
    if not surface["skip_background_review"]:
        _refuse("reduced.background_review.enabled")
    return surface


def describe_route_identity(agent):
    """Read the *resolved* route identity off the constructed object.

    Same rule as :func:`describe_resolved_surface`: every value is read back
    from the agent -- and from the client the factory actually built --
    never from the arguments passed in.  ``model`` is what the request
    builder puts on the wire for every API-mode branch; ``api_mode`` and
    ``base_url`` pick the transport and the endpoint it posts to.
    """
    client = getattr(agent, "client", None)
    client_base_url = getattr(client, "base_url", None)
    return {
        "model": getattr(agent, "model", None),
        "provider": getattr(agent, "provider", None),
        "requested_provider": getattr(agent, "requested_provider", None),
        "api_mode": getattr(agent, "api_mode", None),
        "base_url": getattr(agent, "base_url", None),
        "client_base_url": (None if client_base_url is None
                            else str(client_base_url)),
        "fallback_activated": bool(getattr(agent, "_fallback_activated", False)),
    }


def verify_route_identity(agent):
    """Refuse unless the resolved identity is exactly the dedicated route.

    Runs before the first conversation, so a run that would otherwise emit a
    request carrying an empty model, an ambiently resolved provider, a
    substituted API mode or an init-time fallback route refuses instead of
    speaking to it.  Every comparison is exact: no prefix match, no case
    folding, no default.

    The base URL is the deployment's to declare, so it is verified by
    provenance and agreement rather than a pinned literal: the router was
    asked for the dedicated provider by name, the resolved endpoint must be
    a real non-empty HTTP endpoint, and the object used for request shaping
    and the client that will carry the request must name the same one.
    """
    identity = describe_route_identity(agent)
    if identity["fallback_activated"]:
        _refuse("route.fallback.activated")
    if identity["model"] != STAGEA_ROUTE_MODEL:
        _refuse("route.model.mismatch")
    if identity["provider"] != STAGEA_ROUTE_PROVIDER:
        _refuse("route.provider.mismatch")
    if identity["requested_provider"] != STAGEA_ROUTE_PROVIDER:
        # A differing *requested* provider means the effective route was
        # canonicalized or re-pointed after this program named it.
        _refuse("route.requested_provider.mismatch")
    if identity["api_mode"] != STAGEA_ROUTE_API_MODE:
        _refuse("route.api_mode.mismatch")
    base_url = identity["base_url"]
    if type(base_url) is not str or not base_url:
        _refuse("route.base_url.absent")
    if not base_url.lower().startswith(STAGEA_ROUTE_URL_SCHEMES):
        _refuse("route.base_url.scheme")
    client_base_url = identity["client_base_url"]
    if type(client_base_url) is not str or not client_base_url:
        _refuse("route.client.absent")
    if client_base_url.rstrip("/") != base_url.rstrip("/"):
        _refuse("route.client.base_url_mismatch")
    return identity


# ─────────────────────────────────────────────────────────────────────────
# The bounded proposal task
# ─────────────────────────────────────────────────────────────────────────

SYSTEM_INSTRUCTION = (
    "You produce exactly one Stage-A mutation envelope and nothing else.\n"
    "Reply with a single JSON object with exactly two keys: "
    '"schema_version" and "mutations".\n'
    '"mutations" is a non-empty list; each item is an object with "op" set '
    'to "create", "replace" or "delete".\n'
    '"create" carries "path" and "content"; "replace" carries "path", '
    '"base_blob_sha" and "content"; "delete" carries "path" and '
    '"base_blob_sha".\n'
    "Every path must come from the admitted write set you are given.\n"
    "Emit no prose, no explanation and no code fence."
)


def build_task_prompt(request):
    """Derive the model prompt from the request and nothing else.

    A pure function of the already-screened request document, mirroring the
    controller-side property that the request is a pure projection of the
    kernel action: no ambient string, caller text or environment value can
    enter the prompt.
    """
    return (
        "Stage-A proposal request:\n"
        + canonical_dumps(request).decode("ascii")
        + "\n\nProduce the envelope now."
    )


def run_proposal(agent, request):
    """Run the single bounded attempt and return canonical envelope bytes."""
    result = agent.run_conversation(
        build_task_prompt(request),
        system_message=SYSTEM_INSTRUCTION,
    )
    if type(result) is not dict:
        _refuse("attempt.result.type")
    text = result.get("response") or result.get("content") or ""
    if type(text) is not str or not text.strip():
        _refuse("attempt.result.empty")
    try:
        document = json.loads(text)
    except ValueError:
        _refuse("attempt.result.json")
    return validate_envelope_document(
        document,
        int(request["max_envelope_bytes"]),
        request["envelope_schema_version"],
    )


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────

def main(argv=None):
    """Run one proposal-only attempt.  Returns a process exit status."""
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    try:
        if arguments != FROZEN_ARGV_TAIL:
            _refuse("argv.not_frozen_tail")

        request = parse_request(_read_stdin_bounded())

        try:
            agent, attempted_egress = build_reduced_agent()
        except Refusal:
            raise
        except Exception:
            # The factory itself refused -- the ordinary case when no
            # provider is configured, which is exactly the default-off
            # posture this program is supposed to have.  Classify it
            # rather than letting an opaque traceback stand in for it.
            _refuse("construction.factory_refused")
        if attempted_egress:
            # Something reached for the network while the agent was being
            # built.  Accepted source swallows those failures, so continuing
            # would mean running a child whose isolation was never actually
            # established.  Fail closed instead.
            _refuse("construction.egress_attempted")
        verify_reduced_surface(agent)
        # Bind-then-verify, before the first conversation: an empty or
        # ambiently resolved identity must refuse rather than be sent.
        verify_route_identity(agent)

        line = frame_envelope(run_proposal(agent, request))
    except Refusal as refusal:
        sys.stderr.write("stagea-proposal-only: refused: %s\n" % refusal.code)
        return EXIT_REFUSED
    except Exception as error:  # noqa: BLE001 -- closed, classifiable exit
        # Never let an unclassified traceback become the child's behaviour:
        # the controller's outcome family has no room for one, and a partial
        # observation is not a proposal.
        sys.stderr.write(
            "stagea-proposal-only: failed: %s\n" % type(error).__name__
        )
        return EXIT_REFUSED

    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    return EXIT_OK


def _read_stdin_bounded():
    """Read at most one byte more than the cap, so overflow is detectable."""
    stream = getattr(sys.stdin, "buffer", sys.stdin)
    return stream.read(MAX_REQUEST_BYTES + 1)


if __name__ == "__main__":
    sys.exit(main())
