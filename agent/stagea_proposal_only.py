"""Stage-A proposal-only Hermes program (issue #198, Phase A).

Bootstrap-first entry point: resolve ONE task-scoped non-secret Hermes config
root under the confined cwd, install it as the context-local home override
BEFORE any config/AIAgent construction, issue exactly one real proposal
request on the accepted chat-completions transport, and frame at most one
mutation envelope on stdout.  Fail closed: no credential, tool, retry,
auxiliary request, environment mutation or HERMES_HOME expansion.  Route,
model and context identity are activation truth read from that root.
"""

import hermes_bootstrap  # noqa: F401  bootstrap first, before anything else

import json
import os
import stat
import sys
from pathlib import Path
from urllib.parse import urlsplit

CONFIG_ROOT_NAME = ".stagea-hermes-home"
FRAMING_PREFIX = "DYHANO-STAGE-A-PROPOSAL-V1 "
ENVELOPE_SCHEMA_VERSION = "dyhano-trusted-boundary-envelope-v1"
REQUEST_SCHEMA_VERSION = "dyhano-stage-a-proposal-request-v1"
API_MODE = "chat_completions"
NO_CREDENTIAL_PLACEHOLDER = "stage-a-proposal-only-no-credential"
FROZEN_ARGV = ("-z",)
REQUEST_FIELDS = frozenset((
    "schema_version", "action_id", "decision_id", "action_intent",
    "kernel_version", "work_item_id", "contract_ref",
    "work_contract_fingerprint", "canonical_repo", "canonical_branch_ref",
    "expected_main_sha", "admitted_write_set", "envelope_schema_version",
    "max_envelope_bytes"))
UNSCREENED_REQUEST_FIELDS = frozenset(("admitted_write_set",))
FORBIDDEN_REQUEST_TOKENS = (
    "resume", "continue", "--session", "session_id", "chat", "history",
    "provider", "fallback", "api_key", "apikey", "token", "jwt", "secret",
    "password", "credential", "private_key", "client_secret", "bearer",
    "authorization", "netrc", "ssh", "keychain")
CREDENTIAL_KEY_TOKENS = (
    "key", "token", "secret", "password", "credential", "bearer",
    "authorization", "netrc", "cookie")
# F3: the same vocabulary decides consumed VALUES, matched as delimited words
# so an ordinary route ("tokenhub") is not mistaken for "?token=<secret>".
CREDENTIAL_VALUE_TOKENS = CREDENTIAL_KEY_TOKENS + ("apikey", "jwt", "passwd")
# F6: the accepted producer's own forms, read from accepted main, not invented.
# DISPATCH_AUTHOR is the only intent dyhano_stage_a/proposal.py builds or
# revalidates a proposal request for (build_proposal_request and
# validate_proposal both pin expected_intent="DISPATCH_AUTHOR").  The grammars
# below are dyhano_stage_a/contracts.py require_token / require_ref_branch_name
# / require_name_with_owner / require_sha40; the 32-char kernel_version bound is
# dyhano_stage_a/snapshot.py._require_registered_kernel; MAX_ADMITTED_PATHS is
# contracts.MAX_SEQUENCE_ITEMS.
ADMITTED_ACTION_INTENT = "DISPATCH_AUTHOR"
TOKEN_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
MAX_TOKEN_CHARS = 128
MAX_KERNEL_VERSION_CHARS = 32
MAX_REF_NAME_CHARS = 120
MAX_NAME_WITH_OWNER_CHARS = 140
MAX_SCALAR_CHARS = 200
MAX_ADMITTED_PATHS = 256
TOP_KEYS = frozenset(("schema_version", "mutations"))
OP_KEYS = {"create": frozenset(("op", "path", "content")),
           "replace": frozenset(("op", "path", "base_blob_sha", "content")),
           "delete": frozenset(("op", "path", "base_blob_sha"))}
PATH_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/")
HEX_CHARS = frozenset("0123456789abcdef")
MAX_ENVELOPE_BYTES = 262144
MAX_ENVELOPE_FILES = 32
MAX_FILE_CONTENT_BYTES = 65536
MAX_PATH_CHARS = 200
MAX_PATH_SEGMENTS = 16
MAX_STDOUT_BYTES = 262144
# F5: one budget derived once instead of two ceilings that collide.  The
# controller marks a stream truncated the moment its buffer REACHES the cap and
# refuses a longer stdout (dyhano_stage_a/hermes_oneshot.py pump_bounded /
# extract_framed_envelope), so the framed line plus its newline must stay
# strictly below MAX_STDOUT_BYTES.  request["max_envelope_bytes"] stays the
# producer's outer bound; this is what is advertised AND enforced here.
MAX_FRAMED_STDOUT_BYTES = MAX_STDOUT_BYTES - 1
MAX_EMITTED_ENVELOPE_BYTES = MAX_FRAMED_STDOUT_BYTES - len(FRAMING_PREFIX) - 1
INSTRUCTIONS = (
    "Answer one bounded Stage-A proposal request. The user message is "
    "the request JSON; obey it and this contract exactly. Emit exactly "
    "one line: " + FRAMING_PREFIX + "<envelope>. Emit nothing else: no "
    "prose, no explanation, no markdown, no code fence, no tool call. "
    "Your entire reply is that single line; a reply carrying any other "
    "line, before or after it, is refused, not searched for an "
    "acceptable line. "
    "You get one attempt; any violation is refused without retry.\n"
    "<envelope> is one canonical printable-ASCII JSON object: keys "
    "sorted, no whitespace between tokens, non-ASCII escaped, at most "
    + str(MAX_EMITTED_ENVELOPE_BYTES) + " bytes. Its keys are exactly "
    "schema_version and mutations, with schema_version "
    + ENVELOPE_SCHEMA_VERSION + " and mutations a list of 1 to "
    + str(MAX_ENVELOPE_FILES) + " mutation objects.\n"
    "Each mutation object has exactly one of these closed key sets and "
    "no other key: {op=create, path, content}; {op=replace, path, "
    "base_blob_sha, content}; {op=delete, path, base_blob_sha}. No "
    "other op exists. base_blob_sha is required by replace and delete "
    "only, and is exactly 40 lowercase hex characters naming the blob "
    "currently at that path. content is a JSON string of at most "
    + str(MAX_FILE_CONTENT_BYTES) + " UTF-8 bytes.\n"
    "Propose only paths listed in the request's admitted_write_set; "
    "every other path is refused before this program prints anything. "
    "A path is relative, at most "
    + str(MAX_PATH_CHARS) + " characters and " + str(MAX_PATH_SEGMENTS)
    + " slash-separated segments, uses only ASCII letters, digits, "
    "'.', '_', '-' and '/', has no empty, '.' or '..' segment, and does "
    "not start with the .git segment. Two mutations may not share a "
    "path, and no path may be an ancestor (directory prefix) of "
    "another mutation's path.")


class ProposalRefusal(Exception):
    """Deterministic fail-closed refusal carrying a stable code."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


def need(condition, code):
    """Fail closed unless *condition* holds."""
    if not condition:
        raise ProposalRefusal(code)


def _pos_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _no_dupes(pairs):
    mapping = {}
    for key, value in pairs:
        need(type(key) is str and key not in mapping, "json.duplicate_key")
        mapping[key] = value
    return mapping


def canonical_bytes(value):
    """The frozen boundary's canonical form: sorted, tight, ASCII."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")


def _owned_private_stat(path, code):
    try:
        info = os.lstat(path)
    except OSError:
        raise ProposalRefusal(code + ".absent") from None
    need(not stat.S_ISLNK(info.st_mode), code + ".symlink")
    need(not stat.S_IMODE(info.st_mode) & 0o022, code + ".writable_by_others")
    getuid = getattr(os, "getuid", None)
    need(getuid is None or info.st_uid == getuid(), code + ".foreign_owner")
    return info


def resolve_config_root(cwd=None):
    """The one fixed task-scoped config root under the confined cwd."""
    base = Path(cwd) if cwd is not None else Path.cwd()
    need(base.is_absolute(), "root.cwd_not_absolute")
    real_base = Path(os.path.realpath(base))
    root = real_base / CONFIG_ROOT_NAME
    need(Path(os.path.realpath(root)) == root and root.parent == real_base,
         "root.escape")
    need(stat.S_ISDIR(_owned_private_stat(root, "root").st_mode),
         "root.not_directory")
    return root


def activate_config_root(root):
    """Install the context-local home override; prove it took effect."""
    from hermes_constants import (get_hermes_home, reset_hermes_home_override,
                                  set_hermes_home_override)

    token = set_hermes_home_override(root)
    if Path(get_hermes_home()) != root:
        reset_hermes_home_override(token)
        raise ProposalRefusal("home.override_ineffective")
    return token


def _delimited(text, token):
    """True when *token* occurs in *text* as a whole delimited word."""
    start = text.find(token)
    while start != -1:
        before = text[start - 1] if start else ""
        stop = start + len(token)
        after = text[stop] if stop < len(text) else ""
        if not before.isalnum() and not after.isalnum():
            return True
        start = text.find(token, start + 1)
    return False


def screen_scalar_value(value):
    """F3: refuse a consumed scalar that CARRIES credential material.

    Key-name screening alone let ``model.base_url: https://h/v1?token=<secret>``
    through: resolve_route forwards that URL and agent/agent_init.py:1299-1309
    turns its query into the client's ``default_query``, transmitting the
    secret on the proposal request; userinfo is presented the same way.  A
    query cannot be proven non-secret here and is forwarded verbatim, so any
    query/fragment/userinfo on a consumed URL is refused; every other scalar is
    refused when it names a credential as a delimited word, under any key.
    """
    # The task route is literal activation truth. Never interpolate inherited
    # environment values (including either shell or Windows syntax).
    need("$" not in value and "%" not in value, "config.environment_syntax")
    lowered = value.lower()
    if "://" in lowered:
        parsed = urlsplit(value)
        if parsed.scheme and parsed.netloc:
            need("@" not in parsed.netloc, "config.credential_url.userinfo")
            need(not parsed.query, "config.credential_url.query")
            need(not parsed.fragment, "config.credential_url.fragment")
    for token in CREDENTIAL_VALUE_TOKENS:
        need(not _delimited(lowered, token), "config.credential_value")


def screen_no_credential(node, default=None, depth=0):
    """Refuse credential-shaped material this task root contributes.

    The task-only loader supplies no defaults, so every raw task scalar is
    screened before route interpretation; no environment expansion occurs.
    """
    need(depth <= 8, "config.depth")
    if isinstance(node, dict):
        base = default if isinstance(default, dict) else {}
        for key, value in node.items():
            if key in base and base[key] == value:
                continue
            need(not any(t in str(key).lower() for t in CREDENTIAL_KEY_TOKENS),
                 "config.credential_key")
            screen_no_credential(value, base.get(key), depth + 1)
    elif isinstance(node, (list, tuple)):
        for item in node:
            screen_no_credential(item, None, depth + 1)
    elif isinstance(node, str):
        # Only contributed scalars reach here: a subtree still equal to its
        # shipped default was skipped above.
        screen_scalar_value(node)


def assert_no_managed_overlay():
    """Refuse managed scope without reading any external overlay content."""
    from hermes_cli import managed_scope

    need(not managed_scope.get_managed_dir(), "config.managed_dir")


def read_task_config(root):
    """Read literal task config without general Hermes loader side effects.

    The general loader initializes the home, expands environment values and
    merges external managed configuration even on its readonly fast path.
    This task-only entry point needs none of those behaviors or defaults: every
    load-bearing route value must be explicitly pinned in this one file.
    """
    import yaml
    from hermes_constants import get_hermes_home

    path = root / "config.yaml"
    need(stat.S_ISREG(_owned_private_stat(path, "config").st_mode),
         "config.not_regular_file")
    need(Path(get_hermes_home()) / "config.yaml" == path, "config.loader_path")
    assert_no_managed_overlay()
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_ENVELOPE_BYTES + 1)
        need(len(raw) <= MAX_ENVELOPE_BYTES, "config.bytes_length")
        document = yaml.safe_load(raw)
    except (OSError, ValueError, yaml.YAMLError):
        raise ProposalRefusal("config.parse") from None
    need(isinstance(document, dict) and document, "config.not_mapping")
    screen_no_credential(document)
    return document


def resolve_route(config):
    """Route/model/context identity: activation truth, read not invented."""
    section = config.get("model") if isinstance(config, dict) else None
    need(isinstance(section, dict), "route.model_section")
    values = []
    for field in ("default", "base_url", "provider"):
        value = section.get(field)
        need(isinstance(value, str) and value.strip(), "route." + field)
        values.append(value.strip())
    model, base_url, provider = values
    need(provider.lower() != "lmstudio", "route.provider_probes")
    need(_pos_int(section.get("context_length")), "route.context_pin")
    from agent.model_metadata import is_local_endpoint

    # Retained route contract: a local/loopback route must pin its served
    # window.  build_client constructs no generic agent, so nothing probes the
    # endpoint either way; the pin stays a required, directly tested property.
    need(not is_local_endpoint(base_url)
         or _pos_int(section.get("ollama_num_ctx")), "route.local_probe_pin")
    return model, base_url, provider


def _screened_strings(value, depth=0):
    """Every string a non-exempt request value carries, at any depth."""
    need(depth <= 8, "request.depth")
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            out.append(str(key))
            out.extend(_screened_strings(item, depth + 1))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_screened_strings(item, depth + 1))
        return out
    return [value if isinstance(value, str) else str(value)]


def _require_str(value, limit, code):
    need(type(value) is str and 0 < len(value) <= limit, code)
    return value


def _require_token(value, code):
    """dyhano_stage_a/contracts.py require_token."""
    _require_str(value, MAX_TOKEN_CHARS, code)
    need(all(char in TOKEN_CHARS for char in value), code + ".grammar")
    need(value[0] not in "._-" and value[-1] not in "._-", code + ".grammar")
    return value


def _require_name_with_owner(value, code):
    """dyhano_stage_a/contracts.py require_name_with_owner."""
    _require_str(value, MAX_NAME_WITH_OWNER_CHARS, code)
    parts = value.split("/")
    need(len(parts) == 2, code + ".grammar")
    for part in parts:
        need(part and all(char in TOKEN_CHARS for char in part),
             code + ".grammar")
    return value


def _require_branch_ref(value, code):
    """dyhano_stage_a/contracts.py require_ref_branch_name."""
    _require_str(value, MAX_REF_NAME_CHARS, code)
    for segment in value.split("/"):
        need(segment not in ("", ".", ".."), code + ".grammar")
        need(all(char in TOKEN_CHARS for char in segment), code + ".grammar")
        need(not segment.startswith(".") and not segment.endswith(".lock"),
             code + ".grammar")
    return value


def _require_sha40(value, code):
    """dyhano_stage_a/contracts.py require_sha40."""
    _require_str(value, 40, code)
    need(len(value) == 40 and all(char in HEX_CHARS for char in value),
         code + ".grammar")
    return value


def admitted_paths(document):
    """F1: the validated, normalized write set admission is judged against.

    Every member is proven against the same path grammar the envelope uses and
    normalized to its segment tuple, so membership is decided on normalized
    forms.  A member that is not a literal path in that grammar is refused
    rather than silently reinterpreted: the accepted contract already binds
    every proposable path to exactly this grammar, so nothing proposable is
    lost, and the controller still re-decides with the frozen kernel's own
    predicate (dyhano_stage_a/proposal.py validate_proposal).
    """
    entries = document["admitted_write_set"]
    need(type(entries) is list and 0 < len(entries) <= MAX_ADMITTED_PATHS,
         "request.admitted_write_set")
    admitted = set()
    for entry in entries:
        admitted.add(_segments(entry, "request.admitted_write_set.path"))
    need(len(admitted) == len(entries), "request.admitted_write_set.duplicate")
    return frozenset(admitted)


def validate_request_contract(document):
    """F6: the exact value contract the accepted producer emits.

    Key closure plus three constant comparisons accepted a "MERGE" intent, an
    integer SHA, object-valued identifiers and a mapping-valued write set --
    each of which would still have consumed the one authorized request.
    kernel_version is bounded, never pinned to a value: the kernel registry is
    append-only and belongs to the producer, not to this entry point.
    """
    need(document["action_intent"] == ADMITTED_ACTION_INTENT,
         "request.action_intent")
    _require_token(document["action_id"], "request.action_id")
    _require_token(document["decision_id"], "request.decision_id")
    _require_str(document["kernel_version"], MAX_KERNEL_VERSION_CHARS,
                 "request.kernel_version")
    _require_str(document["work_item_id"], MAX_TOKEN_CHARS,
                 "request.work_item_id")
    _require_str(document["contract_ref"], MAX_SCALAR_CHARS,
                 "request.contract_ref")
    _require_str(document["work_contract_fingerprint"], MAX_TOKEN_CHARS,
                 "request.work_contract_fingerprint")
    _require_name_with_owner(document["canonical_repo"],
                             "request.canonical_repo")
    _require_branch_ref(document["canonical_branch_ref"],
                        "request.canonical_branch_ref")
    _require_sha40(document["expected_main_sha"], "request.expected_main_sha")
    admitted_paths(document)
    return document


def parse_request(raw):
    """F1: a closed, hygienic projection of the frozen kernel action."""
    need(type(raw) is bytes and 0 < len(raw) <= MAX_ENVELOPE_BYTES,
         "request.bytes")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_dupes)
    except (UnicodeDecodeError, ValueError):
        raise ProposalRefusal("request.parse") from None
    need(type(document) is dict and set(document) == set(REQUEST_FIELDS),
         "request.closure")
    need(document["schema_version"] == REQUEST_SCHEMA_VERSION,
         "request.schema_version")
    need(document["envelope_schema_version"] == ENVELOPE_SCHEMA_VERSION,
         "request.envelope_schema_version")
    need(document["max_envelope_bytes"] == MAX_ENVELOPE_BYTES,
         "request.max_envelope_bytes")
    screened = []
    for key in sorted(document):
        screened.append(key)
        if key not in UNSCREENED_REQUEST_FIELDS:
            screened.extend(_screened_strings(document[key]))
    lowered = "\n".join(screened).lower()
    for token in FORBIDDEN_REQUEST_TOKENS:
        need(token not in lowered, "request.forbidden_token")
    # F6: hygiene screening stays first so a steering token is still reported
    # as one; the value contract then proves every load-bearing form BEFORE
    # this request can consume the single authorized provider attempt.
    return validate_request_contract(document)


#: F2: retries disabled at the client.  The controller authorizes exactly ONE
#: attempt (dyhano_stage_a/hermes_oneshot.py AttemptGuard consumes the action
#: identity on the first start), so an SDK retry would silently turn that one
#: authorized attempt into several provider requests.
MAX_PROVIDER_RETRIES = 0
#: Strictly below the controller's own MAX_ONESHOT_SECONDS (900) so a stalled
#: provider ends as a printed refusal rather than a killed child, which the
#: controller could only report as UNKNOWN_MODEL_ATTEMPT_INTERRUPTED.
PROVIDER_TIMEOUT_SECONDS = 600
# Both the provider generation and the HTTP response body are bounded before
# SDK JSON parsing. The latter includes protocol metadata/JSON escaping.
MAX_COMPLETION_TOKENS = 32768
MAX_PROVIDER_RESPONSE_BYTES = 4 * MAX_STDOUT_BYTES


class ProposalClient:
    """The Stage-A provider seam: one client, no agent, no plugin surface.

    Deliberately NOT ``run_agent.AIAgent``.  That constructor calls
    ``discover_plugins()`` unconditionally (agent/agent_init.py:1607-1615),
    which scans the active task root's ``plugins/`` and, with
    ``HERMES_ENABLE_PROJECT_PLUGINS`` inherited, the confined repository's
    ``.hermes/plugins``, then executes each plugin ``__init__.py`` through
    ``spec.loader.exec_module`` (hermes_cli/plugins.py:5490-5504) -- arbitrary
    code before the proposal request even though no tool schema is emitted
    (F7).  On a permitted OpenRouter route the same constructor starts the
    metadata prewarm thread (agent/agent_init.py:852-866) whose
    ``fetch_model_metadata()`` performs a second network request (F2).  It also
    loads custom-provider config and a fallback chain this entry point must not
    have.  Generic Hermes behaviour is unchanged: this path simply does not go
    through it, and constructs the OpenAI-compatible client and nothing else.
    """

    __slots__ = ("model", "base_url", "provider", "api_mode", "api_key",
                 "client")

    def __init__(self, model, base_url, provider, client):
        self.model = model
        self.base_url = base_url
        self.provider = provider
        self.api_mode = API_MODE
        self.api_key = NO_CREDENTIAL_PLACEHOLDER
        self.client = client


def build_client(model, base_url, provider):
    """Side-effect-free Stage-A client construction on the explicit route."""
    import httpx
    from openai import OpenAI

    class BoundedStream(httpx.SyncByteStream):
        def __init__(self, inner):
            self.inner = inner

        def __iter__(self):
            total = 0
            for chunk in self.inner:
                total += len(chunk)
                need(total <= MAX_PROVIDER_RESPONSE_BYTES, "response.bytes_length")
                yield chunk

        def close(self):
            self.inner.close()

    class BoundedTransport(httpx.BaseTransport):
        def __init__(self):
            self.inner = httpx.HTTPTransport(retries=0, trust_env=False)

        def handle_request(self, request):
            response = self.inner.handle_request(request)
            # Compressed bodies can expand after the raw-byte bound. Refuse
            # them before HTTPX's decoder or the SDK can accumulate content.
            if response.headers.get("content-encoding", "identity") != "identity":
                response.close()
                raise ProposalRefusal("response.content_encoding")
            response.stream = BoundedStream(response.stream)
            return response

        def close(self):
            self.inner.close()

    client = OpenAI(api_key=NO_CREDENTIAL_PLACEHOLDER, base_url=base_url,
                    max_retries=MAX_PROVIDER_RETRIES,
                    timeout=PROVIDER_TIMEOUT_SECONDS,
                    http_client=httpx.Client(
                        transport=BoundedTransport(), follow_redirects=False,
                        trust_env=False, timeout=PROVIDER_TIMEOUT_SECONDS,
                        headers={"Accept-Encoding": "identity"}))
    need(getattr(client, "max_retries", None) == MAX_PROVIDER_RETRIES,
         "route.retries_enabled")
    return ProposalClient(model, base_url, provider, client)


def create_completion(client, **kwargs):
    """The single provider seam; offline tests patch exactly this."""
    try:
        return client.chat.completions.create(**kwargs)
    except ProposalRefusal:
        raise
    except Exception:
        # SDK wrappers may wrap a transport refusal. Do not retry or expose
        # response bodies, route details or provider diagnostics on stderr.
        raise ProposalRefusal("response.provider_failure") from None


def build_request_kwargs(seam, document):
    """Build the one request on the accepted transport, with no tools."""
    from agent.transports import get_transport

    need(getattr(seam, "api_mode", "") == API_MODE, "route.api_mode")
    transport = get_transport(API_MODE)
    need(transport is not None, "route.transport")
    model = getattr(seam, "model", "")
    need(isinstance(model, str) and model.strip(), "route.model_empty")
    messages = [{"role": "system", "content": INSTRUCTIONS},
                {"role": "user",
                 "content": canonical_bytes(document).decode("ascii")}]
    kwargs = transport.build_kwargs(model=model, messages=messages, tools=None)
    for banned in ("tools", "tool_choice", "functions", "function_call",
                   "stream"):
        need(banned not in kwargs, "request.tool_surface")
    need(kwargs.get("model") == model
         and len(kwargs.get("messages") or []) == 2, "request.identity")
    kwargs["max_tokens"] = MAX_COMPLETION_TOKENS
    return kwargs


def response_text(response):
    """Exactly one proposal-only choice; any tool call is a refusal."""
    choices = getattr(response, "choices", None)
    need(isinstance(choices, (list, tuple)) and len(choices) == 1,
         "response.choices")
    message = getattr(choices[0], "message", None)
    need(not getattr(message, "tool_calls", None), "response.tool_calls")
    text = getattr(message, "content", None)
    need(isinstance(text, str) and text, "response.empty")
    return text


def extract_framed(text):
    """F4: the COMPLETE response must BE exactly one framed line.

    The executable contract states the reply is exactly one line and that any
    violation is refused without retry, so a reply carrying prose before or
    after the frame is refused rather than filtered down to the one acceptable
    line.  One trailing newline (and its CRLF form) is the line terminator, not
    extra content.  Two framing lines keep the distinct ``framing.ambiguous``
    code, which the controller maps to its own terminal reason.
    """
    body = text[:-1] if text.endswith("\n") else text
    lines = body.split("\n")
    framed = [ln for ln in lines if ln.startswith(FRAMING_PREFIX)]
    need(framed, "framing.absent")
    need(len(framed) == 1, "framing.ambiguous")
    need(len(lines) == 1, "framing.surrounding_text")
    payload = framed[0][len(FRAMING_PREFIX):]
    if payload.endswith("\r"):
        payload = payload[:-1]
    need(payload, "framing.empty_payload")
    for char in payload:
        need(" " <= char <= "~", "framing.non_printable")
    return payload.encode("ascii")


def _segments(value, code="envelope.path"):
    """One path proven against the grammar and normalized to its segments.

    *code* lets the request half report request-family refusals while the
    grammar itself stays single-sourced with the envelope half.
    """
    need(isinstance(value, str) and 0 < len(value) <= MAX_PATH_CHARS, code)
    need(all(char in PATH_CHARS for char in value), code + ".grammar")
    parts = tuple(value.split("/"))
    need(len(parts) <= MAX_PATH_SEGMENTS, code + ".depth")
    for part in parts:
        need(part not in ("", ".", ".."), code + ".segment")
    need(parts[0] != ".git", code + ".git_dir")
    return parts


def validate_envelope(raw, admitted):
    """F2: the frozen mutation-envelope grammar plus write-set admission.

    *admitted* is the validated, normalized frozenset from admitted_paths().
    It is a REQUIRED argument precisely so no caller can reach this boundary
    holding only the response half of the decision.
    """
    need(0 < len(raw) <= MAX_EMITTED_ENVELOPE_BYTES, "envelope.bytes_length")
    try:
        top = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_dupes)
    except (UnicodeDecodeError, ValueError):
        raise ProposalRefusal("envelope.parse") from None
    need(type(top) is dict and set(top) == set(TOP_KEYS), "envelope.closure")
    need(top["schema_version"] == ENVELOPE_SCHEMA_VERSION,
         "envelope.schema_version")
    mutations = top["mutations"]
    need(type(mutations) is list and 0 < len(mutations) <= MAX_ENVELOPE_FILES,
         "envelope.mutations")
    seen = set()
    for entry in mutations:
        need(type(entry) is dict, "envelope.mutation")
        keys = OP_KEYS.get(entry.get("op"))
        need(keys is not None and set(entry) == set(keys),
             "envelope.mutation.op")
        parts = _segments(entry["path"])
        # F1: the write-set authority boundary is decided HERE, before stdout;
        # it is not prompt-only guidance.
        need(parts in admitted, "envelope.path.not_admitted")
        need(parts not in seen, "envelope.path.duplicate")
        seen.add(parts)
        if "base_blob_sha" in keys:
            sha = entry["base_blob_sha"]
            need(isinstance(sha, str) and len(sha) == 40
                 and all(c in HEX_CHARS for c in sha),
                 "envelope.base_blob_sha")
        if "content" in keys:
            content = entry["content"]
            need(isinstance(content, str), "envelope.content")
            need(len(content.encode("utf-8")) <= MAX_FILE_CONTENT_BYTES,
                 "envelope.content.bytes")
    for entry in mutations:
        parts = tuple(entry["path"].split("/"))
        for cut in range(1, len(parts)):
            need(parts[:cut] not in seen, "envelope.path.prefix_collision")
    need(canonical_bytes(top) == raw, "envelope.not_canonical")
    return raw


def run(root, request_bytes):
    """One bounded proposal attempt under an already-active config root."""
    document = parse_request(request_bytes)
    admitted = admitted_paths(document)
    # One task-only, literal config read, without loader mutation or overlays.
    route = resolve_route(read_task_config(root))
    seam = build_client(*route)
    kwargs = build_request_kwargs(seam, document)
    try:
        return validate_envelope(extract_framed(
            response_text(create_completion(seam.client, **kwargs))), admitted)
    finally:
        seam.client.close()


def main(argv=None):
    argv = tuple(sys.argv[1:] if argv is None else argv)
    token = None
    try:
        need(argv in ((), FROZEN_ARGV), "argv.unexpected")
        root = resolve_config_root()
        token = activate_config_root(root)
        envelope = run(root, sys.stdin.buffer.read(MAX_ENVELOPE_BYTES + 1))
        line = FRAMING_PREFIX.encode("ascii") + envelope
        need(len(line) + 1 <= MAX_FRAMED_STDOUT_BYTES, "framing.too_long")
        sys.stdout.buffer.write(line + b"\n")
        sys.stdout.buffer.flush()
        return 0
    except ProposalRefusal as refusal:
        sys.stderr.write("STAGE_A_PROPOSAL_REFUSED " + refusal.code + "\n")
        return 1
    finally:
        if token is not None:
            from hermes_constants import reset_hermes_home_override

            reset_hermes_home_override(token)


if __name__ == "__main__":
    sys.exit(main())
