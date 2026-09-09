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
INSTRUCTIONS = (
    "Answer one bounded Stage-A proposal request. Emit exactly one line: "
    + FRAMING_PREFIX + "<envelope>, where <envelope> is one canonical "
    "printable-ASCII JSON mutation envelope (sorted keys, no spaces, "
    "schema_version " + ENVELOPE_SCHEMA_VERSION + "). Emit nothing else.")


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


def screen_no_credential(node, default=None, depth=0):
    """Refuse credential-shaped material this task root contributes.

    The behavioral loader answers with Hermes' shipped defaults merged
    in, so a subtree still equal to its default is that skeleton and not
    task material: only what this root actually contributes is screened.
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


def read_task_config(root):
    """Screen this task root's config through the accepted loader."""
    from hermes_cli.config import (DEFAULT_CONFIG, get_config_path,
                                   load_config_readonly)

    path = root / "config.yaml"
    need(stat.S_ISREG(_owned_private_stat(path, "config").st_mode),
         "config.not_regular_file")
    # Behavioral truth only: the accepted loader must already be bound to
    # this exact task config path.  An unreadable or unparseable file
    # degrades to the shipped defaults, whose empty route section
    # resolve_route refuses.
    need(Path(get_config_path()) == path, "config.loader_path")
    document = load_config_readonly()
    need(isinstance(document, dict) and document, "config.not_mapping")
    screen_no_credential(document, DEFAULT_CONFIG)
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

    # A local/loopback route makes construction probe the endpoint unless the
    # served window is pinned; require the pin rather than emit an attempt.
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
    return document


def build_agent(model, base_url, provider):
    """Real AIAgent construction on the explicit task route."""
    from run_agent import AIAgent

    return AIAgent(
        model=model, base_url=base_url, api_key=NO_CREDENTIAL_PLACEHOLDER,
        provider=provider, api_mode=API_MODE, quiet_mode=True,
        enabled_toolsets=[], disabled_toolsets=[], skip_context_files=True,
        skip_memory=True, skip_background_review=True,
        save_trajectories=False, max_iterations=1)


def create_completion(client, **kwargs):
    """The single provider seam; offline tests patch exactly this."""
    return client.chat.completions.create(**kwargs)


def build_request_kwargs(agent, document):
    """Build the one request on the accepted transport, with no tools."""
    from agent.transports import get_transport

    need(getattr(agent, "api_mode", "") == API_MODE, "route.api_mode")
    transport = get_transport(API_MODE)
    need(transport is not None, "route.transport")
    model = getattr(agent, "model", "")
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
    """Exactly one framing line; two candidates are ambiguous, never first."""
    framed = [ln for ln in text.split("\n") if ln.startswith(FRAMING_PREFIX)]
    need(framed, "framing.absent")
    need(len(framed) == 1, "framing.ambiguous")
    payload = framed[0][len(FRAMING_PREFIX):]
    if payload.endswith("\r"):
        payload = payload[:-1]
    need(payload, "framing.empty_payload")
    for char in payload:
        need(" " <= char <= "~", "framing.non_printable")
    return payload.encode("ascii")


def _segments(value):
    need(isinstance(value, str) and 0 < len(value) <= MAX_PATH_CHARS,
         "envelope.path")
    need(all(char in PATH_CHARS for char in value), "envelope.path.grammar")
    parts = tuple(value.split("/"))
    need(len(parts) <= MAX_PATH_SEGMENTS, "envelope.path.depth")
    for part in parts:
        need(part not in ("", ".", ".."), "envelope.path.segment")
    need(parts[0] != ".git", "envelope.path.git_dir")
    return parts


def validate_envelope(raw):
    """F2: the exact frozen mutation-envelope grammar, realized here."""
    need(0 < len(raw) <= MAX_ENVELOPE_BYTES, "envelope.bytes_length")
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
    # One behavioral config read: read_task_config proves the accepted
    # loader is bound to this root, screens what the root contributes,
    # and hands back the document this route is resolved from.
    route = resolve_route(read_task_config(root))
    agent = build_agent(*route)
    kwargs = build_request_kwargs(agent, document)
    return validate_envelope(extract_framed(
        response_text(create_completion(agent.client, **kwargs))))


def main(argv=None):
    argv = tuple(sys.argv[1:] if argv is None else argv)
    token = None
    try:
        need(argv in ((), FROZEN_ARGV), "argv.unexpected")
        root = resolve_config_root()
        token = activate_config_root(root)
        envelope = run(root, sys.stdin.buffer.read(MAX_ENVELOPE_BYTES + 1))
        line = FRAMING_PREFIX + envelope.decode("ascii")
        need(len(line) + 1 <= MAX_STDOUT_BYTES, "framing.too_long")
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
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
