"""Offline tests for the Stage-A proposal-only program (issue #198).

No provider, credential, real home or network is touched: the task config
root is a tmp_path fixture, the provider seam is bound in-process, and every
connect attempt is counted and refused.
"""

import builtins
import io
import json
import os
import pathlib
import socket
import sys

import pytest
import yaml

from agent import stagea_proposal_only as sp
from hermes_constants import get_hermes_home, reset_hermes_home_override

MODEL = "stage-a-proposal-fixture"
BASE_URL = "http://127.0.0.1:8791/v1"
PROVIDER = "stage-a-relay"
CONTEXT_PIN = 262144
ROUTE = {"default": MODEL, "base_url": BASE_URL, "provider": PROVIDER,
         "context_length": CONTEXT_PIN, "ollama_num_ctx": CONTEXT_PIN}
CREATE = {"op": "create", "path": "docs/stage_a.md", "content": "x"}


def _envelope(mutations, schema=None):
    return sp.canonical_bytes({
        "schema_version": schema or sp.ENVELOPE_SCHEMA_VERSION,
        "mutations": mutations})


ENVELOPE = _envelope([CREATE])
REQUEST = sp.canonical_bytes({
    "schema_version": sp.REQUEST_SCHEMA_VERSION, "action_id": "a1",
    "decision_id": "d1", "action_intent": "DISPATCH_AUTHOR",
    "kernel_version": "k1", "work_item_id": "w1", "contract_ref": "c1",
    "work_contract_fingerprint": "f1",
    "canonical_repo": "yihangdong/dyhano-ai-organization",
    "canonical_branch_ref": "refs/heads/main",
    "expected_main_sha": "0" * 40,
    "admitted_write_set": ["docs/stage_a.md"],
    "envelope_schema_version": sp.ENVELOPE_SCHEMA_VERSION,
    "max_envelope_bytes": sp.MAX_ENVELOPE_BYTES})
#: The normalized write set the corrected envelope boundary is judged against.
REQUEST_ADMITTED = sp.admitted_paths(json.loads(REQUEST))
#: Extra members the grammar cases legitimately mutate, so a grammar refusal is
#: still observed as a grammar refusal rather than masked by admission.
GRAMMAR_ADMITTED = REQUEST_ADMITTED | frozenset({("docs",), ("a",)})
#: The controller's own bound on one child attempt (hermes_oneshot.py).
CONTROLLER_ONESHOT_SECONDS = 900


class _ForbiddenModule:
    """A sys.modules stand-in that fails if the proposal path reaches it."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, attribute):
        raise AssertionError(
            "proposal-only path touched " + self._name + "." + attribute)


def _forbid_discovery(*args, **kwargs):
    raise AssertionError("plugin discovery reached the proposal-only path")


def _bind_managed(monkeypatch, directory, document):
    """Bind the accepted managed-scope probes to an exact synthetic state."""
    import hermes_cli.managed_scope as managed

    monkeypatch.setattr(managed, "get_managed_dir", lambda: directory)
    monkeypatch.setattr(managed, "load_managed_config", lambda: document)
    return managed


def _sized_envelope(total_bytes, paths):
    """A canonical envelope of exactly *total_bytes* bytes over *paths*."""
    mutations = [{"op": "create", "path": path, "content": ""}
                 for path in paths]
    padding = total_bytes - len(_envelope(mutations))
    assert 0 <= padding <= len(paths) * sp.MAX_FILE_CONTENT_BYTES
    for mutation in mutations:
        chunk = min(padding, sp.MAX_FILE_CONTENT_BYTES)
        mutation["content"] = "x" * chunk
        padding -= chunk
    assert padding == 0
    return _envelope(mutations)


class Stub:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def _response(text):
    return Stub(choices=[Stub(message=Stub(content=text, tool_calls=None))])


def make_root(tmp_path, document=None):
    root = tmp_path / sp.CONFIG_ROOT_NAME
    root.mkdir()
    root.chmod(0o700)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(
        {"model": dict(ROUTE)} if document is None else document),
        encoding="utf-8")
    path.chmod(0o600)
    return root


class Observer:
    """Counts real construction file opens and every egress attempt."""

    def __init__(self, monkeypatch, root):
        self.root = str(root)
        self.inside = 0
        self.outside = 0
        self.connects = []
        real_open = builtins.open
        real_path_open = pathlib.Path.open

        def count(name):
            if self.root in str(name):
                self.inside += 1
            else:
                self.outside += 1

        def fake_open(file, *args, **kwargs):
            count(file)
            return real_open(file, *args, **kwargs)

        def fake_path_open(this, *args, **kwargs):
            count(this)
            return real_path_open(this, *args, **kwargs)

        def fake_connect(this, address):
            self.connects.append(address)
            raise OSError("stage-a offline test: egress refused")

        monkeypatch.setattr(builtins, "open", fake_open)
        monkeypatch.setattr(io, "open", fake_open)
        monkeypatch.setattr(pathlib.Path, "open", fake_path_open)
        monkeypatch.setattr(socket.socket, "connect", fake_connect)
        monkeypatch.setattr(socket.socket, "connect_ex", fake_connect)


@pytest.fixture
def no_managed_overlay(monkeypatch):
    """No host managed scope: the task root is the only configuration."""
    return _bind_managed(monkeypatch, None, {})


@pytest.fixture
def live(tmp_path, monkeypatch, no_managed_overlay):
    """Real construction under the task root with the provider seam bound."""
    root = make_root(tmp_path)
    monkeypatch.chdir(tmp_path)
    observer = Observer(monkeypatch, root)
    calls = []

    def seam(client, **kwargs):
        calls.append((client, kwargs, get_hermes_home() == root))
        return _response(sp.FRAMING_PREFIX + ENVELOPE.decode("ascii") + "\n")

    monkeypatch.setattr(sp, "create_completion", seam)
    return Stub(root=root, observer=observer, calls=calls)


def test_config_root_fails_closed(tmp_path):
    with pytest.raises(sp.ProposalRefusal) as absent:
        sp.resolve_config_root(tmp_path)
    assert absent.value.code == "root.absent"
    (tmp_path / "elsewhere").mkdir()
    candidate = tmp_path / sp.CONFIG_ROOT_NAME
    candidate.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(sp.ProposalRefusal) as escaped:
        sp.resolve_config_root(tmp_path)
    assert escaped.value.code == "root.escape"
    candidate.unlink()
    candidate.write_text("", encoding="utf-8")
    with pytest.raises(sp.ProposalRefusal) as plain:
        sp.resolve_config_root(tmp_path)
    assert plain.value.code == "root.not_directory"
    candidate.unlink()
    candidate.mkdir()
    candidate.chmod(0o777)
    with pytest.raises(sp.ProposalRefusal) as loose:
        sp.resolve_config_root(tmp_path)
    assert loose.value.code == "root.writable_by_others"


def test_override_is_task_scoped_and_restored(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    monkeypatch.chdir(tmp_path)
    env_home = os.environ.get("HERMES_HOME")
    process_home = get_hermes_home()
    token = sp.activate_config_root(sp.resolve_config_root())
    try:
        assert get_hermes_home() == root
        assert os.environ.get("HERMES_HOME") == env_home
    finally:
        reset_hermes_home_override(token)
    assert get_hermes_home() == process_home


def test_task_config_must_be_non_secret_and_pinned(
        tmp_path, no_managed_overlay):
    root = make_root(tmp_path, {"model": dict(ROUTE, api_key="x")})
    token = sp.activate_config_root(root)
    try:
        with pytest.raises(sp.ProposalRefusal) as secret:
            sp.read_task_config(root)
    finally:
        reset_hermes_home_override(token)
    assert secret.value.code == "config.credential_key"
    unpinned = dict(ROUTE)
    unpinned.pop("context_length")
    with pytest.raises(sp.ProposalRefusal) as pin:
        sp.resolve_route({"model": unpinned})
    assert pin.value.code == "route.context_pin"
    unprobed = dict(ROUTE)
    unprobed.pop("ollama_num_ctx")
    with pytest.raises(sp.ProposalRefusal) as probe:
        sp.resolve_route({"model": unprobed})
    assert probe.value.code == "route.local_probe_pin"
    assert sp.resolve_route({"model": dict(ROUTE)}) == (
        MODEL, BASE_URL, PROVIDER)


def test_task_config_is_read_through_the_behavioral_loader(
        tmp_path, monkeypatch, no_managed_overlay):
    """Screening/route reads use the loader, never a raw primitive."""
    import hermes_cli.config as hermes_config

    root = make_root(tmp_path)
    seen = []
    accepted = hermes_config.load_config_readonly

    def refuse(*args, **kwargs):
        raise AssertionError("raw behavioral config read")

    def recording():
        seen.append(hermes_config.get_config_path())
        return accepted()

    for raw in ("read_user_config_raw", "read_raw_config",
                "read_raw_config_readonly"):
        monkeypatch.setattr(hermes_config, raw, refuse)
    monkeypatch.setattr(hermes_config, "load_config_readonly", recording)
    token = sp.activate_config_root(root)
    try:
        document = sp.read_task_config(root)
    finally:
        reset_hermes_home_override(token)
    assert seen == [root / "config.yaml"]
    assert document["model"] == dict(ROUTE)


def test_run_refuses_when_loader_is_not_bound_to_task_root(tmp_path):
    """Without the active override the loader is not this task's config."""
    root = make_root(tmp_path)
    with pytest.raises(sp.ProposalRefusal) as refusal:
        sp.run(root, REQUEST)
    assert refusal.value.code == "config.loader_path"


def test_client_construction_is_offline_agentless_and_plugin_free(
        live, monkeypatch):
    """F2/F7: no generic agent, no plugin discovery, no constructor egress."""
    import hermes_cli.plugins as hermes_plugins

    # Any `from run_agent import AIAgent` on this path is a hard failure, and
    # so is any plugin discovery -- the two constructor side effects the
    # findings name (agent_init.py:852-866 prewarm, 1607-1615 discovery).
    monkeypatch.setattr(hermes_plugins, "discover_plugins", _forbid_discovery)
    monkeypatch.setitem(sys.modules, "run_agent",
                        _ForbiddenModule("run_agent"))
    token = sp.activate_config_root(live.root)
    try:
        seam = sp.build_client(MODEL, BASE_URL, PROVIDER)
    finally:
        reset_hermes_home_override(token)
    # Negative control: zero egress attempts, on counters proven live here.
    assert live.observer.connects == []
    with pytest.raises(OSError):
        socket.socket().connect(("127.0.0.1", 9))
    assert live.observer.connects
    inside_before = live.observer.inside
    (live.root / "config.yaml").read_text(encoding="utf-8")
    assert live.observer.inside > inside_before
    assert seam.model == MODEL
    assert seam.base_url == BASE_URL
    assert seam.provider == PROVIDER
    assert seam.api_mode == sp.API_MODE
    assert seam.api_key == sp.NO_CREDENTIAL_PLACEHOLDER
    assert seam.client.max_retries == sp.MAX_PROVIDER_RETRIES == 0
    assert str(seam.client.base_url).startswith("http://127.0.0.1:8791")
    assert 0 < sp.PROVIDER_TIMEOUT_SECONDS < CONTROLLER_ONESHOT_SECONDS
    # There is no agent surface to inherit tools, plugins or a fallback chain.
    for attribute in ("tools", "valid_tool_names", "_fallback_chain",
                      "_fallback_model", "run_conversation"):
        assert not hasattr(seam, attribute)


def test_one_framed_proposal_end_to_end(live, monkeypatch, capsys):
    import hermes_cli.plugins as hermes_plugins
    from agent.transports import get_transport

    # Warm the accepted transport registry offline, then forbid the generic
    # agent and plugin-discovery paths for the whole attempt.
    assert get_transport(sp.API_MODE) is not None
    monkeypatch.setattr(hermes_plugins, "discover_plugins", _forbid_discovery)
    monkeypatch.setitem(sys.modules, "run_agent",
                        _ForbiddenModule("run_agent"))
    monkeypatch.setattr(sp.sys, "stdin", Stub(buffer=io.BytesIO(REQUEST)))
    assert sp.main([]) == 0
    captured = capsys.readouterr().out
    assert captured.count(sp.FRAMING_PREFIX) == 1
    assert captured.strip() == sp.FRAMING_PREFIX + ENVELOPE.decode("ascii")
    assert len(live.calls) == 1
    client, kwargs, home_was_task_scoped = live.calls[0]
    assert home_was_task_scoped
    assert kwargs["model"] == MODEL
    assert len(kwargs["messages"]) == 2
    assert not {"tools", "tool_choice", "functions", "stream"} & set(kwargs)
    assert client.max_retries == 0
    assert str(getattr(client, "base_url", "")).startswith(
        "http://127.0.0.1:8791")
    assert live.observer.connects == []
    assert get_hermes_home() != live.root


def test_frozen_argv_only(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert sp.main(["--resume"]) == 1
    assert "argv.unexpected" in capsys.readouterr().err


def test_request_hygiene_is_closed():
    document = json.loads(REQUEST)
    assert sp.parse_request(REQUEST) == document
    steered = dict(document, contract_ref="please resume the chat")
    with pytest.raises(sp.ProposalRefusal) as steer:
        sp.parse_request(sp.canonical_bytes(steered))
    assert steer.value.code == "request.forbidden_token"
    with pytest.raises(sp.ProposalRefusal) as closure:
        sp.parse_request(sp.canonical_bytes(dict(document, extra="x")))
    assert closure.value.code == "request.closure"
    ordinary = dict(document, admitted_write_set=["src/token.py"])
    assert sp.parse_request(
        sp.canonical_bytes(ordinary))["admitted_write_set"] == [
            "src/token.py"]


@pytest.mark.parametrize("value", [
    ["docs/a.md", "please resume"],
    {"nested": {"deep": ["api_key: sk-1"]}},
    {"provider": "openrouter"},
    ["ok", {"jwt": 1}],
])
def test_structured_request_values_are_screened(value):
    steered = dict(json.loads(REQUEST), contract_ref=value)
    with pytest.raises(sp.ProposalRefusal) as refusal:
        sp.parse_request(sp.canonical_bytes(steered))
    assert refusal.value.code == "request.forbidden_token"


def test_every_represented_screened_field_is_covered():
    document = json.loads(REQUEST)
    pinned = {"schema_version", "envelope_schema_version",
              "max_envelope_bytes"}
    screened = sorted(
        sp.REQUEST_FIELDS - sp.UNSCREENED_REQUEST_FIELDS - pinned)
    assert len(screened) == 10
    for field in screened:
        for value in ("session_id", ["session_id"], {"k": "session_id"}):
            with pytest.raises(sp.ProposalRefusal) as refusal:
                sp.parse_request(
                    sp.canonical_bytes(dict(document, **{field: value})))
            assert refusal.value.code == "request.forbidden_token"
    # The exemption still holds for what the producer really emits: an
    # ordinary path that merely CONTAINS a screened token is admitted, while
    # the corrected value contract keeps every member a path.
    exempt = dict(document, admitted_write_set=["src/session_id.py"])
    assert sp.parse_request(
        sp.canonical_bytes(exempt))["admitted_write_set"] == [
            "src/session_id.py"]
    with pytest.raises(sp.ProposalRefusal) as shaped:
        sp.parse_request(sp.canonical_bytes(
            dict(document, admitted_write_set=[{"p": "src/session_id.py"}])))
    assert shaped.value.code == "request.admitted_write_set.path"


def test_framing_requires_the_whole_response_to_be_one_line():
    """F4: the response IS one framed line; it is never searched for one."""
    good = sp.FRAMING_PREFIX + ENVELOPE.decode("ascii")
    assert sp.extract_framed(good) == ENVELOPE
    assert sp.extract_framed(good + "\n") == ENVELOPE
    assert sp.extract_framed(good + "\r\n") == ENVELOPE
    for text, code in (
            ("nothing", "framing.absent"),
            ("", "framing.absent"),
            (sp.FRAMING_PREFIX, "framing.empty_payload"),
            (good + "\n" + good, "framing.ambiguous"),
            ("prose\n" + good, "framing.surrounding_text"),
            (good + "\nmore prose", "framing.surrounding_text"),
            ("prose\n" + good + "\nmore prose", "framing.surrounding_text")):
        with pytest.raises(sp.ProposalRefusal) as refusal:
            sp.extract_framed(text)
        assert refusal.value.code == code


@pytest.mark.parametrize("raw, code", [
    (_envelope([CREATE], "other-v1"), "envelope.schema_version"),
    (_envelope([]), "envelope.mutations"),
    (_envelope([dict(CREATE, op="rename")]), "envelope.mutation.op"),
    (_envelope([dict(CREATE, extra=1)]), "envelope.mutation.op"),
    (_envelope([dict(CREATE, path="a b")]), "envelope.path.grammar"),
    (_envelope([dict(CREATE, path="../x")]), "envelope.path.segment"),
    (_envelope([dict(CREATE, path=".git/x")]), "envelope.path.git_dir"),
    (_envelope([CREATE, dict(CREATE, content="y")]),
     "envelope.path.duplicate"),
    (_envelope([CREATE, dict(CREATE, path="docs")]),
     "envelope.path.prefix_collision"),
    (_envelope([{"op": "replace", "path": "a", "base_blob_sha": "z" * 40,
                 "content": "x"}]), "envelope.base_blob_sha"),
    (b'{"schema_version": "x"}', "envelope.closure"),
    (json.dumps({"schema_version": sp.ENVELOPE_SCHEMA_VERSION,
                 "mutations": [CREATE]}).encode("ascii"),
     "envelope.not_canonical"),
    (_envelope([dict(CREATE, path="src/elsewhere.py")]),
     "envelope.path.not_admitted"),
    (_envelope([CREATE, dict(CREATE, path="src/elsewhere.py")]),
     "envelope.path.not_admitted"),
])
def test_envelope_grammar_is_fail_closed(raw, code):
    with pytest.raises(sp.ProposalRefusal) as refusal:
        sp.validate_envelope(raw, GRAMMAR_ADMITTED)
    assert refusal.value.code == code


def test_canonical_envelope_round_trips():
    assert sp.validate_envelope(ENVELOPE, REQUEST_ADMITTED) == ENVELOPE


def test_validate_envelope_cannot_be_called_without_the_request_half():
    """F1: admission is a required argument, never an optional add-on."""
    with pytest.raises(TypeError):
        sp.validate_envelope(ENVELOPE)


def test_instructions_state_the_enforced_grammar():
    """The one system prompt must carry what the parser really enforces."""
    text = sp.INSTRUCTIONS
    assert "exactly one line: " + sp.FRAMING_PREFIX in text
    assert sp.ENVELOPE_SCHEMA_VERSION in text
    for op, keys in sp.OP_KEYS.items():
        assert "op=" + op in text
        for key in keys - {"op"}:
            assert key in text
    for bound in (sp.MAX_EMITTED_ENVELOPE_BYTES, sp.MAX_ENVELOPE_FILES,
                  sp.MAX_FILE_CONTENT_BYTES, sp.MAX_PATH_CHARS,
                  sp.MAX_PATH_SEGMENTS):
        assert str(bound) in text
    for fragment in ("admitted_write_set", "40 lowercase hex",
                     "not share a path", "ancestor (directory prefix)",
                     "printable-ASCII", "sorted", "no whitespace",
                     "no markdown", "no code fence", "no tool call"):
        assert fragment in text
    # The superseded PR #204 model-level REFUSED protocol is not inherited.
    assert "REFUSED" not in text


def test_accepted_producer_request_shape_is_admitted():
    """The exact projection dyhano_stage_a/proposal.py emits must pass."""
    document = sp.parse_request(REQUEST)
    assert document["action_intent"] == sp.ADMITTED_ACTION_INTENT
    assert set(document) == set(sp.REQUEST_FIELDS)
    assert sp.admitted_paths(document) == frozenset({("docs", "stage_a.md")})


@pytest.mark.parametrize("field, value, code", [
    ("action_intent", "MERGE", "request.action_intent"),
    ("action_intent", "dispatch_author", "request.action_intent"),
    ("action_intent", 1, "request.action_intent"),
    ("expected_main_sha", 0, "request.expected_main_sha"),
    ("expected_main_sha", "0" * 41, "request.expected_main_sha"),
    ("expected_main_sha", "0" * 39, "request.expected_main_sha.grammar"),
    ("expected_main_sha", "Z" * 40, "request.expected_main_sha.grammar"),
    ("action_id", {"a": "b"}, "request.action_id"),
    ("action_id", "", "request.action_id"),
    ("action_id", "-a1", "request.action_id.grammar"),
    ("action_id", "a1-", "request.action_id.grammar"),
    ("action_id", "a/1", "request.action_id.grammar"),
    ("decision_id", ["d1"], "request.decision_id"),
    ("kernel_version", "k" * 33, "request.kernel_version"),
    ("work_item_id", 12, "request.work_item_id"),
    ("contract_ref", "c" * 201, "request.contract_ref"),
    ("work_contract_fingerprint", None,
     "request.work_contract_fingerprint"),
    ("canonical_repo", "dyhano-ai-organization",
     "request.canonical_repo.grammar"),
    ("canonical_repo", "a/b/c", "request.canonical_repo.grammar"),
    ("canonical_repo", "a/", "request.canonical_repo.grammar"),
    ("canonical_branch_ref", "refs//main",
     "request.canonical_branch_ref.grammar"),
    ("canonical_branch_ref", "refs/heads/..",
     "request.canonical_branch_ref.grammar"),
    ("canonical_branch_ref", "refs/heads/main.lock",
     "request.canonical_branch_ref.grammar"),
    ("canonical_branch_ref", "refs/heads/.hidden",
     "request.canonical_branch_ref.grammar"),
])
def test_request_value_contract_is_enforced_before_dispatch(
        field, value, code):
    """F6: key closure is not the contract; bad values never reach a model."""
    document = dict(json.loads(REQUEST), **{field: value})
    with pytest.raises(sp.ProposalRefusal) as refusal:
        sp.parse_request(sp.canonical_bytes(document))
    assert refusal.value.code == code


@pytest.mark.parametrize("value, code", [
    ([], "request.admitted_write_set"),
    ("docs/stage_a.md", "request.admitted_write_set"),
    ({"docs/stage_a.md": True}, "request.admitted_write_set"),
    ([1], "request.admitted_write_set.path"),
    (["x" * 201], "request.admitted_write_set.path"),
    (["docs/a b.md"], "request.admitted_write_set.path.grammar"),
    (["docs/../etc"], "request.admitted_write_set.path.segment"),
    ([".git/config"], "request.admitted_write_set.path.git_dir"),
    (["a/" * 16 + "b"], "request.admitted_write_set.path.depth"),
    (["docs/stage_a.md", "docs/stage_a.md"],
     "request.admitted_write_set.duplicate"),
])
def test_admitted_write_set_is_validated_before_dispatch(value, code):
    """F6: the write set is a bounded list of normalizable paths."""
    document = dict(json.loads(REQUEST), admitted_write_set=value)
    with pytest.raises(sp.ProposalRefusal) as refusal:
        sp.parse_request(sp.canonical_bytes(document))
    assert refusal.value.code == code


def test_out_of_admitted_set_mutation_is_refused_before_stdout(
        live, monkeypatch, capsys):
    """F1: a syntactically valid mutation outside the write set never prints."""
    outside = _envelope([{"op": "create", "path": "src/evil.py",
                          "content": "x"}])
    # Same shape as the accepted proposal, only the path is a non-member.
    assert sp.validate_envelope(
        outside, frozenset({("src", "evil.py")})) == outside
    monkeypatch.setattr(
        sp, "create_completion", lambda client, **kwargs: _response(
            sp.FRAMING_PREFIX + outside.decode("ascii") + "\n"))
    monkeypatch.setattr(sp.sys, "stdin", Stub(buffer=io.BytesIO(REQUEST)))
    assert sp.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "envelope.path.not_admitted" in captured.err


@pytest.mark.parametrize("patch, code", [
    ({"base_url": "https://relay.example/v1?token=real-secret"},
     "config.credential_url.query"),
    ({"base_url": "https://user:pw@relay.example/v1"},
     "config.credential_url.userinfo"),
    ({"base_url": "https://relay.example/v1#api_key=real-secret"},
     "config.credential_url.fragment"),
    ({"default": "Bearer sk-live-0"}, "config.credential_value"),
    ({"provider": "relay?api_key=x"}, "config.credential_value"),
])
def test_credential_bearing_config_values_are_refused(
        tmp_path, no_managed_overlay, patch, code):
    """F3: consumed VALUES must be proven non-secret, not just key names."""
    root = make_root(tmp_path, {"model": dict(ROUTE, **patch)})
    token = sp.activate_config_root(root)
    try:
        with pytest.raises(sp.ProposalRefusal) as refusal:
            sp.read_task_config(root)
    finally:
        reset_hermes_home_override(token)
    assert refusal.value.code == code


def test_ordinary_route_values_are_not_mistaken_for_credentials(
        tmp_path, no_managed_overlay):
    """The value screen must not refuse a legitimate provider route."""
    ordinary = {"default": "tokenhub/keystone-v2", "provider": "tokenhub",
                "base_url": "https://tokenhub.example.com/v1",
                "context_length": CONTEXT_PIN}
    root = make_root(tmp_path, {"model": ordinary})
    token = sp.activate_config_root(root)
    try:
        document = sp.read_task_config(root)
    finally:
        reset_hermes_home_override(token)
    assert sp.resolve_route(document) == (
        "tokenhub/keystone-v2", "https://tokenhub.example.com/v1", "tokenhub")


@pytest.mark.parametrize("directory, document, code", [
    ("/etc/hermes", {}, "config.managed_dir"),
    (None, {"model": {"default": "managed/other"}}, "config.managed_overlay"),
])
def test_managed_overlay_fails_closed(
        tmp_path, monkeypatch, directory, document, code):
    """F8: an external managed layer never silently wins this route."""
    root = make_root(tmp_path)
    _bind_managed(monkeypatch, directory, document)
    token = sp.activate_config_root(root)
    try:
        with pytest.raises(sp.ProposalRefusal) as refusal:
            sp.read_task_config(root)
    finally:
        reset_hermes_home_override(token)
    assert refusal.value.code == code


def test_managed_overlay_refusal_precedes_any_client_or_request(
        tmp_path, monkeypatch, capsys):
    """F8: the refusal lands before a client exists or a request is built."""
    make_root(tmp_path)
    monkeypatch.chdir(tmp_path)
    _bind_managed(monkeypatch, None,
                  {"model": {"base_url": "https://managed.example/v1"}})
    reached = []
    monkeypatch.setattr(sp, "build_client",
                        lambda *route: reached.append(route))
    monkeypatch.setattr(sp, "create_completion",
                        lambda client, **kwargs: reached.append(kwargs))
    monkeypatch.setattr(sp.sys, "stdin", Stub(buffer=io.BytesIO(REQUEST)))
    assert sp.main([]) == 1
    assert reached == []
    assert "config.managed_overlay" in capsys.readouterr().err


def test_envelope_and_stdout_budgets_are_internally_consistent():
    """F5: the advertised envelope limit is exactly what can be framed."""
    assert sp.MAX_FRAMED_STDOUT_BYTES < sp.MAX_STDOUT_BYTES
    assert (len(sp.FRAMING_PREFIX) + sp.MAX_EMITTED_ENVELOPE_BYTES + 1
            == sp.MAX_FRAMED_STDOUT_BYTES)
    assert sp.MAX_EMITTED_ENVELOPE_BYTES < sp.MAX_ENVELOPE_BYTES
    assert str(sp.MAX_EMITTED_ENVELOPE_BYTES) in sp.INSTRUCTIONS


def test_exact_maximum_advertised_envelope_is_framed_and_printed(
        live, monkeypatch, capsys):
    """F5: the largest advertised envelope survives validation AND stdout."""
    paths = tuple("docs/big%d.md" % index for index in range(6))
    admitted = frozenset(tuple(path.split("/")) for path in paths)
    biggest = _sized_envelope(sp.MAX_EMITTED_ENVELOPE_BYTES, paths)
    assert len(biggest) == sp.MAX_EMITTED_ENVELOPE_BYTES
    assert sp.validate_envelope(biggest, admitted) == biggest
    request = sp.canonical_bytes(
        dict(json.loads(REQUEST), admitted_write_set=sorted(paths)))
    monkeypatch.setattr(
        sp, "create_completion", lambda client, **kwargs: _response(
            sp.FRAMING_PREFIX + biggest.decode("ascii") + "\n"))
    monkeypatch.setattr(sp.sys, "stdin", Stub(buffer=io.BytesIO(request)))
    assert sp.main([]) == 0
    printed = capsys.readouterr().out.encode("ascii")
    # Exactly the framing budget, and strictly under the controller's cap --
    # which marks a stream truncated the moment it REACHES that cap.
    assert len(printed) == sp.MAX_FRAMED_STDOUT_BYTES < sp.MAX_STDOUT_BYTES
    # One byte more is refused here rather than truncated downstream.
    with pytest.raises(sp.ProposalRefusal) as refusal:
        sp.validate_envelope(
            _sized_envelope(sp.MAX_EMITTED_ENVELOPE_BYTES + 1, paths),
            admitted)
    assert refusal.value.code == "envelope.bytes_length"
