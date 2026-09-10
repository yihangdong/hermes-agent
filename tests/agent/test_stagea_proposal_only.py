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
def live(tmp_path, monkeypatch):
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


def test_task_config_must_be_non_secret_and_pinned(tmp_path):
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
        tmp_path, monkeypatch):
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


def test_real_construction_is_offline_and_route_consistent(live):
    token = sp.activate_config_root(live.root)
    try:
        before = live.observer.inside
        agent = sp.build_agent(MODEL, BASE_URL, PROVIDER)
    finally:
        reset_hermes_home_override(token)
    # Positive control: construction really opened files in the task root.
    assert live.observer.inside > before
    # Negative control: zero egress attempts, on a counter proven live below.
    assert live.observer.connects == []
    with pytest.raises(OSError):
        socket.socket().connect(("127.0.0.1", 9))
    assert live.observer.connects
    assert str(agent.logs_dir).startswith(str(live.root))
    assert agent.model == MODEL
    assert agent.base_url == BASE_URL
    assert agent.api_mode == sp.API_MODE
    assert agent.api_key == sp.NO_CREDENTIAL_PLACEHOLDER
    assert agent._fallback_chain == []


def test_one_framed_proposal_end_to_end(live, monkeypatch, capsys):
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
    assert not {"tools", "tool_choice", "functions"} & set(kwargs)
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
    exempt = dict(document, admitted_write_set=[{"p": "src/session_id.py"}])
    assert sp.parse_request(
        sp.canonical_bytes(exempt))["admitted_write_set"] == [
            {"p": "src/session_id.py"}]


def test_framing_requires_exactly_one_line():
    good = sp.FRAMING_PREFIX + ENVELOPE.decode("ascii")
    assert sp.extract_framed("noise\n" + good + "\n") == ENVELOPE
    for text, code in (("nothing", "framing.absent"),
                       (good + "\n" + good, "framing.ambiguous")):
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
])
def test_envelope_grammar_is_fail_closed(raw, code):
    with pytest.raises(sp.ProposalRefusal) as refusal:
        sp.validate_envelope(raw)
    assert refusal.value.code == code


def test_canonical_envelope_round_trips():
    assert sp.validate_envelope(ENVELOPE) == ENVELOPE


def test_instructions_state_the_enforced_grammar():
    """The one system prompt must carry what the parser really enforces."""
    text = sp.INSTRUCTIONS
    assert "exactly one line: " + sp.FRAMING_PREFIX in text
    assert sp.ENVELOPE_SCHEMA_VERSION in text
    for op, keys in sp.OP_KEYS.items():
        assert "op=" + op in text
        for key in keys - {"op"}:
            assert key in text
    for bound in (sp.MAX_ENVELOPE_BYTES, sp.MAX_ENVELOPE_FILES,
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
