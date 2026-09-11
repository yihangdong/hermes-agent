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
import subprocess
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


_COLD_CONFIG_CHECK = r'''
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

repo, task, scenario, request_json, envelope_json = sys.argv[1:]
sys.path.insert(0, repo)
task = Path(task)
root = task / '.stagea-hermes-home'
forbidden = ('agent.model_metadata', 'hermes_cli.config', 'providers',
             'hermes_cli.plugins', 'run_agent')
def is_forbidden(name):
    return any(name == prefix or name.startswith(prefix + '.')
               for prefix in forbidden)
assert not any(is_forbidden(name) for name in sys.modules)
assert 'stagea_config_owner' not in sys.modules
class ColdImportFence:
    def find_spec(self, fullname, path=None, target=None):
        assert not is_forbidden(fullname), 'generic cold import: ' + fullname
sys.meta_path.insert(0, ColdImportFence())

def snapshot():
    return {p.relative_to(task).as_posix():
            (p.stat().st_mode, p.read_bytes() if p.is_file() else None)
            for p in task.rglob('*')}
before = snapshot()
from agent import stagea_proposal_only as sp
from hermes_constants import reset_hermes_home_override
assert 'stagea_config_owner' not in sys.modules
from hermes_cli import managed_scope
def refuse_content(*args, **kwargs):
    raise AssertionError('managed content must not be loaded')
managed_scope.load_managed_config = refuse_content
managed_scope.load_managed_env = refuse_content

class LiteralEnvironment(dict):
    def __getitem__(self, key):
        assert key != 'STAGEA_INERT', 'literal field read inherited environment'
        return super().__getitem__(key)
    def get(self, key, default=None):
        assert key != 'STAGEA_INERT', 'literal field read inherited environment'
        return super().get(key, default)
os.environ = LiteralEnvironment(os.environ)
checking = [True]
def refuse_managed_reads(event, args):
    assert event not in ('socket.connect', 'socket.getaddrinfo'), 'network attempt'
    if event == 'open' and checking[0] and isinstance(args[0], (str, bytes)):
        path = os.fsdecode(args[0])
        assert not path.startswith(str(task / 'managed') + os.sep)
        if scenario == 'managed':
            assert path != str(root / 'config.yaml'), 'content read before managed refusal'
sys.addaudithook(refuse_managed_reads)
expected = {'managed': 'config.managed_dir',
            'environment': 'config.environment_syntax',
            'malformed': 'config.parse'}
token = sp.activate_config_root(sp.resolve_config_root(task))
try:
    try:
        document = sp.read_task_config(root)
    except sp.ProposalRefusal as exc:
        assert scenario in expected and exc.code == expected[scenario], exc.code
    else:
        assert scenario == 'valid'
        assert document['model'] == {
            'default': 'stage-a-proposal-fixture',
            'base_url': 'http://127.0.0.1:8791/v1',
            'provider': 'stage-a-relay', 'context_length': 262144,
            'ollama_num_ctx': 262144}
        assert sp.resolve_route(document) == (
            'stage-a-proposal-fixture', 'http://127.0.0.1:8791/v1',
            'stage-a-relay')

        # Exercise the real entrypoint, client and request preparation cold.
        # Only the single completion is inert; no registry is prewarmed.
        calls = []
        def inert_completion(client, **kwargs):
            from hermes_constants import get_hermes_home
            assert Path(get_hermes_home()) == root
            assert not any(is_forbidden(name) for name in sys.modules)
            assert snapshot() == before
            assert client.max_retries == 0
            assert client.api_key == sp.NO_CREDENTIAL_PLACEHOLDER
            assert str(client.base_url).rstrip('/') == document['model']['base_url']
            assert kwargs['model'] == document['model']['default']
            assert kwargs['max_tokens'] == sp.MAX_COMPLETION_TOKENS
            assert kwargs['messages'] == [
                {'role': 'system', 'content': sp.INSTRUCTIONS},
                {'role': 'user', 'content': request_json}]
            assert not {'tools', 'tool_choice', 'functions', 'function_call',
                        'stream'} & set(kwargs)
            calls.append(client)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=sp.FRAMING_PREFIX + envelope_json, tool_calls=None))])
        sp.create_completion = inert_completion
        stdout = io.BytesIO()
        original_stdin, original_stdout = sys.stdin, sys.stdout
        try:
            sys.stdin = SimpleNamespace(buffer=io.BytesIO(request_json.encode('ascii')))
            sys.stdout = SimpleNamespace(buffer=stdout)
            assert sp.main(sp.FROZEN_ARGV) == 0
        finally:
            sys.stdin, sys.stdout = original_stdin, original_stdout
        assert len(calls) == 1 and calls[0].is_closed()
        assert stdout.getvalue() == (
            sp.FRAMING_PREFIX + envelope_json + '\n').encode('ascii')
finally:
    reset_hermes_home_override(token)
    checking[0] = False
assert not any(is_forbidden(name) for name in sys.modules)
if scenario == 'managed':
    assert 'stagea_config_owner' not in sys.modules
else:
    assert 'stagea_config_owner' in sys.modules
assert snapshot() == before
print(json.dumps({'cold_path': 'PASS', 'scenario': scenario}))
'''


@pytest.mark.parametrize("scenario", ["valid", "managed", "environment", "malformed"])
def test_task_config_cold_import_boundary_in_fresh_interpreter(tmp_path, scenario):
    """Cold config, route and full attempt preserve the task-home snapshot."""
    document = {"model": dict(ROUTE)}
    if scenario == "environment":
        document["model"]["default"] = "${STAGEA_INERT}"
    root = make_root(tmp_path, document)
    if scenario == "malformed":
        (root / "config.yaml").write_bytes(b"model: [")
    managed = tmp_path / "managed"
    if scenario == "managed":
        managed.mkdir()
    environment = {
        "PATH": os.defpath, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path),
        "HERMES_HOME": str(root), "HERMES_MANAGED_DIR": str(managed),
        "PYTEST_CURRENT_TEST": "Stage-A synthetic cold import",
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
        "STAGEA_INERT": "synthetic-unconsumed-value",
    }
    if "SYSTEMROOT" in os.environ:
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", _COLD_CONFIG_CHECK,
         str(pathlib.Path(__file__).resolve().parents[2]), str(tmp_path), scenario,
         REQUEST.decode("ascii"), ENVELOPE.decode("ascii")],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"cold_path": "PASS", "scenario": scenario}


def test_dedicated_owner_retains_exact_bytes_and_literal_object_contract():
    from stagea_config_owner import parse_stagea_task_config_bytes

    assert parse_stagea_task_config_bytes(b"value: ${INERT}\n") == {"value": "${INERT}"}
    assert parse_stagea_task_config_bytes(b"") is None
    assert parse_stagea_task_config_bytes(b"[1, 2]") == [1, 2]
    for value in ("value: 1", bytearray(b"value: 1"), None):
        with pytest.raises(TypeError):
            parse_stagea_task_config_bytes(value)
    with pytest.raises(yaml.YAMLError):
        parse_stagea_task_config_bytes(b"value: [")


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


def _snapshot(root):
    return {str(path.relative_to(root)):
            (path.stat().st_mode, path.read_bytes() if path.is_file() else None)
            for path in root.rglob("*")}


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


@pytest.mark.parametrize("base_url,is_local", [
    ("http://localhost:8791/v1", True),
    ("http://127.12.34.56/v1", True),
    ("http://[::1]/v1", True),
    ("http://host.docker.internal/v1", True),
    ("http://host.containers.internal/v1", True),
    ("http://host.lima.internal/v1", True),
    ("http://relay/v1", True),
    ("10.20.30.40:8791/v1", True),
    ("http://172.16.1.2/v1", True),
    ("http://172.31.1.2/v1", True),
    ("http://192.168.1.2/v1", True),
    ("http://169.254.1.2/v1", True),
    ("http://100.64.1.2/v1", True),
    ("http://100.127.1.2/v1", True),
    ("http://[fd00::1]/v1", True),
    ("http://172.32.1.2/v1", False),
    ("http://100.128.1.2/v1", False),
    ("https://relay.example.com/v1", False),
    ("https://8.8.8.8/v1", False),
])
def test_route_local_context_pin_contract(base_url, is_local):
    """Literal locality controls the extra pin; classification performs no I/O."""
    route = dict(ROUTE, base_url=base_url)
    assert sp.resolve_route({"model": route}) == (MODEL, base_url, PROVIDER)
    route.pop("ollama_num_ctx")
    if is_local:
        with pytest.raises(sp.ProposalRefusal) as missing:
            sp.resolve_route({"model": route})
        assert missing.value.code == "route.local_probe_pin"
    else:
        assert sp.resolve_route({"model": route}) == (MODEL, base_url, PROVIDER)


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


def test_task_config_read_is_literal_and_has_no_loader_side_effects(
        tmp_path, monkeypatch, no_managed_overlay):
    """Read the actual task file without the general loader's home writes."""
    import hermes_cli.config as hermes_config

    root = make_root(tmp_path)
    before = _snapshot(tmp_path)

    def refuse(*args, **kwargs):
        raise AssertionError("general loader or managed content reached")

    for raw in ("read_user_config_raw", "read_raw_config",
                "read_raw_config_readonly"):
        monkeypatch.setattr(hermes_config, raw, refuse)
    monkeypatch.setattr(hermes_config, "load_config_readonly", refuse)
    monkeypatch.setattr(hermes_config, "ensure_hermes_home", refuse)
    monkeypatch.setattr(no_managed_overlay, "load_managed_config", refuse)
    token = sp.activate_config_root(root)
    try:
        document = sp.read_task_config(root)
    finally:
        reset_hermes_home_override(token)
    assert _snapshot(tmp_path) == before
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
    ("/synthetic/managed", {"model": {"default": "managed/other"}},
     "config.managed_dir"),
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
    _bind_managed(monkeypatch, "/synthetic/managed",
                  {"model": {"base_url": "https://managed.example/v1"}})
    reached = []
    monkeypatch.setattr(sp, "build_client",
                        lambda *route: reached.append(route))
    monkeypatch.setattr(sp, "create_completion",
                        lambda client, **kwargs: reached.append(kwargs))
    monkeypatch.setattr(sp.sys, "stdin", Stub(buffer=io.BytesIO(REQUEST)))
    assert sp.main([]) == 1
    assert reached == []
    assert "config.managed_dir" in capsys.readouterr().err


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


@pytest.mark.parametrize("field", ["default", "provider", "base_url"])
@pytest.mark.parametrize("literal", ["${STAGEA_INERT}", "$STAGEA_INERT",
                                     "%STAGEA_INERT%"])
def test_raw_route_substitution_is_refused_before_environment_read(
        tmp_path, monkeypatch, no_managed_overlay, field, literal):
    root = make_root(tmp_path, {"model": dict(ROUTE, **{field: literal})})
    monkeypatch.setenv("STAGEA_INERT", "inert-fixture-value")
    original = os._Environ.__getitem__

    def guarded(environment, key):
        assert key != "STAGEA_INERT", "route tried to read inherited value"
        return original(environment, key)

    monkeypatch.setattr(os._Environ, "__getitem__", guarded)
    before = _snapshot(tmp_path)
    token = sp.activate_config_root(root)
    try:
        with pytest.raises(sp.ProposalRefusal, match="config.environment_syntax"):
            sp.read_task_config(root)
    finally:
        reset_hermes_home_override(token)
    assert _snapshot(tmp_path) == before


def test_external_overlay_content_is_never_consumed(
        tmp_path, monkeypatch, no_managed_overlay):
    root = make_root(tmp_path)
    before = _snapshot(tmp_path)

    def refuse():
        raise AssertionError("external managed content was consumed")

    monkeypatch.setattr(no_managed_overlay, "load_managed_config", refuse)
    token = sp.activate_config_root(root)
    try:
        assert sp.resolve_route(sp.read_task_config(root)) == (
            MODEL, BASE_URL, PROVIDER)
        monkeypatch.setattr(no_managed_overlay, "get_managed_dir",
                            lambda: tmp_path / "managed")
        with pytest.raises(sp.ProposalRefusal, match="config.managed_dir"):
            sp.read_task_config(root)
    finally:
        reset_hermes_home_override(token)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("maximum", [False, True])
def test_binary_stdout_ignores_text_newline_translation(
        live, monkeypatch, maximum):
    paths = tuple("docs/frame%d.md" % index for index in range(6))
    raw = (_sized_envelope(sp.MAX_EMITTED_ENVELOPE_BYTES, paths)
           if maximum else ENVELOPE)
    request = (sp.canonical_bytes(dict(json.loads(REQUEST),
                                     admitted_write_set=list(paths)))
               if maximum else REQUEST)
    binary = io.BytesIO()
    text = io.TextIOWrapper(binary, encoding="ascii", newline="\r\n")
    monkeypatch.setattr(sp.sys, "stdout", text)
    monkeypatch.setattr(sp.sys, "stdin", Stub(buffer=io.BytesIO(request)))
    monkeypatch.setattr(sp, "create_completion", lambda *a, **k:
                        _response(sp.FRAMING_PREFIX + raw.decode("ascii")))
    try:
        assert sp.main([]) == 0
        assert binary.getvalue() == sp.FRAMING_PREFIX.encode("ascii") + raw + b"\n"
        assert len(binary.getvalue()) <= sp.MAX_FRAMED_STDOUT_BYTES
    finally:
        text.detach()


def _bind_offline_http(monkeypatch, handler):
    """Replace only the socket transport; exercise the actual SDK/client."""
    import httpx

    calls = []
    class OfflineTransport(httpx.BaseTransport):
        def __init__(self, **options):
            assert options == {"retries": 0, "trust_env": False}

        def handle_request(self, request):
            calls.append(request)
            return handler(request)

    monkeypatch.setattr(httpx, "HTTPTransport", OfflineTransport)
    return calls


def test_request_has_finite_generation_and_redirects_cannot_add_request(
        monkeypatch):
    import httpx

    calls = _bind_offline_http(monkeypatch, lambda request:
                              httpx.Response(302, headers={
                                  "location": "https://inert.invalid/unused"}))
    seam = sp.build_client(MODEL, BASE_URL, PROVIDER)
    try:
        kwargs = sp.build_request_kwargs(seam, json.loads(REQUEST))
        assert type(kwargs["max_tokens"]) is int
        assert 0 < kwargs["max_tokens"] == sp.MAX_COMPLETION_TOKENS
        assert seam.client._client.follow_redirects is False
        with pytest.raises(sp.ProposalRefusal, match="response.provider_failure"):
            sp.create_completion(seam.client, **kwargs)
        assert len(calls) == 1
        assert json.loads(calls[0].content)["max_tokens"] == sp.MAX_COMPLETION_TOKENS
    finally:
        seam.client.close()


def test_single_bounded_sdk_completion_preserves_whole_response(monkeypatch):
    import httpx

    framed = sp.FRAMING_PREFIX + ENVELOPE.decode("ascii")
    body = json.dumps({"id": "synthetic", "object": "chat.completion",
                       "created": 0, "model": MODEL,
                       "choices": [{"index": 0, "finish_reason": "stop",
                                    "message": {"role": "assistant",
                                                "content": framed}}]}).encode()
    monkeypatch.setattr(sp, "MAX_PROVIDER_RESPONSE_BYTES", len(body))
    calls = _bind_offline_http(monkeypatch, lambda request: httpx.Response(
        200, headers={"content-type": "application/json"},
        stream=httpx.ByteStream(body)))
    seam = sp.build_client(MODEL, BASE_URL, PROVIDER)
    try:
        response = sp.create_completion(
            seam.client, **sp.build_request_kwargs(seam, json.loads(REQUEST)))
        assert sp.response_text(response) == framed
        assert sp.validate_envelope(sp.extract_framed(sp.response_text(response)),
                                    REQUEST_ADMITTED) == ENVELOPE
        assert len(calls) == 1
    finally:
        seam.client.close()


def test_http_consumption_refuses_before_accumulating_past_ceiling(monkeypatch):
    import httpx

    # A tiny synthetic ceiling verifies consumption order without large input.
    monkeypatch.setattr(sp, "MAX_PROVIDER_RESPONSE_BYTES", 16)
    progress = []
    class SyntheticStream(httpx.SyncByteStream):
        def __iter__(self):
            for index in range(6):
                progress.append(index)
                yield b"12345678"

        def close(self):
            progress.append("closed")

    _bind_offline_http(monkeypatch, lambda request:
                       httpx.Response(200, stream=SyntheticStream()))
    seam = sp.build_client(MODEL, BASE_URL, PROVIDER)
    try:
        with seam.client._client.stream("GET", BASE_URL) as response:
            accepted = bytearray()
            with pytest.raises(sp.ProposalRefusal, match="response.bytes_length"):
                for chunk in response.iter_bytes():
                    accepted.extend(chunk)
            assert len(accepted) == 16
            assert progress == [0, 1, 2]
        assert progress == [0, 1, 2, "closed"]
    finally:
        seam.client.close()


def test_compressed_response_is_refused_before_consumption(monkeypatch):
    import httpx

    consumed = []
    class SyntheticStream(httpx.SyncByteStream):
        def __iter__(self):
            consumed.append("read")
            yield b"inert"

        def close(self):
            consumed.append("closed")

    _bind_offline_http(monkeypatch, lambda request: httpx.Response(
        200, headers={"content-encoding": "gzip"}, stream=SyntheticStream()))
    seam = sp.build_client(MODEL, BASE_URL, PROVIDER)
    try:
        with pytest.raises(sp.ProposalRefusal, match="response.content_encoding"):
            seam.client._client.get(BASE_URL)
        assert consumed == ["closed"]
    finally:
        seam.client.close()
