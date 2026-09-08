"""Controls for the Stage-A proposal-only program identity (AI-Org #198).

Two kinds of control live here.

**In-process controls** exercise the pure contract functions directly: argv,
request parsing, canonical bytes, envelope shape, framing, the egress guard
and the reduced-surface verifier.

**Hermetic subprocess controls (C14)** run the *real* entrypoint and the
*real* ``run_agent.AIAgent`` factory -- never a stub, fake or signature
check -- inside a process whose filesystem and network boundaries are
instrumented, and assert that the reduced path attempts **zero** access to
real configuration, managed scope, credentials or the network.

Why the isolation is written the way it is
------------------------------------------
``hermes_cli.managed_scope.get_managed_dir()`` resolves in this order:
``$HERMES_MANAGED_DIR`` when non-empty *and* an existing directory, then --
only when ``PYTEST_CURRENT_TEST`` is absent -- the system default.  A child
process is not pytest, so isolating ``HOME`` alone leaves the system managed
scope reachable.  Every subprocess here therefore points
``HERMES_MANAGED_DIR`` at an **existing, empty, test-local** directory, and
the observer additionally denies the system path outright, so neither the
environment nor a source change can silently re-widen the boundary.

An attempt is counted at the boundary *before* the underlying call runs, so
a broad ``except Exception`` downstream cannot turn an attempted access into
a false pass, and a denied attempt performs no real syscall.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Ambient names that are known to re-widen a deliberately reduced surface.
#: ``HERMES_KANBAN_TASK`` re-adds the kanban toolset to an explicitly empty
#: selection; ``HERMES_LAZY_INSTALL_TARGET`` makes the bootstrap import put a
#: writable directory on ``sys.path``.  The rest are what the Nix wrapper
#: injects into an ordinary Hermes process.
AMBIENT_REWIDENERS = (
    "HERMES_KANBAN_TASK",
    "HERMES_LAZY_INSTALL_TARGET",
    "HERMES_BUNDLED_SKILLS",
    "HERMES_OPTIONAL_SKILLS",
    "HERMES_BUNDLED_PLUGINS",
    "HERMES_OPTIONAL_MCPS",
    "HERMES_WEB_DIST",
    "HERMES_TUI_DIR",
    "HERMES_BIN",
    "HERMES_PYTHON",
    "HERMES_NODE",
    "HERMES_REVISION",
    "PYTHONPATH",
)

VALID_REQUEST = {
    "schema_version": "dyhano-stage-a-proposal-request-v1",
    "action_id": "act-0001",
    "decision_id": "dec-0001",
    "action_intent": "DISPATCH_AUTHOR",
    "kernel_version": "v1",
    "work_item_id": "wi-1",
    "contract_ref": "ref-1",
    "work_contract_fingerprint": "f" * 64,
    "canonical_repo": "owner/repo",
    "canonical_branch_ref": "refs/heads/main",
    "expected_main_sha": "a" * 40,
    "admitted_write_set": ["docs/a.md"],
    "envelope_schema_version": "dyhano-trusted-boundary-envelope-v1",
    "max_envelope_bytes": 262144,
}


def _request_bytes(**overrides):
    document = dict(VALID_REQUEST)
    document.update(overrides)
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ═════════════════════════════════════════════════════════════════════════
# In-process contract controls
# ═════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def mod():
    import agent.stagea_proposal_only as module
    return module


class TestArgvContract:
    def test_frozen_tail_is_exactly_dash_z(self, mod):
        assert mod.FROZEN_ARGV_TAIL == ("-z",)

    def test_frozen_tail_carries_no_escape_token(self, mod):
        """The tail must not be able to resume, continue or reselect."""
        joined = " ".join(mod.FROZEN_ARGV_TAIL).lower()
        for token in ("resume", "continue", "session", "chat", "history",
                      "provider", "fallback", "model", "api", "key", "token"):
            assert token not in joined

    @pytest.mark.parametrize("argv", [
        (), ("-z", "-z"), ("-z", "--resume"), ("--resume",), ("-Z",),
        ("-z", "--session", "abc"), ("",),
    ])
    def test_non_frozen_argv_is_refused_without_reading_stdin(self, mod, argv, monkeypatch):
        def _explode():
            raise AssertionError("stdin must not be read when argv is wrong")
        monkeypatch.setattr(mod, "_read_stdin_bounded", _explode)
        assert mod.main(list(argv)) == mod.EXIT_REFUSED


class TestRequestParsing:
    def test_valid_request_round_trips(self, mod):
        assert mod.parse_request(_request_bytes()) == VALID_REQUEST

    def test_extra_key_is_refused(self, mod):
        raw = json.dumps(dict(VALID_REQUEST, extra="x"), sort_keys=True).encode()
        with pytest.raises(mod.Refusal) as caught:
            mod.parse_request(raw)
        assert caught.value.code == "request.fields"

    def test_missing_key_is_refused(self, mod):
        trimmed = dict(VALID_REQUEST)
        del trimmed["action_id"]
        with pytest.raises(mod.Refusal) as caught:
            mod.parse_request(json.dumps(trimmed).encode())
        assert caught.value.code == "request.fields"

    def test_wrong_schema_version_is_refused(self, mod):
        with pytest.raises(mod.Refusal) as caught:
            mod.parse_request(_request_bytes(schema_version="something-else"))
        assert caught.value.code == "request.schema_version"

    @pytest.mark.parametrize("raw,code", [
        (b"", "request.length"),
        (b"not json", "request.json"),
        (b'"a string"', "request.shape"),
        (b"[]", "request.shape"),
        (b"\xff\xfe", "request.encoding"),
    ])
    def test_malformed_input_is_refused(self, mod, raw, code):
        with pytest.raises(mod.Refusal) as caught:
            mod.parse_request(raw)
        assert caught.value.code == code

    def test_oversize_request_is_refused(self, mod):
        with pytest.raises(mod.Refusal) as caught:
            mod.parse_request(b"x" * (mod.MAX_REQUEST_BYTES + 1))
        assert caught.value.code == "request.length"

    @pytest.mark.parametrize("field,value", [
        ("work_item_id", "resume-me"),
        ("contract_ref", "session_id-42"),
        ("canonical_repo", "owner/api_key"),
        ("admitted_write_set", ["docs/credential.md"]),
    ])
    def test_forbidden_token_anywhere_is_refused(self, mod, field, value):
        """A resume/provider/credential-shaped token is refused in any field."""
        with pytest.raises(mod.Refusal) as caught:
            mod.parse_request(_request_bytes(**{field: value}))
        assert caught.value.code == "request.hygiene.forbidden_token"

    def test_hygiene_screen_is_not_vacuous(self, mod):
        """Positive control: the clean request must survive the same screen."""
        mod._screen_request_hygiene(VALID_REQUEST)


class TestCanonicalBytes:
    def test_matches_the_frozen_canonicalization(self, mod):
        value = {"b": 1, "a": [2, 3], "u": "café"}
        raw = mod.canonical_dumps(value)
        assert raw == b'{"a":[2,3],"b":1,"u":"caf\\u00e9"}'

    def test_output_is_pure_ascii_so_framing_survives(self, mod):
        raw = mod.canonical_dumps({"k": "你好 — café"})
        assert all(0x20 <= byte <= 0x7E for byte in raw)

    def test_reserialization_is_byte_stable(self, mod):
        value = {"schema_version": "v", "mutations": [{"op": "create",
                 "path": "a/b.md", "content": "x"}]}
        raw = mod.canonical_dumps(value)
        assert mod.canonical_dumps(json.loads(raw.decode("ascii"))) == raw


class TestEnvelopeValidation:
    def _envelope(self, **overrides):
        document = {
            "schema_version": "dyhano-trusted-boundary-envelope-v1",
            "mutations": [{"op": "create", "path": "docs/a.md", "content": "hi"}],
        }
        document.update(overrides)
        return document

    def test_well_formed_envelope_is_accepted(self, mod):
        raw = mod.validate_envelope_document(self._envelope(), 262144)
        assert raw.startswith(b'{"mutations":')

    @pytest.mark.parametrize("document,code", [
        ({"schema_version": "v"}, "envelope.top_keys"),
        ({"schema_version": "v", "mutations": [], "x": 1}, "envelope.top_keys"),
        ({"schema_version": "v", "mutations": {}}, "envelope.mutations.type"),
        ({"schema_version": "v", "mutations": []}, "envelope.mutations.length"),
    ])
    def test_bad_top_level_is_refused(self, mod, document, code):
        with pytest.raises(mod.Refusal) as caught:
            mod.validate_envelope_document(document, 262144)
        assert caught.value.code == code

    @pytest.mark.parametrize("mutation,code", [
        ({"op": "rename", "path": "a.md", "content": "x"}, "envelope.mutation.op"),
        ({"op": "create", "path": "a.md"}, "envelope.mutation.keys"),
        ({"op": "create", "path": "a.md", "content": "x", "extra": 1}, "envelope.mutation.keys"),
        ({"op": "delete", "path": "a.md", "base_blob_sha": "s"}, None),
        ({"op": "create", "path": "../escape.md", "content": "x"}, "envelope.path.segment"),
        ({"op": "create", "path": ".git/config", "content": "x"}, "envelope.path.git_dir"),
        ({"op": "create", "path": "a//b.md", "content": "x"}, "envelope.path.segment"),
        ({"op": "create", "path": "a b.md", "content": "x"}, "envelope.path.grammar"),
        ({"op": "create", "path": "", "content": "x"}, "envelope.path.length"),
        ({"op": "create", "path": "a.md", "content": 5}, "envelope.mutation.content.type"),
    ])
    def test_mutation_shapes(self, mod, mutation, code):
        document = self._envelope(mutations=[mutation])
        if code is None:
            mod.validate_envelope_document(document, 262144)
            return
        with pytest.raises(mod.Refusal) as caught:
            mod.validate_envelope_document(document, 262144)
        assert caught.value.code == code

    def test_duplicate_paths_are_refused(self, mod):
        document = self._envelope(mutations=[
            {"op": "create", "path": "docs/a.md", "content": "1"},
            {"op": "create", "path": "docs/a.md", "content": "2"},
        ])
        with pytest.raises(mod.Refusal) as caught:
            mod.validate_envelope_document(document, 262144)
        assert caught.value.code == "envelope.paths.collide"

    def test_prefix_colliding_paths_are_refused(self, mod):
        document = self._envelope(mutations=[
            {"op": "create", "path": "docs", "content": "1"},
            {"op": "create", "path": "docs/a.md", "content": "2"},
        ])
        with pytest.raises(mod.Refusal) as caught:
            mod.validate_envelope_document(document, 262144)
        assert caught.value.code == "envelope.paths.collide"

    def test_too_many_files_is_refused(self, mod):
        mutations = [{"op": "create", "path": "d/f%d.md" % i, "content": "x"}
                     for i in range(mod.MAX_ENVELOPE_FILES + 1)]
        with pytest.raises(mod.Refusal) as caught:
            mod.validate_envelope_document(self._envelope(mutations=mutations), 262144)
        assert caught.value.code == "envelope.mutations.length"

    def test_oversize_file_content_is_refused(self, mod):
        mutation = {"op": "create", "path": "d/f.md",
                    "content": "x" * (mod.MAX_FILE_CONTENT_BYTES + 1)}
        with pytest.raises(mod.Refusal) as caught:
            mod.validate_envelope_document(self._envelope(mutations=[mutation]), 262144)
        assert caught.value.code == "envelope.mutation.content.length"

    def test_envelope_over_the_request_cap_is_refused(self, mod):
        with pytest.raises(mod.Refusal) as caught:
            mod.validate_envelope_document(self._envelope(), 8)
        assert caught.value.code == "envelope.length"


class TestFraming:
    def test_happy_path_is_one_prefixed_printable_line(self, mod):
        line = mod.frame_envelope(b'{"a":1}')
        assert line == mod.FRAMING_PREFIX + '{"a":1}'
        assert "\n" not in line

    @pytest.mark.parametrize("payload,code", [
        (b"", "framing.empty_payload"),
        ("x".encode(), None),
        (b"caf\xc3\xa9", "framing.non_printable"),
        (b"a\nb", "framing.non_printable"),
        (b"a\tb", "framing.non_printable"),
    ])
    def test_payload_screening(self, mod, payload, code):
        if code is None:
            mod.frame_envelope(payload)
            return
        with pytest.raises(mod.Refusal) as caught:
            mod.frame_envelope(payload)
        assert caught.value.code == code

    def test_oversize_line_is_refused_rather_than_truncated(self, mod):
        """Truncation invalidates the attempt upstream, so refuse instead."""
        with pytest.raises(mod.Refusal) as caught:
            mod.frame_envelope(b"x" * mod.MAX_STDOUT_BYTES)
        assert caught.value.code == "framing.length"


class TestEgressGuard:
    def test_counts_and_denies_each_boundary(self, mod):
        import socket
        guard = mod.ConstructionEgressGuard()
        with guard:
            for call in (
                lambda: socket.create_connection(("127.0.0.1", 9)),
                lambda: socket.getaddrinfo("localhost", 80),
            ):
                with pytest.raises(mod.Refusal):
                    call()
        assert guard.attempts == 2

    def test_restores_every_original(self, mod):
        import socket
        before = (socket.create_connection, socket.getaddrinfo,
                  socket.socket.connect, socket.socket.connect_ex)
        with mod.ConstructionEgressGuard():
            assert socket.create_connection is not before[0]
        after = (socket.create_connection, socket.getaddrinfo,
                 socket.socket.connect, socket.socket.connect_ex)
        assert before == after

    def test_restores_originals_even_when_the_block_raises(self, mod):
        import socket
        before = socket.create_connection
        with pytest.raises(ValueError):
            with mod.ConstructionEgressGuard():
                raise ValueError("boom")
        assert socket.create_connection is before

    def test_a_swallowed_attempt_is_still_counted(self, mod):
        """The counter must see through a broad ``except Exception``.

        This is the shape accepted Hermes source actually uses around its
        endpoint probes, so a guard that only observed escaping exceptions
        would report an attempted access as an absence.
        """
        import socket
        guard = mod.ConstructionEgressGuard()
        with guard:
            try:
                socket.create_connection(("127.0.0.1", 9))
            except Exception:
                pass
        assert guard.attempts == 1


class _FakeAgent:
    """A plain object used only to exercise the *verifier*.

    This never stands in for the factory: every C14 control below builds the
    real ``AIAgent``.  Here we only need objects whose resolved attributes
    are wrong in one specific way each.
    """

    def __init__(self, **attributes):
        defaults = {
            "tools": [], "valid_tool_names": set(), "enabled_toolsets": [],
            "max_iterations": 1, "_memory_enabled": False,
            "_memory_manager": None, "_memory_store": None,
            "load_soul_identity": False, "skip_context_files": True,
            "skip_background_review": True,
        }
        defaults.update(attributes)
        for key, value in defaults.items():
            setattr(self, key, value)


class TestReducedSurfaceVerifier:
    def test_accepts_a_fully_reduced_surface(self, mod):
        surface = mod.verify_reduced_surface(_FakeAgent())
        assert surface["tool_count"] == 0
        assert surface["enabled_toolsets"] == []

    @pytest.mark.parametrize("attributes,code", [
        ({"enabled_toolsets": None}, "reduced.toolsets.absent_selection"),
        ({"enabled_toolsets": ["memory"]}, "reduced.toolsets.not_empty"),
        ({"tools": [{"function": {"name": "terminal"}}]}, "reduced.tools.not_empty"),
        ({"valid_tool_names": {"terminal"}}, "reduced.tools.not_empty"),
        ({"max_iterations": sys.maxsize}, "reduced.iterations.unbounded"),
        ({"max_iterations": 0}, "reduced.iterations.unbounded"),
        ({"max_iterations": True}, "reduced.iterations.type"),
        ({"max_iterations": "1"}, "reduced.iterations.type"),
        ({"_memory_enabled": True}, "reduced.memory.present"),
        ({"_memory_manager": object()}, "reduced.memory.present"),
        ({"_memory_store": object()}, "reduced.memory.present"),
        ({"load_soul_identity": True}, "reduced.soul.loaded"),
        ({"skip_context_files": False}, "reduced.context_files.loaded"),
        ({"skip_background_review": False}, "reduced.background_review.enabled"),
    ])
    def test_refuses_every_non_reduced_shape(self, mod, attributes, code):
        with pytest.raises(mod.Refusal) as caught:
            mod.verify_reduced_surface(_FakeAgent(**attributes))
        assert caught.value.code == code

    def test_none_toolsets_is_not_treated_as_empty(self, mod):
        """``None`` means *every* toolset to ``model_tools``, not none."""
        assert mod.REDUCED_ENABLED_TOOLSETS == []
        assert mod.REDUCED_ENABLED_TOOLSETS is not None


class TestPackagingDeclaration:
    """Declaration invariants -- assertions about a config file, not source text."""

    def test_entry_point_is_declared_and_targets_this_module(self):
        import tomllib
        with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
            scripts = tomllib.load(handle)["project"]["scripts"]
        assert scripts["hermes-stagea-proposal"] == "agent.stagea_proposal_only:main"

    def test_pre_existing_entry_points_are_unchanged(self):
        import tomllib
        with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
            scripts = tomllib.load(handle)["project"]["scripts"]
        assert scripts["hermes"] == "hermes_cli.main:main"
        assert scripts["hermes-agent"] == "run_agent:main"
        assert scripts["hermes-acp"] == "acp_adapter.entry:main"

    def test_declared_target_is_importable_and_callable(self):
        import agent.stagea_proposal_only as module
        assert callable(module.main)


# ═════════════════════════════════════════════════════════════════════════
# C14 — hermetic subprocess controls over the REAL entrypoint and factory
# ═════════════════════════════════════════════════════════════════════════

#: Observer installed in the child *before* any Hermes module is imported.
#:
#: Every monitored call is classified and counted at the boundary, and a
#: prohibited one is denied *without* invoking the original -- so recording
#: an attempt never performs a real read, connect or DNS lookup.
OBSERVER_SOURCE = r'''
import builtins
import io
import json
import os
import pathlib
import socket
import sys

import sysconfig

ROOT = os.environ["STAGEA_ROOT"]
REPORT = os.environ["STAGEA_REPORT"]
MODE = os.environ["STAGEA_MODE"]


def _variants(path):
    """A path and its realpath -- macOS /tmp vs /private/tmp, venv symlinks."""
    out = {path}
    try:
        out.add(os.path.realpath(path))
    except OSError:
        pass
    return {p for p in out if p}


# Code roots: the repo, plus this interpreter's own prefixes and stdlib /
# site-packages directories, resolved by the child itself so a venv symlink
# or a differing base prefix cannot masquerade as unknown state.
_CODE_ROOTS = set()
for _entry in os.environ["STAGEA_ALLOWED"].split(os.pathsep):
    if _entry:
        _CODE_ROOTS |= _variants(_entry)
for _entry in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
    if _entry:
        _CODE_ROOTS |= _variants(_entry)
for _entry in sysconfig.get_paths().values():
    if _entry:
        _CODE_ROOTS |= _variants(_entry)
ALLOWED = tuple(sorted(_CODE_ROOTS))
ROOTS = tuple(sorted(_variants(ROOT)))

# Hermes configuration / managed-scope / credential / provider state.
# Denied for EVERY monitored class, including metadata, because whether any
# of it is even reachable is exactly what this control exists to settle.
# REAL_HOME is the parent's actual home, passed in explicitly so the child
# can name it without expanding "~" (which points at the ephemeral root).
REAL_HOME = os.environ.get("STAGEA_REAL_HOME", "")
_PROHIBITED = ["/etc/hermes", "/private/etc/hermes"]
for _home in _variants(REAL_HOME) if REAL_HOME else ():
    for _leaf in (".hermes", ".config", ".aws", ".ssh", ".netrc", ".docker",
                  ".npmrc", "Library/Keychains",
                  "Library/Application Support/hermes"):
        _PROHIBITED.append(os.path.join(_home, _leaf))
ALWAYS_PROHIBITED = tuple(sorted(set(_PROHIBITED)))

#: Credential/config-shaped basenames, prohibited wherever they appear
#: outside the ephemeral root.
PROHIBITED_LEAVES = ("auth.json", ".netrc", "credentials", "keychain-db",
                     ".env", "config.yaml", "config.json", "id_rsa")

#: Metadata-only classes.  Existence/mode of a path carries no configuration
#: content, and platform/capability detection legitimately stats system
#: paths, so these are permitted outside the allowlist -- but every such path
#: is recorded and asserted about.  Content reads (open/os.open) and
#: directory enumeration (listdir/scandir) are NEVER permitted outside it.
METADATA_CLASSES = ("os.stat", "os.lstat")

#: Device nodes and kernel pseudo-filesystems.  These are OS interfaces
#: (``/dev/null``, container detection via ``/proc/1/cgroup``), not files
#: carrying Hermes configuration, credential or provider content.  The list
#: is closed by construction rather than grown per failure.
OS_PSEUDO_ROOTS = ("/dev/", "/proc/", "/sys/")

#: OS / platform identification. Reading these tells you which operating
#: system you are on and nothing about Hermes configuration, credentials or
#: providers. Closed set, matched exactly.
OS_PLATFORM_FILES = frozenset((
    "/System/Library/CoreServices/SystemVersion.plist",
    "/etc/os-release", "/private/etc/os-release",
    "/etc/lsb-release", "/etc/debian_version", "/etc/redhat-release",
    "/etc/alpine-release", "/etc/machine-id", "/etc/localtime",
    "/private/etc/localtime",
))

STATE = {
    "prohibited": [],       # config / managed-scope / credential / provider
    "network": [],          # any outbound socket or DNS attempt
    "testlocal_reads": 0,   # reads under the ephemeral test root (permitted)
    "code_reads": 0,        # interpreter / repo / venv code and package data
    "metadata_outside": [], # stat-only paths outside the allowlist
    "config_probes": [],    # existence checks on checkout-local config leaves
    "os_pseudo_reads": 0,   # /dev, /proc, /sys kernel interfaces
    "unclassified": 0,
    "fired": {},            # monitored class -> times the observer fired
    "run_conversation": 0,
}


class Denied(Exception):
    pass


def _record(cls):
    STATE["fired"][cls] = STATE["fired"].get(cls, 0) + 1


def _classify(raw, cls):
    try:
        path = os.fspath(raw)
    except TypeError:
        return "unclassified"
    if isinstance(path, bytes):
        try:
            path = path.decode("utf-8", "strict")
        except UnicodeDecodeError:
            return "unclassified"
    if not isinstance(path, str):
        return "unclassified"
    absolute = os.path.abspath(path)

    for prefix in ALWAYS_PROHIBITED:
        if absolute == prefix or absolute.startswith(prefix + os.sep):
            return "prohibited"

    for prefix in ROOTS:
        if absolute == prefix or absolute.startswith(prefix + os.sep):
            return "testlocal"

    for prefix in OS_PSEUDO_ROOTS:
        if absolute.startswith(prefix):
            return "os_pseudo"

    if absolute in OS_PLATFORM_FILES:
        return "os_platform"

    config_shaped = os.path.basename(absolute) in PROHIBITED_LEAVES
    in_code_root = any(absolute == prefix or absolute.startswith(prefix + os.sep)
                       for prefix in ALLOWED)

    if in_code_root:
        if config_shaped:
            # Accepted source existence-checks a checkout-local dotenv during
            # import.  Reading one would be a real configuration load, so that
            # is denied; the existence check itself is recorded for disclosure.
            return "config_probe" if cls in METADATA_CLASSES else "prohibited"
        return "code"

    if config_shaped:
        return "prohibited"

    if cls in METADATA_CLASSES:
        return "metadata_outside"
    return "prohibited"


def _guard_path(cls, original):
    def wrapper(path, *args, **kwargs):
        _record(cls)
        verdict = _classify(path, cls)
        if verdict == "prohibited":
            STATE["prohibited"].append({"class": cls, "path": str(path)})
            raise Denied("prohibited filesystem access: %s" % (path,))
        if verdict == "testlocal":
            STATE["testlocal_reads"] += 1
        elif verdict == "code":
            STATE["code_reads"] += 1
        elif verdict == "metadata_outside":
            STATE["metadata_outside"].append(os.path.abspath(str(path)))
        elif verdict in ("os_pseudo", "os_platform"):
            STATE["os_pseudo_reads"] += 1
        elif verdict == "config_probe":
            STATE["config_probes"].append({
                "class": cls,
                "path": os.path.abspath(str(path)),
                "exists": _EXISTS(os.path.abspath(str(path))),
            })
        else:
            STATE["unclassified"] += 1
        return original(path, *args, **kwargs)
    return wrapper


def _guard_network(cls, original):
    def wrapper(*args, **kwargs):
        _record(cls)
        target = args[1] if (cls.startswith("socket.") and len(args) > 1) else (
            args[0] if args else None)
        STATE["network"].append({"class": cls, "target": repr(target)[:120]})
        raise Denied("outbound network denied: %s" % cls)
    return wrapper


_ORIGINALS = {}


def _EXISTS(path):
    """Existence via the ORIGINAL stat, so the instrument never self-reports."""
    try:
        _ORIGINALS[("os", "stat")](path)
        return True
    except OSError:
        return False


def install():
    for name in ("open",):
        _ORIGINALS[("builtins", name)] = getattr(builtins, name)
        setattr(builtins, name, _guard_path("builtins.open", getattr(builtins, name)))
    original_io_open = io.open
    _ORIGINALS[("io", "open")] = original_io_open
    io.open = _guard_path("io.open", original_io_open)
    for name in ("open", "stat", "lstat", "listdir", "scandir"):
        original = getattr(os, name)
        _ORIGINALS[("os", name)] = original
        setattr(os, name, _guard_path("os." + name, original))
    for name in ("create_connection", "getaddrinfo", "gethostbyname"):
        original = getattr(socket, name, None)
        if original is None:
            continue
        _ORIGINALS[("socket", name)] = original
        setattr(socket, name, _guard_network("socket." + name, original))
    for name in ("connect", "connect_ex"):
        original = getattr(socket.socket, name)
        _ORIGINALS[("socket.socket", name)] = original
        setattr(socket.socket, name, _guard_network("socket.socket." + name, original))


def dump(extra):
    STATE.update(extra)
    original_open = _ORIGINALS[("builtins", "open")]
    with original_open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(STATE, handle, default=str)


install()

result = {}
try:
    # Import the real modules under observation -- nothing is stubbed.
    import run_agent

    def _counted_run_conversation(self, *args, **kwargs):
        STATE["run_conversation"] += 1
        raise Denied("run_conversation must not be reached in these controls")

    run_agent.AIAgent.run_conversation = _counted_run_conversation

    import agent.stagea_proposal_only as program

    if MODE == "main":
        result["exit_code"] = program.main(["-z"])
    elif MODE == "factory":
        agent, attempts = program.build_reduced_agent()
        result["construction_egress_attempts"] = attempts
        result["surface"] = program.describe_resolved_surface(agent)
        program.verify_reduced_surface(agent)
        result["verified_reduced"] = True
    elif MODE == "negative":
        # Deliberately attempt one access of every monitored class against a
        # prohibited target.  Denial happens before the original call, so no
        # real read, connect or lookup occurs.
        probes = [
            ("io.open", lambda: pathlib.Path(__file__).read_text(encoding="utf-8")),
            ("builtins.open", lambda: builtins.open("/etc/hermes/config.yaml")),
            ("os.open", lambda: os.open("/etc/hermes/config.yaml", os.O_RDONLY)),
            ("os.stat", lambda: os.stat("/etc/hermes")),
            ("os.lstat", lambda: os.lstat("/etc/hermes")),
            ("os.listdir", lambda: os.listdir("/etc/hermes")),
            ("os.scandir", lambda: os.scandir("/etc/hermes")),
            ("socket.create_connection", lambda: socket.create_connection(("127.0.0.1", 9))),
            ("socket.getaddrinfo", lambda: socket.getaddrinfo("localhost", 80)),
            ("socket.socket.connect",
             lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("127.0.0.1", 9))),
        ]
        observed = {}
        for label, probe in probes:
            before = STATE["fired"].get(label, 0)
            try:
                probe()
            except Exception:
                # Swallow exactly the way accepted Hermes source does around
                # its endpoint probes: the counter must still have recorded it.
                pass
            observed[label] = STATE["fired"].get(label, 0) - before
        result["negative_control"] = observed
    else:
        raise AssertionError("unknown mode: %s" % MODE)
except BaseException as exc:
    result["error"] = "%s: %s" % (type(exc).__name__, exc)
finally:
    dump(result)
'''


def _hermetic_root(tmp_path):
    """Build the ephemeral state the child is allowed to see.

    ``managed`` is created **empty but existing** on purpose: accepted
    ``get_managed_dir()`` honours ``HERMES_MANAGED_DIR`` only when it names an
    existing directory, and falls through to the system managed scope when it
    does not.  A non-existent path here would silently re-open the very
    boundary these controls exist to close.
    """
    root = tmp_path / "stagea-hermetic"
    for name in ("home", "hermes_home", "managed", "tmp", "cwd", "out"):
        (root / name).mkdir(parents=True)
    assert (root / "managed").is_dir()
    assert not any((root / "managed").iterdir())
    return root


def _hermetic_env(root, extra=None):
    """An explicitly constructed child environment -- nothing is inherited."""
    env = {
        "HOME": str(root / "home"),
        "TMPDIR": str(root / "tmp"),
        "HERMES_HOME": str(root / "hermes_home"),
        # The load-bearing correction: an existing, empty, test-local managed
        # scope so accepted source cannot reach the system default.
        "HERMES_MANAGED_DIR": str(root / "managed"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "PYTHONDONTWRITEBYTECODE": "1",
        "STAGEA_ROOT": str(root),
        "STAGEA_REPORT": str(root / "out" / "report.json"),
        "STAGEA_REAL_HOME": os.path.expanduser("~"),
        "STAGEA_ALLOWED": os.pathsep.join(
            [str(REPO_ROOT), sys.prefix, sys.base_prefix]
        ),
    }
    for name in AMBIENT_REWIDENERS:
        assert name not in env, "re-widener must not be in the base environment"
    if extra:
        env.update(extra)
    return env


def _run_observed(root, mode, stdin=b"", extra_env=None, script=None):
    """Run the observer child and return ``(completed, report)``."""
    env = _hermetic_env(root, extra_env)
    env["STAGEA_MODE"] = mode
    harness = root / "observer.py"
    harness.write_text(OBSERVER_SOURCE if script is None else script, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(harness)],
        input=stdin, env=env, cwd=str(root / "cwd"),
        capture_output=True, timeout=600,
    )
    report_path = root / "out" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    return completed, report


#: Substrings that would make a stat-only path config/credential shaped.
#: ``metadata_outside`` is permitted, so it must be asserted about rather
#: than merely tolerated: a stat of a Hermes/credential path would mean the
#: boundary was reached even though nothing was read.
CONFIG_SHAPED_LEAVES = frozenset((
    ".hermes", "auth.json", ".netrc", "credentials", "keychain-db",
    "id_rsa", ".aws", ".ssh", "config.yaml", "config.json", ".env",
))


def _assert_clean_metadata(report):
    """Stat-only access outside the allowlist is permitted but bounded.

    Matched on the basename: an ancestor directory whose *name* happens to
    contain "hermes" is traversal, not configuration access.
    """
    offenders = sorted({path for path in report["metadata_outside"]
                        if os.path.basename(path).lower() in CONFIG_SHAPED_LEAVES})
    assert offenders == [], "config-shaped metadata access: %r" % (offenders,)


def _assert_no_prohibited_access(report):
    assert report["prohibited"] == [], report["prohibited"]
    assert report["network"] == [], report["network"]
    _assert_clean_metadata(report)
    # Checkout-local config existence checks are permitted but disclosed:
    # they must be metadata only, and must not have found a real file.
    for probe in report["config_probes"]:
        assert probe["class"] in ("os.stat", "os.lstat"), probe
        assert probe["exists"] is False, (
            "accepted source found a checkout-local config file: %r" % (probe,))


def _write_relay_config(root, hardened=True):
    """A synthetic, entirely test-local provider configuration.

    Mirrors the accepted Stage-A shape -- a fixed loopback endpoint and a
    non-secret local sentinel (#187 B) -- so the reduced surface becomes
    observable.  No real endpoint, account or credential is involved, and
    the observer denies every outbound attempt regardless.
    """
    lines = [
        "model:",
        "  provider: stagea-local-relay",
        "  default: stagea-proposal-only",
        "  api_mode: chat_completions",
    ]
    if hardened:
        # Suppress the two accepted-source construction-time endpoint probes.
        lines += ["  ollama_num_ctx: 8192", "  context_length: 8192"]
    lines += [
        "providers:",
        "  stagea-local-relay:",
        "    api: http://127.0.0.1:18731/v1",
        "    api_key: not-a-secret-local-sentinel",
        "    default_model: stagea-proposal-only",
    ]
    if hardened:
        lines += ["tools:", "  tool_search:", "    enabled: off"]
    (root / "hermes_home" / "config.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


class TestC14Instrumentation:
    """Negative controls: prove every monitored class actually fires."""

    def test_every_monitored_class_records_a_denied_attempt(self, tmp_path):
        root = _hermetic_root(tmp_path)
        completed, report = _run_observed(root, "negative")
        assert report is not None, completed.stderr.decode()[-2000:]
        observed = report["negative_control"]
        assert observed, "negative control produced no observations"
        for label, times in sorted(observed.items()):
            assert times == 1, "observer did not fire for %s: %r" % (label, observed)

    def test_denied_attempts_are_recorded_despite_being_swallowed(self, tmp_path):
        """Each probe above ran inside ``except Exception: pass``.

        The counts still land, which is exactly what makes a zero-attempt
        assertion meaningful against accepted source's broad handlers.
        """
        root = _hermetic_root(tmp_path)
        _, report = _run_observed(root, "negative")
        assert len(report["prohibited"]) >= 6
        assert len(report["network"]) >= 3

    def test_the_system_managed_scope_path_is_always_prohibited(self, tmp_path):
        """The exact real managed-scope target is denied before any syscall."""
        root = _hermetic_root(tmp_path)
        _, report = _run_observed(root, "negative")
        prohibited = {entry["path"] for entry in report["prohibited"]}
        assert any(path.startswith("/etc/hermes") for path in prohibited)


class TestC14DefaultOffPath:
    """The real entrypoint, no provider configured: the default posture."""

    def test_refuses_with_zero_prohibited_access_and_no_conversation(self, tmp_path):
        root = _hermetic_root(tmp_path)
        completed, report = _run_observed(root, "main", stdin=_request_bytes())
        assert report is not None, completed.stderr.decode()[-2000:]
        assert report.get("error") is None, report.get("error")

        assert report["exit_code"] != 0, "default-off path must fail closed"
        _assert_no_prohibited_access(report)
        assert report["run_conversation"] == 0

    def test_emits_no_framing_line_on_stdout(self, tmp_path):
        root = _hermetic_root(tmp_path)
        completed, _ = _run_observed(root, "main", stdin=_request_bytes())
        assert b"DYHANO-STAGE-A-PROPOSAL-V1" not in completed.stdout
        assert completed.stdout == b""

    def test_the_observer_was_actually_live_during_the_run(self, tmp_path):
        """Positive control against a vacuous zero: the spies did fire."""
        root = _hermetic_root(tmp_path)
        _, report = _run_observed(root, "main", stdin=_request_bytes())
        assert report["fired"], "no monitored class fired at all"
        assert report["code_reads"] > 0, "no code reads observed -- spies inert?"


class TestC14ReducedSurface:
    """The real factory with test-local ephemeral relay config."""

    def test_resolved_surface_is_reduced_with_zero_prohibited_access(self, tmp_path):
        root = _hermetic_root(tmp_path)
        _write_relay_config(root, hardened=True)
        completed, report = _run_observed(root, "factory")
        assert report is not None, completed.stderr.decode()[-2000:]
        assert report.get("error") is None, report.get("error")

        _assert_no_prohibited_access(report)
        assert report["construction_egress_attempts"] == 0
        assert report["run_conversation"] == 0
        assert report["verified_reduced"] is True

        surface = report["surface"]
        assert surface["tool_count"] == 0
        assert surface["valid_tool_names"] == []
        assert surface["enabled_toolsets"] == []
        assert surface["enabled_toolsets_is_none"] is False
        assert surface["max_iterations"] == 1
        assert surface["memory_enabled"] is False
        assert surface["memory_manager_present"] is False
        assert surface["memory_store_present"] is False
        assert surface["load_soul_identity"] is False
        assert surface["skip_context_files"] is True
        assert surface["skip_background_review"] is True

    def test_the_ephemeral_config_was_really_consumed(self, tmp_path):
        """Distinguish a test-local config read from prohibited real access."""
        root = _hermetic_root(tmp_path)
        _write_relay_config(root, hardened=True)
        _, report = _run_observed(root, "factory")
        assert report["testlocal_reads"] > 0
        assert report["prohibited"] == []


class TestC14AmbientRewideners:
    """Ambient re-wideners are real, and the program must refuse them.

    ``model_tools`` appends the kanban toolset to an explicitly EMPTY
    selection when ``HERMES_KANBAN_TASK`` is set in a dispatcher-owned
    worker context.  These controls prove that it genuinely happens against
    the real factory, and that the post-construction verifier catches it
    instead of running a silently re-widened agent.
    """

    def _run(self, tmp_path, extra_env, mode="factory"):
        root = _hermetic_root(tmp_path)
        _write_relay_config(root, hardened=True)
        completed, report = _run_observed(
            root, mode, stdin=_request_bytes(), extra_env=extra_env)
        assert report is not None, completed.stderr.decode()[-2000:]
        return completed, report

    def test_kanban_task_really_re_widens_the_empty_toolset(self, tmp_path):
        _, baseline = self._run(tmp_path / "a", None)
        _, widened = self._run(tmp_path / "b", {"HERMES_KANBAN_TASK": "task-1"})
        assert baseline["surface"]["tool_count"] == 0
        assert widened["surface"]["tool_count"] > 0, (
            "expected the kanban re-widening path to fire; if it no longer "
            "does, this control and the reduced-surface claim need revisiting"
        )
        # The empty *selection* is unchanged -- the widening is invisible there,
        # which is exactly why the verifier reads the resolved surface instead.
        assert widened["surface"]["enabled_toolsets"] == []
        _assert_no_prohibited_access(widened)

    def test_the_program_refuses_a_re_widened_surface(self, tmp_path):
        _, widened = self._run(tmp_path, {"HERMES_KANBAN_TASK": "task-1"})
        assert widened.get("verified_reduced") is not True
        assert "reduced.tools.not_empty" in (widened.get("error") or "")

    def test_end_to_end_main_fails_closed_under_a_re_widener(self, tmp_path):
        completed, report = self._run(
            tmp_path, {"HERMES_KANBAN_TASK": "task-1"}, mode="main")
        assert report["exit_code"] != 0
        assert report["run_conversation"] == 0
        assert completed.stdout == b""
        assert b"reduced.tools.not_empty" in completed.stderr
        _assert_no_prohibited_access(report)

    def test_wrapper_environment_without_kanban_does_not_re_widen(self, tmp_path):
        _, baseline = self._run(tmp_path / "a", None)
        injected = {name: "1" for name in AMBIENT_REWIDENERS
                    if name not in ("PYTHONPATH", "HERMES_LAZY_INSTALL_TARGET",
                                    "HERMES_KANBAN_TASK")}
        _, widened = self._run(tmp_path / "b", injected)
        assert widened["surface"] == baseline["surface"]
        _assert_no_prohibited_access(widened)

    def test_lazy_install_target_does_not_re_widen(self, tmp_path):
        _, baseline = self._run(tmp_path / "a", None)
        target = tmp_path / "b" / "lazy"
        _, widened = self._run(
            tmp_path / "b", {"HERMES_LAZY_INSTALL_TARGET": str(target)})
        assert widened["surface"] == baseline["surface"]
        _assert_no_prohibited_access(widened)


class TestC14SwallowedProbeDetection:
    """Accepted source probes its endpoint unless configuration stops it.

    Without the two suppressing keys the construction-time context-length
    resolution and Ollama detection run, and their failures are swallowed by
    broad handlers.  Recording them proves the zero-attempt assertions
    elsewhere are not vacuous, and pins a concrete deployment requirement.
    """

    def test_unhardened_config_produces_recorded_swallowed_attempts(self, tmp_path):
        root = _hermetic_root(tmp_path)
        _write_relay_config(root, hardened=False)
        _, report = _run_observed(root, "factory", stdin=_request_bytes())
        assert report["construction_egress_attempts"] > 0, (
            "expected accepted source to attempt endpoint metadata probes; "
            "if this is now zero the suppression requirement may have been "
            "fixed upstream and this control needs revisiting"
        )

    def test_the_program_refuses_when_construction_attempted_egress(self, tmp_path):
        root = _hermetic_root(tmp_path)
        _write_relay_config(root, hardened=False)
        completed, report = _run_observed(root, "main", stdin=_request_bytes())
        assert report["exit_code"] != 0
        assert report["run_conversation"] == 0
        assert completed.stdout == b""
        assert b"construction.egress_attempted" in completed.stderr

    def test_hardened_config_reaches_zero_attempts(self, tmp_path):
        root = _hermetic_root(tmp_path)
        _write_relay_config(root, hardened=True)
        _, report = _run_observed(root, "factory", stdin=_request_bytes())
        assert report["construction_egress_attempts"] == 0
        _assert_no_prohibited_access(report)


class TestBootstrapFirstInvariant:
    """The entry point must import ``hermes_bootstrap`` before anything else.

    Proven behaviourally by recording real import order in a fresh child --
    not by reading the module's source text, which AGENTS.md bans outright
    and which would pass even if the import were wired wrong.
    """

    IMPORT_ORDER_SOURCE = r'''
import json, os, sys

ORDER = []


class Recorder:
    def find_module(self, fullname, path=None):
        return None

    def find_spec(self, fullname, path=None, target=None):
        ORDER.append(fullname)
        return None


sys.meta_path.insert(0, Recorder())
import agent.stagea_proposal_only  # noqa: F401
sys.meta_path.pop(0)

entry = "agent.stagea_proposal_only"
after = ORDER[ORDER.index(entry) + 1:] if entry in ORDER else None
with open(os.environ["STAGEA_REPORT"], "w", encoding="utf-8") as handle:
    json.dump({"order": ORDER, "after_entry": after,
               "bootstrap_imported": "hermes_bootstrap" in sys.modules},
              handle)
'''

    def test_bootstrap_is_the_first_module_the_entry_point_imports(self, tmp_path):
        root = _hermetic_root(tmp_path)
        completed, report = _run_observed(
            root, "import-order", script=self.IMPORT_ORDER_SOURCE)
        assert report is not None, completed.stderr.decode()[-2000:]
        assert report["bootstrap_imported"] is True
        after = report["after_entry"]
        assert after is not None, "entry point never appeared in the import order"
        assert after, "entry point imported nothing at all"
        assert after[0] == "hermes_bootstrap", (
            "first import after the entry point began executing was %r" % (after[:5],))


class TestC14RealInstalledExecutable:
    """The installed console script, under the same corrected isolation.

    This replaces the two earlier five-key smoke runs, which omitted managed
    scope entirely and are recorded as void.
    """

    def _console_script(self):
        candidate = pathlib.Path(sys.executable).parent / "hermes-stagea-proposal"
        return candidate if candidate.exists() else None

    def test_installed_entry_point_fails_closed_with_no_stdout(self, tmp_path):
        script = self._console_script()
        if script is None:
            pytest.skip("console script not installed in this environment")
        root = _hermetic_root(tmp_path)
        env = _hermetic_env(root)
        env.pop("STAGEA_REPORT", None)
        completed = subprocess.run(
            [str(script), "-z"], input=_request_bytes(), env=env,
            cwd=str(root / "cwd"), capture_output=True, timeout=600)
        assert completed.returncode != 0
        assert completed.stdout == b""
        assert b"DYHANO-STAGE-A-PROPOSAL-V1" not in completed.stdout

    def test_installed_entry_point_refuses_a_non_frozen_argv(self, tmp_path):
        script = self._console_script()
        if script is None:
            pytest.skip("console script not installed in this environment")
        root = _hermetic_root(tmp_path)
        env = _hermetic_env(root)
        env.pop("STAGEA_REPORT", None)
        completed = subprocess.run(
            [str(script), "-z", "--resume"], input=_request_bytes(), env=env,
            cwd=str(root / "cwd"), capture_output=True, timeout=600)
        assert completed.returncode != 0
        assert completed.stdout == b""
        assert b"argv.not_frozen_tail" in completed.stderr
