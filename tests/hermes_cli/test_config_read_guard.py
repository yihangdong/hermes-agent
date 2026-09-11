"""Lint guard: no new raw yaml.safe_load(config.yaml) reads outside owner modules.

The drift class this kills: scattered ``yaml.safe_load`` reads of the user's
``config.yaml`` silently miss the managed-scope overlay, ``${ENV_VAR}``
expansion, profile-aware pathing, and root-model normalization. Each new
config feature has historically required an N-site sweep (incident chain:
9cbcc0c9c8 → 732293cf87 → b0e47a98f9 → 1928aa0443).

Canonical owners:

  * ``hermes_cli/config.py`` — ``load_config()`` / ``load_config_readonly()``
    (merged + managed + env-expanded), ``read_raw_config()`` and
    ``read_user_config_raw()`` (the ONLY legal raw primitives: write-back
    round-trips + raw-file diagnostics).
  * ``gateway/config.py`` — the gateway's ``load_gateway_config`` owner.
  * ``gateway/run.py`` — ``_load_gateway_config()``'s monkeypatched-home
    fallback path (delegates to ``read_raw_config`` when paths agree).
  * ``stagea_config_owner.py`` — cold-safe Stage-A bounded literal bytes only;
    structurally policed below, never a generic configuration loader.

Everything else must import one of those. If this test fails on your new
code, use ``load_config()``/``load_config_readonly()`` for behavioral reads,
or ``read_user_config_raw()`` for write-back round-trips — do not add your
file to the allowlist without a reason of the same class.
"""

from __future__ import annotations

import os
import re
import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Files where a yaml.safe_load near a config.yaml reference is legal.
# Keep this list SHORT and justified:
ALLOWLIST = {
    # Canonical loader owners.
    "hermes_cli/config.py",
    "gateway/config.py",
    # Dedicated cold-safe bounded-bytes Stage-A owner, structurally policed.
    "stagea_config_owner.py",
    # _load_gateway_config()'s fallback path for tests that monkeypatch
    # gateway.run._hermes_home (delegates to read_raw_config otherwise).
    "gateway/run.py",
    # Reads the MANAGED-scope config.yaml (/etc/hermes/...), not the user's —
    # it IS the overlay source; the canonical loaders call into it.
    "hermes_cli/managed_scope.py",
    # Parse-health probe: intentionally answers "does the raw file parse?".
    "gateway/readiness.py",
}

# Directories that never count (tests may build fixture configs freely).
EXCLUDED_DIR_PARTS = {
    "tests", ".venv", ".git", ".worktrees", "node_modules", "website",
    "docs", "scripts", "examples", "apps",
    # Compiled bytecode is not source. Sibling test processes also create
    # and delete these directories while this scan walks the tree.
    "__pycache__",
}

# A safe_load within this many lines of a config.yaml reference is treated
# as a raw user-config read.
PROXIMITY = 6

SAFE_LOAD_RE = re.compile(r"\bsafe_load\s*\(")
CONFIG_YAML_RE = re.compile(r"""["']config\.yaml["']""")


def _iter_source_files():
    # This uses os.walk with a pruned dirnames, and not rglob. rglob descends
    # into every directory and filters after that, so it calls scandir() on
    # __pycache__ trees that this guard never inspects. Sibling test processes
    # create and delete those entries during the run.
    #
    # A directory that disappears in the middle of a walk raises
    # FileNotFoundError out of rglob. The test then fails for a reason that it
    # does not assert.
    #
    # The prune skips those trees. The onerror callback ignores a directory
    # that disappears anyway.
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT, onerror=lambda _e: None):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIR_PARTS]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = Path(dirpath) / name
            rel = path.relative_to(REPO_ROOT)
            if any(part in EXCLUDED_DIR_PARTS for part in rel.parts):
                continue
            yield rel, path


def test_no_raw_config_yaml_reads_outside_owner_modules():
    offenders: list[str] = []
    for rel, path in _iter_source_files():
        rel_str = str(rel).replace("\\", "/")
        if rel_str in ALLOWLIST:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        cfg_lines = [i for i, ln in enumerate(lines) if CONFIG_YAML_RE.search(ln)]
        if not cfg_lines:
            continue
        for i, ln in enumerate(lines):
            if not SAFE_LOAD_RE.search(ln):
                continue
            # Comment/docstring mentions don't count.
            stripped = ln.strip()
            if stripped.startswith("#"):
                continue
            if any(abs(i - j) <= PROXIMITY for j in cfg_lines):
                offenders.append(f"{rel_str}:{i + 1}: {stripped}")

    assert not offenders, (
        "Raw yaml.safe_load of config.yaml outside allowlisted owner modules.\n"
        "Behavioral reads must use hermes_cli.config.load_config()/"
        "load_config_readonly() (or gateway _load_gateway_config); write-back "
        "round-trips and raw-file diagnostics must use "
        "hermes_cli.config.read_user_config_raw().\nOffenders:\n  "
        + "\n  ".join(offenders)
    )


def test_read_user_config_raw_exists_and_documented():
    """The shared raw primitive must exist and carry its legality docstring."""
    from hermes_cli.config import read_user_config_raw

    doc = read_user_config_raw.__doc__ or ""
    assert "ONLY legal for write-back round-trips and raw-file diagnostics" in doc
    assert "load_config()" in doc


def _assert_stagea_owner_structure(source):
    """Closed AST capability surface: YAML and exact-bytes parsing only."""
    tree = ast.parse(source)
    declarations = [n for n in tree.body if not (
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
        and isinstance(n.value.value, str))]
    assert len(declarations) == 2
    imports, parser = declarations
    assert isinstance(imports, ast.Import)
    assert [(a.name, a.asname) for a in imports.names] == [("yaml", None)]
    assert isinstance(parser, ast.FunctionDef)
    assert parser.name == "parse_stagea_task_config_bytes"
    assert not parser.decorator_list and parser.returns is None
    assert len(parser.args.args) == 1 and parser.args.args[0].arg == "raw"
    assert ast.dump(parser.args.args[0].annotation) == "Name(id='bytes', ctx=Load())"
    assert not (parser.args.posonlyargs or parser.args.kwonlyargs
                or parser.args.defaults or parser.args.kw_defaults
                or parser.args.vararg or parser.args.kwarg)
    allowed = (ast.Module, ast.Expr, ast.Constant, ast.Import, ast.alias,
               ast.FunctionDef, ast.arguments, ast.arg, ast.If, ast.Compare,
               ast.Call, ast.Name, ast.Load, ast.IsNot, ast.Raise, ast.Return,
               ast.Attribute)
    assert all(isinstance(n, allowed) for n in ast.walk(tree))
    calls = [n for n in ast.walk(parser) if isinstance(n, ast.Call)]
    assert len(calls) == 3
    names = []
    for call in calls:
        assert not call.keywords and len(call.args) == 1
        if isinstance(call.func, ast.Name):
            assert call.func.id in {"type", "TypeError"}
            names.append(call.func.id)
        else:
            assert isinstance(call.func, ast.Attribute)
            assert isinstance(call.func.value, ast.Name)
            assert (call.func.value.id, call.func.attr) == ("yaml", "safe_load")
            assert isinstance(call.args[0], ast.Name) and call.args[0].id == "raw"
            names.append("yaml.safe_load")
    assert sorted(names) == ["TypeError", "type", "yaml.safe_load"]
    assert {n.id for n in ast.walk(parser) if isinstance(n, ast.Name)} <= {
        "raw", "bytes", "type", "TypeError", "yaml"}


def test_stagea_owner_has_only_cold_safe_parser_capabilities():
    _assert_stagea_owner_structure(
        (REPO_ROOT / "stagea_config_owner.py").read_text(encoding="utf-8"))


@pytest.mark.parametrize("addition", [
    "import os", "from hermes_cli import config", "import providers",
    "import hermes_cli.plugins", "open('unexpected', 'w')",
    "__import__('os')", "import os\nos.environ.get('HOME')",
])
def test_stagea_owner_structural_guard_rejects_new_capabilities(addition):
    source = (REPO_ROOT / "stagea_config_owner.py").read_text(encoding="utf-8")
    with pytest.raises(AssertionError):
        _assert_stagea_owner_structure(source + "\n" + addition + "\n")


def test_stagea_owner_is_consumed_only_by_its_two_admitted_surfaces():
    consumers = set()
    for rel, path in _iter_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.ImportFrom)
                    and node.module == "stagea_config_owner") or (
                    isinstance(node, ast.Import)
                    and any(a.name == "stagea_config_owner" for a in node.names)):
                consumers.add(rel.as_posix())
    assert consumers == {"agent/stagea_proposal_only.py", "hermes_cli/config.py"}
