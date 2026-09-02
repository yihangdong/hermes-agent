"""Bare ``-z``/``--oneshot`` reads exactly one EOF-driven prompt from stdin.

``hermes -z "PROMPT"`` keeps working unchanged; giving ``-z`` no argv value
now means "the prompt is on stdin". The tests below pin the whole contract,
including the parts that are only interesting when they *fail*:

* stdin that is a terminal, empty, whitespace-only, undecodable or over
  262144 bytes is refused **before** the one-shot runner is imported, so no
  request is ever sent to a provider;
* an oversized prompt is refused whole — never truncated and sent anyway;
* the prompt travels in the process's stdin, never in ``argv``;
* nothing re-execs, shells out, or consults ``PATH`` to deliver it, and no
  session/history/provider/auth/config behaviour changes.

The last group is deliberately adversarial about the *other* stdin surface
this release already has: ``hermes chat --query-file -`` also reads stdin,
but it is a different flag on a different subparser with different
semantics, and it does not satisfy this contract. Those tests exist so a
future reader cannot mistake one for the other.
"""

import argparse
import builtins
import os
import subprocess
import sys

import pytest

from hermes_cli._parser import (
    ONESHOT_PROMPT_FROM_STDIN,
    build_top_level_parser,
    top_level_value_flag_sets,
)
import hermes_cli.main as main_mod


LIMIT = main_mod._ONESHOT_STDIN_MAX_BYTES


def _parse(argv):
    return build_top_level_parser()[0].parse_args(argv)


class _FakeStdin:
    """Minimal stand-in for a non-TTY ``sys.stdin`` backed by real bytes."""

    def __init__(self, data: bytes, tty: bool = False):
        import io

        self.buffer = io.BytesIO(data)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class _Exited(Exception):
    """Raised in place of the real ``os._exit`` so tests can observe rc."""

    def __init__(self, rc):
        super().__init__(rc)
        self.rc = rc


@pytest.fixture
def oneshot_spy(monkeypatch):
    """Capture what reaches ``run_oneshot`` without running an agent.

    Also neuters the hard ``os._exit`` at the end of one-shot dispatch so the
    test process survives to make assertions.
    """
    import hermes_cli.oneshot as oneshot_mod

    calls = []

    def fake_run_oneshot(prompt, **kwargs):
        calls.append({"prompt": prompt, "kwargs": kwargs})
        return 0

    monkeypatch.setattr(oneshot_mod, "run_oneshot", fake_run_oneshot)
    monkeypatch.setattr(main_mod, "_cleanup_oneshot_runtime", lambda: None)

    def fake_exit(rc):
        raise _Exited(rc)

    monkeypatch.setattr(main_mod, "_exit_after_oneshot", fake_exit)
    return calls


# ---------------------------------------------------------------------------
# existing `-z PROMPT` behaviour is untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["-z", "PROMPT"], "PROMPT"),
        (["--oneshot", "PROMPT"], "PROMPT"),
        (["--oneshot=PROMPT"], "PROMPT"),
        (["-z", "a prompt with spaces"], "a prompt with spaces"),
        (["-m", "x", "-z", "PROMPT"], "PROMPT"),
    ],
)
def test_argv_prompt_forms_are_unchanged(argv, expected):
    args = _parse(argv)
    assert args.oneshot == expected
    assert args.oneshot is not ONESHOT_PROMPT_FROM_STDIN


def test_argv_prompt_never_consults_stdin(oneshot_spy, monkeypatch):
    def exploding_stdin_read(*_a, **_k):
        raise AssertionError("an argv prompt must not read stdin")

    monkeypatch.setattr(main_mod, "_read_oneshot_prompt_from_stdin", exploding_stdin_read)

    with pytest.raises(_Exited):
        main_mod._run_and_exit_oneshot("PROMPT")

    assert [c["prompt"] for c in oneshot_spy] == ["PROMPT"]


def test_explicit_empty_argv_prompt_keeps_its_pre_existing_falsy_behaviour():
    """``hermes -z ""`` stays an empty string, not the stdin sentinel.

    It was falsy before this change and is falsy after, so the dispatch
    sites keep falling through to chat exactly as they always did. Bare
    ``-z`` is the *absence* of a value, which is a different thing.
    """
    args = _parse(["-z", ""])
    assert args.oneshot == ""
    assert args.oneshot is not ONESHOT_PROMPT_FROM_STDIN
    assert not args.oneshot


# ---------------------------------------------------------------------------
# bare `-z` selects the stdin transport
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [["-z"], ["--oneshot"], ["-m", "x", "-z"]])
def test_bare_flag_selects_the_stdin_sentinel(argv):
    assert _parse(argv).oneshot is ONESHOT_PROMPT_FROM_STDIN


def test_bare_flag_stays_truthy_for_every_dispatch_site():
    """All three one-shot dispatch sites gate on ``if args.oneshot``.

    A falsy marker would silently fall through to interactive chat, so the
    sentinel's truthiness is load-bearing, not cosmetic.
    """
    assert bool(_parse(["-z"]).oneshot) is True


def test_bare_flag_does_not_swallow_the_next_flag():
    args = _parse(["-z", "--usage-file", "/tmp/u.json"])
    assert args.oneshot is ONESHOT_PROMPT_FROM_STDIN
    assert args.usage_file == "/tmp/u.json"


def test_stdin_prompt_reaches_the_runner_verbatim_exactly_once(oneshot_spy, monkeypatch):
    monkeypatch.setattr(sys, "stdin", _FakeStdin(b"  summarize this repo \n\n"))

    with pytest.raises(_Exited):
        main_mod._run_and_exit_oneshot(ONESHOT_PROMPT_FROM_STDIN)

    assert len(oneshot_spy) == 1, "exactly one prompt must reach the runner"
    # Verbatim: not stripped, not normalized, not re-wrapped.
    assert oneshot_spy[0]["prompt"] == "  summarize this repo \n\n"


def test_stdin_prompt_reaches_the_same_runner_with_the_same_options(oneshot_spy, monkeypatch):
    """The stdin path is the existing one-shot path, not a parallel one."""
    monkeypatch.setattr(sys, "stdin", _FakeStdin(b"hello"))

    with pytest.raises(_Exited):
        main_mod._run_and_exit_oneshot(
            ONESHOT_PROMPT_FROM_STDIN,
            model="m",
            provider="p",
            toolsets="t",
            skills="s",
            usage_file="/tmp/u.json",
        )

    kwargs = oneshot_spy[0]["kwargs"]
    # No session/resume/history/retry/fallback key is introduced.
    assert set(kwargs) == {"model", "provider", "toolsets", "skills", "usage_file"}
    assert kwargs == {
        "model": "m",
        "provider": "p",
        "toolsets": "t",
        "skills": "s",
        "usage_file": "/tmp/u.json",
    }


def test_exactly_limit_bytes_is_accepted(oneshot_spy, monkeypatch):
    monkeypatch.setattr(sys, "stdin", _FakeStdin(b"a" * LIMIT))

    with pytest.raises(_Exited):
        main_mod._run_and_exit_oneshot(ONESHOT_PROMPT_FROM_STDIN)

    assert len(oneshot_spy[0]["prompt"]) == LIMIT


def test_multibyte_prompt_is_measured_in_bytes_not_characters(oneshot_spy, monkeypatch):
    """The cap is a byte cap: 2-byte characters get half as many of them."""
    monkeypatch.setattr(sys, "stdin", _FakeStdin("é".encode() * (LIMIT // 2)))

    with pytest.raises(_Exited):
        main_mod._run_and_exit_oneshot(ONESHOT_PROMPT_FROM_STDIN)

    assert oneshot_spy[0]["prompt"] == "é" * (LIMIT // 2)


# ---------------------------------------------------------------------------
# fail-closed: every rejection happens before any provider work
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, stdin",
    [
        ("terminal", _FakeStdin(b"typed by a human", tty=True)),
        ("empty", _FakeStdin(b"")),
        ("whitespace only", _FakeStdin(b" \t\r\n  ")),
        ("undecodable", _FakeStdin(b"good \xff\xfe bad")),
        ("one byte over the limit", _FakeStdin(b"a" * (LIMIT + 1))),
        ("far over the limit", _FakeStdin(b"a" * (LIMIT * 4))),
    ],
)
def test_invalid_stdin_fails_before_the_model(label, stdin, oneshot_spy, monkeypatch):
    monkeypatch.setattr(sys, "stdin", stdin)

    with pytest.raises(SystemExit) as excinfo:
        main_mod._run_and_exit_oneshot(ONESHOT_PROMPT_FROM_STDIN)

    assert excinfo.value.code == 2, f"{label}: rejections are usage errors"
    assert oneshot_spy == [], f"{label}: nothing may reach the model"


def test_oversized_stdin_is_refused_whole_never_truncated(oneshot_spy, monkeypatch):
    """A truncated prompt would be a wrong answer wearing a right answer's face."""
    monkeypatch.setattr(sys, "stdin", _FakeStdin(b"x" * (LIMIT + 1)))

    with pytest.raises(SystemExit):
        main_mod._run_and_exit_oneshot(ONESHOT_PROMPT_FROM_STDIN)

    assert oneshot_spy == []


def test_rejection_precedes_importing_the_oneshot_runner(monkeypatch, capsys):
    """Fail-closed means the provider path is never even loaded."""
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "hermes_cli.oneshot":
            raise AssertionError(
                "the one-shot runner was imported for a rejected prompt"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(sys, "stdin", _FakeStdin(b"   "))
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(SystemExit) as excinfo:
        main_mod._run_and_exit_oneshot(ONESHOT_PROMPT_FROM_STDIN)

    assert excinfo.value.code == 2
    assert "hermes -z:" in capsys.readouterr().err


def test_missing_stdin_is_refused(monkeypatch):
    monkeypatch.setattr(sys, "stdin", None)

    with pytest.raises(SystemExit) as excinfo:
        main_mod._read_oneshot_prompt_from_stdin()

    assert excinfo.value.code == 2


def test_unreadable_stdin_is_refused(monkeypatch):
    class Unreadable:
        buffer = None

        def isatty(self):
            return False

        def read(self, _n=-1):
            raise OSError("stream closed")

    with pytest.raises(SystemExit) as excinfo:
        main_mod._read_oneshot_prompt_from_stdin(Unreadable())

    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# the prompt is not an argument, and nothing is re-executed to deliver it
# ---------------------------------------------------------------------------


def test_prompt_bytes_never_appear_in_argv(oneshot_spy, monkeypatch):
    secret_shaped_prompt = "correct-horse-battery-staple-9f3a7c"
    monkeypatch.setattr(sys, "argv", ["hermes", "-z"])
    monkeypatch.setattr(sys, "stdin", _FakeStdin(secret_shaped_prompt.encode()))

    args = _parse(sys.argv[1:])
    with pytest.raises(_Exited):
        main_mod._run_and_exit_oneshot(args.oneshot)

    assert oneshot_spy[0]["prompt"] == secret_shaped_prompt
    assert all(secret_shaped_prompt not in token for token in sys.argv)
    assert secret_shaped_prompt not in " ".join(sys.argv)


def test_no_wrapper_reexec_or_shell_delivers_the_prompt(oneshot_spy, monkeypatch):
    """No subprocess, exec or PATH lookup may sit on the stdin path."""

    def forbidden(name):
        def _boom(*_args, **_kwargs):
            raise AssertionError(f"bare -z must not call {name}")

        return _boom

    monkeypatch.setattr(subprocess, "Popen", forbidden("subprocess.Popen"))
    monkeypatch.setattr(subprocess, "run", forbidden("subprocess.run"))
    monkeypatch.setattr(os, "system", forbidden("os.system"))
    for exec_name in ("execv", "execve", "execvp", "posix_spawn"):
        if hasattr(os, exec_name):
            monkeypatch.setattr(os, exec_name, forbidden(f"os.{exec_name}"))
    monkeypatch.setattr(sys, "stdin", _FakeStdin(b"no wrapper please"))

    with pytest.raises(_Exited):
        main_mod._run_and_exit_oneshot(ONESHOT_PROMPT_FROM_STDIN)

    assert oneshot_spy[0]["prompt"] == "no wrapper please"


# ---------------------------------------------------------------------------
# argv-scanner classification stays honest
# ---------------------------------------------------------------------------


def test_oneshot_is_classified_as_an_optional_value_flag():
    required, optional = top_level_value_flag_sets()
    assert {"-z", "--oneshot"} <= optional
    assert not ({"-z", "--oneshot"} & required)


def test_static_fallback_snapshot_matches_the_live_parser():
    """The snapshot is only read when parser introspection breaks — so it has
    to stay true, and nothing else would notice if it drifted."""
    from hermes_cli._parser import (
        _OPTIONAL_VALUE_FLAGS_FALLBACK,
        _VALUE_FLAGS_FALLBACK,
    )

    required, optional = top_level_value_flag_sets()
    assert {"-z", "--oneshot"} <= _OPTIONAL_VALUE_FLAGS_FALLBACK
    assert not ({"-z", "--oneshot"} & _VALUE_FLAGS_FALLBACK)
    assert _VALUE_FLAGS_FALLBACK <= required
    assert _OPTIONAL_VALUE_FLAGS_FALLBACK <= optional


# ---------------------------------------------------------------------------
# `chat --query-file -` is a different surface and does not satisfy this contract
# ---------------------------------------------------------------------------


def test_query_file_is_not_a_top_level_flag():
    """``hermes --query-file -`` is not the bare-z transport and never was."""
    parser = build_top_level_parser()[0]
    top_level_options = {
        opt for action in parser._actions for opt in action.option_strings
    }
    assert "--query-file" not in top_level_options

    with pytest.raises(SystemExit):
        parser.parse_args(["--query-file", "-"])


def test_chat_query_file_does_not_enter_the_top_level_oneshot_path():
    args = _parse(["chat", "--query-file", "-"])
    assert args.query_file == "-"
    # This is the exact predicate every one-shot dispatch site uses.
    assert not getattr(args, "oneshot", None)


def test_chat_oneshot_flag_is_a_separate_boolean_surface():
    """``chat --oneshot`` is a bool on ``oneshot_exit``; ``-z`` carries a prompt.

    Sharing a dest would feed ``True`` to the runner as the prompt text.
    """
    args = _parse(["chat", "--query-file", "-", "--oneshot"])
    assert args.oneshot_exit is True
    assert not getattr(args, "oneshot", None)

    bare = _parse(["-z"])
    assert bare.oneshot is ONESHOT_PROMPT_FROM_STDIN
    assert getattr(bare, "oneshot_exit", False) is False


def test_bare_z_enforces_the_guarantees_query_file_does_not():
    """The two stdin readers are independent and are not interchangeable.

    ``chat --query-file -`` decodes with ``errors="replace"``, enforces no
    byte cap and has no TTY guard. Those are acceptable for seeding a chat
    session, and they are exactly the three properties bare ``-z`` must not
    inherit — so ``--query-file -`` can never be offered as evidence that
    the bare-z contract is satisfied.
    """
    assert LIMIT == 262144

    # TTY guard — query-file has none.
    with pytest.raises(SystemExit):
        main_mod._read_oneshot_prompt_from_stdin(_FakeStdin(b"x", tty=True))
    # Strict UTF-8 — query-file replaces undecodable bytes instead.
    with pytest.raises(SystemExit):
        main_mod._read_oneshot_prompt_from_stdin(_FakeStdin(b"\xff"))
    # Byte cap — query-file is unbounded.
    with pytest.raises(SystemExit):
        main_mod._read_oneshot_prompt_from_stdin(_FakeStdin(b"a" * (LIMIT + 1)))
