"""Bare ``hermes -z`` under ``HERMES_STAGEA_TEXT_ONLY_V1=1``: the one-shot seams.

* the single one-shot dispatch funnel refuses an argv prompt or any override
  with rc 2 before stdin is read and before the runner is imported;
* ``run_oneshot`` refuses overrides with rc 2 before an agent is built;
* ``_run_agent`` builds the agent with an empty toolset list,
  ``max_tokens=8192`` and no preloaded-skills prompt;
* the module-level guard in ``hermes_cli.main`` refuses every other command;
* without the opt-in each of those seams behaves exactly as before.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from agent import stagea_text_only as profile
from hermes_cli._parser import ONESHOT_PROMPT_FROM_STDIN
import hermes_cli.main as main_mod
import hermes_cli.oneshot as oneshot_mod


REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeStdin:
    def __init__(self, data: bytes, tty: bool = False):
        import io

        self.buffer = io.BytesIO(data)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class _Exited(Exception):
    def __init__(self, rc):
        super().__init__(rc)
        self.rc = rc


@pytest.fixture
def oneshot_spy(monkeypatch):
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


@pytest.fixture
def opt_in(monkeypatch):
    monkeypatch.setenv(profile.OPT_IN_ENV, "1")


@pytest.fixture
def no_opt_in(monkeypatch):
    monkeypatch.delenv(profile.OPT_IN_ENV, raising=False)


def _exploding_stdin(*_a, **_k):
    raise AssertionError("stdin must not be read")


# ---------------------------------------------------------------------------
# the single dispatch funnel
# ---------------------------------------------------------------------------


def test_funnel_refuses_an_argv_prompt_under_the_opt_in(oneshot_spy, opt_in, monkeypatch, capsys):
    monkeypatch.setattr(main_mod, "_read_oneshot_prompt_from_stdin", _exploding_stdin)
    with pytest.raises(_Exited) as exc:
        main_mod._run_and_exit_oneshot("PROMPT")
    assert exc.value.rc == 2
    assert oneshot_spy == []
    assert profile.OPT_IN_ENV in capsys.readouterr().err


@pytest.mark.parametrize("override", [
    {"model": "glm-4.6"}, {"provider": "zai"}, {"toolsets": "web"},
    {"toolsets": ["web"]}, {"skills": ["x"]}, {"usage_file": "usage.json"},
])
def test_funnel_refuses_every_override_under_the_opt_in(oneshot_spy, opt_in, monkeypatch, capsys, override):
    monkeypatch.setattr(main_mod, "_read_oneshot_prompt_from_stdin", _exploding_stdin)
    with pytest.raises(_Exited) as exc:
        main_mod._run_and_exit_oneshot(ONESHOT_PROMPT_FROM_STDIN, **override)
    assert exc.value.rc == 2
    assert oneshot_spy == []
    assert profile.OPT_IN_ENV in capsys.readouterr().err


def test_funnel_admits_the_bare_stdin_prompt_under_the_opt_in(oneshot_spy, opt_in, monkeypatch):
    monkeypatch.setattr(sys, "stdin", _FakeStdin(b'{"schema_version":"x"}'))
    with pytest.raises(_Exited) as exc:
        main_mod._run_and_exit_oneshot(ONESHOT_PROMPT_FROM_STDIN)
    assert exc.value.rc == 0
    assert oneshot_spy == [{
        "prompt": '{"schema_version":"x"}',
        "kwargs": {"model": None, "provider": None, "toolsets": None,
                   "skills": None, "usage_file": None},
    }]


def test_funnel_is_unchanged_without_the_opt_in(oneshot_spy, no_opt_in, monkeypatch):
    monkeypatch.setattr(main_mod, "_read_oneshot_prompt_from_stdin", _exploding_stdin)
    with pytest.raises(_Exited) as exc:
        main_mod._run_and_exit_oneshot("PROMPT", model="glm-4.6", toolsets="web")
    assert exc.value.rc == 0
    assert oneshot_spy == [{
        "prompt": "PROMPT",
        "kwargs": {"model": "glm-4.6", "provider": None, "toolsets": "web",
                   "skills": None, "usage_file": None},
    }]


# ---------------------------------------------------------------------------
# run_oneshot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("override", [
    {"model": "glm-4.6"}, {"provider": "zai"}, {"toolsets": "web"},
    {"skills": ["x"]}, {"usage_file": "usage.json"},
])
def test_run_oneshot_refuses_overrides_before_building_an_agent(opt_in, monkeypatch, capsys, override):
    def no_agent(*_a, **_k):
        raise AssertionError("no agent may be built")

    monkeypatch.setattr(oneshot_mod, "_run_agent", no_agent)
    assert oneshot_mod.run_oneshot("PROMPT", **override) == 2
    assert profile.OPT_IN_ENV in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _run_agent
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_seams(monkeypatch):
    """Fake every collaborator of ``_run_agent`` and capture the AIAgent kwargs."""
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def run_conversation(self, prompt):
            captured["prompt"] = prompt
            return {"final_response": "ok", "completed": True}

        def shutdown_memory_provider(self, *_a, **_k):
            pass

        def close(self):
            pass

    import run_agent
    from tools.process_registry import process_registry

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: {})
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **k: {
            "api_key": "not-a-credential", "base_url": "http://127.0.0.1:9/v1",
            "provider": "zai", "requested_provider": "zai",
            "api_mode": "chat_completions", "credential_pool": None,
        },
    )
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools",
                        lambda cfg, platform: {"web", "terminal"})
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
                        lambda **k: None)
    monkeypatch.setattr("hermes_cli.models.detect_provider_for_model", lambda *a, **k: None)
    monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr(oneshot_mod, "get_fallback_chain", lambda cfg: [])
    monkeypatch.setattr(oneshot_mod, "_build_preloaded_skills_prompt",
                        lambda skills: "SKILLS-MARK" if skills else None)
    monkeypatch.setattr(process_registry, "wait_for_pending_completions", lambda *a, **k: None)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    return captured


def test_run_agent_builds_a_text_only_agent_under_the_opt_in(agent_seams, opt_in):
    response, result = oneshot_mod._run_agent("PROMPT")
    kwargs = agent_seams["kwargs"]
    assert kwargs["enabled_toolsets"] == []
    assert kwargs["max_tokens"] == profile.MAX_TOKENS == 8192
    assert kwargs["ephemeral_system_prompt"] is None
    assert kwargs["platform"] == "cli"
    assert kwargs["quiet_mode"] is True
    assert agent_seams["prompt"] == "PROMPT"
    assert response == "ok"


def test_run_agent_profile_wins_over_direct_toolset_and_skill_arguments(agent_seams, opt_in):
    oneshot_mod._run_agent("PROMPT", toolsets=["web"], skills=["x"])
    kwargs = agent_seams["kwargs"]
    assert kwargs["enabled_toolsets"] == []
    assert kwargs["ephemeral_system_prompt"] is None
    assert kwargs["max_tokens"] == 8192


def test_run_agent_keeps_the_configured_cli_toolsets_without_the_opt_in(agent_seams, no_opt_in):
    oneshot_mod._run_agent("PROMPT")
    kwargs = agent_seams["kwargs"]
    assert kwargs["enabled_toolsets"] == ["terminal", "web"]
    assert kwargs["max_tokens"] is None
    assert kwargs["ephemeral_system_prompt"] is None


def test_run_agent_keeps_explicit_toolsets_and_skills_without_the_opt_in(agent_seams, no_opt_in):
    oneshot_mod._run_agent("PROMPT", toolsets=["web"], skills=["x"])
    kwargs = agent_seams["kwargs"]
    assert kwargs["enabled_toolsets"] == ["web"]
    assert kwargs["ephemeral_system_prompt"] == "SKILLS-MARK"
    assert kwargs["max_tokens"] is None


# ---------------------------------------------------------------------------
# the module-level guard: no other command may run under the opt-in
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("argv, expected_rc", [
    (["-z"], 0),
    (["--oneshot"], 0),
    (["chat"], 2),
    (["-z", "PROMPT"], 2),
    ([], 2),
    # An EXISTING profile: the CLI strips ``-p work`` from ``sys.argv``
    # before the guard runs, so this case proves the guard decides on the
    # launch vector rather than on the stripped one.
    (["-p", "work", "-z"], 2),
    (["gateway", "run"], 2),
])
def test_importing_the_cli_under_the_opt_in_admits_only_a_bare_oneshot(argv, expected_rc, tmp_path):
    (tmp_path / "profiles" / "work").mkdir(parents=True)
    program = textwrap.dedent(
        f"""
        import os, sys
        os.environ["HERMES_HOME"] = {str(tmp_path)!r}
        os.environ["HERMES_STAGEA_TEXT_ONLY_V1"] = "1"
        sys.argv = ["hermes", *{argv!r}]
        import hermes_cli.main
        print("IMPORTED")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=REPO_ROOT,
        capture_output=True, timeout=180, check=False,
    )
    assert result.returncode == expected_rc, result.stderr.decode("utf-8", "replace")
    if expected_rc == 0:
        assert b"IMPORTED" in result.stdout
    else:
        assert b"IMPORTED" not in result.stdout
        assert b"HERMES_STAGEA_TEXT_ONLY_V1" in result.stderr


def test_importing_the_cli_without_the_opt_in_is_unchanged():
    program = textwrap.dedent(
        """
        import os, sys
        os.environ.pop("HERMES_STAGEA_TEXT_ONLY_V1", None)
        sys.argv = ["hermes", "chat"]
        import hermes_cli.main
        print("IMPORTED")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=REPO_ROOT,
        capture_output=True, timeout=180, check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert b"IMPORTED" in result.stdout
