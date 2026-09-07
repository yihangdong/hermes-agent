"""DYHANO Stage-A text-only profile: the source constants and the prompt seam.

``HERMES_STAGEA_TEXT_ONLY_V1=1`` makes a bare ``hermes -z`` send one fixed
system message, no tools and ``max_tokens=8192``; anywhere else it is refused.
These tests pin the constants the AI-Org controller derives its expected
request body from, the opt-in grammar, the bare-``-z`` refusal helpers and the
system-prompt seam -- including that ordinary Hermes is untouched when the
opt-in is absent.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import stagea_text_only as profile
from agent.system_prompt import build_system_prompt, build_system_prompt_parts


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
        _emit_status=lambda *_args, **_kwargs: None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestConstants:
    def test_opt_in_name_and_value(self):
        assert profile.OPT_IN_ENV == "HERMES_STAGEA_TEXT_ONLY_V1"
        assert profile.OPT_IN_VALUE == "1"

    def test_opt_in_name_is_not_credential_shaped(self):
        lowered = profile.OPT_IN_ENV.lower()
        for token in ("api", "key", "token", "secret", "session", "password",
                      "credential", "auth", "bearer", "cookie"):
            assert token not in lowered
        assert not profile.OPT_IN_ENV.endswith(("_KEY", "_TOKEN", "_SECRET"))
        assert all(c.isupper() or c.isdigit() or c == "_" for c in profile.OPT_IN_ENV)

    def test_max_tokens_is_the_fixed_output_bound(self):
        assert profile.MAX_TOKENS == 8192
        assert type(profile.MAX_TOKENS) is int

    def test_system_prompt_is_already_in_post_sanitization_form(self):
        prompt = profile.SYSTEM_PROMPT
        assert type(prompt) is str and prompt
        assert prompt.isascii()
        assert prompt == prompt.strip()
        assert "\t" not in prompt and "\r" not in prompt
        assert prompt.encode("ascii").decode("ascii") == prompt
        assert all(c == "\n" or 0x20 <= ord(c) <= 0x7E for c in prompt)

    def test_system_prompt_names_the_framing_and_the_request_schema(self):
        prompt = profile.SYSTEM_PROMPT
        assert '"DYHANO-STAGE-A-PROPOSAL-V1 "' in prompt
        assert "dyhano-stage-a-proposal-request-v1" in prompt
        assert "admitted_write_set" in prompt
        assert "max_envelope_bytes" in prompt
        assert "no tools" in prompt


class TestOptInGrammar:
    def test_absent_or_empty_is_ordinary_hermes(self):
        assert profile.opt_in_state({}) == profile.DISABLED
        assert profile.opt_in_state({profile.OPT_IN_ENV: ""}) == profile.DISABLED
        assert profile.is_enabled({}) is False

    def test_exactly_one_enables(self):
        assert profile.opt_in_state({profile.OPT_IN_ENV: "1"}) == profile.ENABLED
        assert profile.is_enabled({profile.OPT_IN_ENV: "1"}) is True

    @pytest.mark.parametrize("value", ["true", "0", " 1", "1 ", "yes", "11", "TRUE"])
    def test_any_other_value_is_refused_not_guessed(self, value):
        with pytest.raises(profile.StageATextOnlyRefused):
            profile.opt_in_state({profile.OPT_IN_ENV: value})

    def test_reads_the_process_environment_by_default(self, monkeypatch):
        monkeypatch.delenv(profile.OPT_IN_ENV, raising=False)
        assert profile.is_enabled() is False
        monkeypatch.setenv(profile.OPT_IN_ENV, "1")
        assert profile.is_enabled() is True


class TestBareOneshotRefusal:
    ENABLED = {profile.OPT_IN_ENV: "1"}

    @pytest.mark.parametrize("argv", [["-z"], ["--oneshot"]])
    def test_bare_oneshot_argv_is_admitted(self, argv):
        profile.refuse_unless_bare_oneshot_argv(argv, self.ENABLED)

    @pytest.mark.parametrize("argv", [
        [], ["chat"], ["-z", "PROMPT"], ["--oneshot=PROMPT"], ["--oneshot", "PROMPT"],
        ["-p", "work", "-z"], ["-z", "-m", "x"], ["gateway", "run"], ["-z", "-z"],
        ["chat", "-z"], ["--version"], ["-Z"],
    ])
    def test_every_other_argv_is_refused(self, argv):
        with pytest.raises(profile.StageATextOnlyRefused) as exc:
            profile.refuse_unless_bare_oneshot_argv(argv, self.ENABLED)
        assert profile.OPT_IN_ENV in str(exc.value)

    @pytest.mark.parametrize("argv", [[], ["chat"], ["-z", "PROMPT"], ["gateway", "run"]])
    def test_argv_guard_is_a_no_op_without_the_opt_in(self, argv):
        profile.refuse_unless_bare_oneshot_argv(argv, {})
        profile.refuse_unless_bare_oneshot_argv(argv, {profile.OPT_IN_ENV: ""})

    def test_bare_call_is_admitted(self):
        profile.refuse_unless_bare_oneshot_call(True, environ=self.ENABLED)
        profile.refuse_unless_bare_oneshot_call(
            True, model=None, provider="", toolsets=[], skills=(), usage_file=None,
            environ=self.ENABLED)

    def test_argv_prompt_is_refused(self):
        with pytest.raises(profile.StageATextOnlyRefused):
            profile.refuse_unless_bare_oneshot_call(False, environ=self.ENABLED)

    @pytest.mark.parametrize("override", [
        {"model": "glm-4.6"}, {"provider": "zai"}, {"toolsets": "web"},
        {"toolsets": ["web"]}, {"skills": ["x"]}, {"usage_file": "usage.json"},
    ])
    def test_every_override_is_refused(self, override):
        with pytest.raises(profile.StageATextOnlyRefused) as exc:
            profile.refuse_unless_bare_oneshot_call(True, environ=self.ENABLED, **override)
        assert next(iter(override)).replace("_", "-") in str(exc.value)
        with pytest.raises(profile.StageATextOnlyRefused):
            profile.refuse_overrides(environ=self.ENABLED, **override)

    def test_call_guards_are_no_ops_without_the_opt_in(self):
        profile.refuse_unless_bare_oneshot_call(False, model="x", environ={})
        profile.refuse_overrides(model="x", provider="y", toolsets="z", environ={})


class TestSystemPromptSeam:
    def test_the_constant_is_the_whole_prompt_under_the_opt_in(self, monkeypatch):
        monkeypatch.setenv(profile.OPT_IN_ENV, "1")
        agent = _make_agent()
        forbidden = AssertionError("environmental prompt input must not be read")
        with (
            patch("run_agent.load_soul_md", side_effect=forbidden),
            patch("run_agent.build_environment_hints", side_effect=forbidden),
            patch("run_agent.build_context_files_prompt", side_effect=forbidden),
        ):
            parts = build_system_prompt_parts(agent)
            joined = build_system_prompt(agent)
        assert parts == {"stable": profile.SYSTEM_PROMPT, "context": "", "volatile": ""}
        assert joined == profile.SYSTEM_PROMPT
        assert agent._cached_system_prompt_static == profile.SYSTEM_PROMPT

    def test_a_caller_system_message_is_refused_under_the_opt_in(self, monkeypatch):
        monkeypatch.setenv(profile.OPT_IN_ENV, "1")
        with pytest.raises(profile.StageATextOnlyRefused):
            build_system_prompt_parts(_make_agent(), system_message="extra")
        with pytest.raises(profile.StageATextOnlyRefused):
            build_system_prompt(_make_agent(), system_message="extra")

    def test_a_malformed_opt_in_is_refused_at_the_prompt_seam(self, monkeypatch):
        monkeypatch.setenv(profile.OPT_IN_ENV, "true")
        with pytest.raises(profile.StageATextOnlyRefused):
            build_system_prompt_parts(_make_agent())

    def test_the_ordinary_prompt_path_is_unchanged_without_the_opt_in(self, monkeypatch):
        monkeypatch.delenv(profile.OPT_IN_ENV, raising=False)
        agent = _make_agent()
        with (
            patch("run_agent.load_soul_md", return_value=""),
            patch("run_agent.build_environment_hints", return_value=""),
            patch("run_agent.build_context_files_prompt",
                  return_value="CONTEXT-FILES-MARK-7f2c") as context_files,
        ):
            parts = build_system_prompt_parts(agent)
            joined = build_system_prompt(agent)
        assert context_files.called
        assert "CONTEXT-FILES-MARK-7f2c" in joined
        assert profile.SYSTEM_PROMPT not in joined
        assert parts["stable"] and parts["stable"] != profile.SYSTEM_PROMPT
