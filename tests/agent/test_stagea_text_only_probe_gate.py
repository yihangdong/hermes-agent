"""Stage-A text-only opt-in: no endpoint metadata probes.

Under ``HERMES_STAGEA_TEXT_ONLY_V1=1`` the configured endpoint is the
controller's loopback relay, which admits exactly one chat-completion
exchange and nothing else.  ``agent.model_metadata`` therefore skips its
three network probe entry points (local-server detection, local
context-length query, OpenAI-compatible ``/models`` fetch) under the opt-in
and probes exactly as before when the opt-in is absent.
"""

import httpx
import pytest

from agent import model_metadata, stagea_text_only

LOOPBACK = "http://127.0.0.1:18731/v1"


class _NoNetworkClient:
    """Stands in for ``httpx.Client``; any construction fails the test."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("network probe attempted under the opt-in")


class _RecordingClient:
    """An ``httpx.Client`` double that records every probe and refuses it."""

    calls = []

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, **kwargs):
        _RecordingClient.calls.append(url)
        raise httpx.ConnectError("connection refused")

    def post(self, url, **kwargs):
        _RecordingClient.calls.append(url)
        raise httpx.ConnectError("connection refused")


def _no_ensure_requests():
    raise AssertionError("requests-based probe attempted under the opt-in")


@pytest.fixture
def opt_in(monkeypatch):
    monkeypatch.setenv(stagea_text_only.OPT_IN_ENV, stagea_text_only.OPT_IN_VALUE)
    monkeypatch.setattr(httpx, "Client", _NoNetworkClient)
    monkeypatch.setattr(model_metadata, "_ensure_requests", _no_ensure_requests)


@pytest.fixture
def opt_out(monkeypatch):
    monkeypatch.delenv(stagea_text_only.OPT_IN_ENV, raising=False)
    _RecordingClient.calls = []
    monkeypatch.setattr(httpx, "Client", _RecordingClient)


# -- the predicate ------------------------------------------------------------


def test_the_gate_is_exactly_the_opt_in_predicate(monkeypatch):
    monkeypatch.delenv(stagea_text_only.OPT_IN_ENV, raising=False)
    assert model_metadata._stagea_text_only_probe_suppressed() is False
    monkeypatch.setenv(stagea_text_only.OPT_IN_ENV, "")
    assert model_metadata._stagea_text_only_probe_suppressed() is False
    monkeypatch.setenv(stagea_text_only.OPT_IN_ENV, stagea_text_only.OPT_IN_VALUE)
    assert model_metadata._stagea_text_only_probe_suppressed() is True
    monkeypatch.setenv(stagea_text_only.OPT_IN_ENV, "yes")
    with pytest.raises(stagea_text_only.StageATextOnlyRefused):
        model_metadata._stagea_text_only_probe_suppressed()


# -- under the opt-in: no probe touches the network --------------------------


def test_opt_in_suppresses_local_server_detection(opt_in):
    assert model_metadata.detect_local_server_type(LOOPBACK, api_key="x") is None


def test_opt_in_suppresses_the_local_context_length_query(opt_in):
    assert model_metadata._query_local_context_length(
        "glm-4.6", LOOPBACK, api_key="x") is None


def test_opt_in_suppresses_the_endpoint_models_fetch(opt_in):
    assert model_metadata.fetch_endpoint_model_metadata(
        LOOPBACK, api_key="x", force_refresh=True) == {}


def test_opt_in_suppresses_the_endpoint_context_resolution(opt_in):
    assert model_metadata._resolve_endpoint_context_length(
        "glm-4.6", LOOPBACK, api_key="x") is None


# -- without the opt-in: ordinary Hermes probes exactly as before -------------


def test_without_the_opt_in_local_server_detection_still_probes(opt_out):
    # A base URL no other test uses, so the process-lifetime detection cache
    # cannot answer for the waterfall.
    result = model_metadata.detect_local_server_type(
        "http://127.0.0.1:18741/v1", api_key="x")
    assert result is None
    assert _RecordingClient.calls, "the waterfall was not attempted"
    assert _RecordingClient.calls[0].endswith("/api/v1/models")


def test_without_the_opt_in_the_local_context_query_still_probes(opt_out):
    model_metadata._query_local_context_length(
        "glm-4.6", "http://127.0.0.1:18742/v1", api_key="x")
    assert _RecordingClient.calls, "the local context probe was not attempted"


def test_without_the_opt_in_the_endpoint_models_fetch_still_probes(monkeypatch):
    monkeypatch.delenv(stagea_text_only.OPT_IN_ENV, raising=False)
    # Bind the real ``requests`` module the lazy import would bind, then
    # replace only its ``get`` for this test: a module-level placeholder
    # would outlive the test and defeat the lazy binding for later suites.
    real_requests = model_metadata._ensure_requests()
    calls = []

    def refused_get(url, **kwargs):
        calls.append(url)
        raise real_requests.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr(real_requests, "get", refused_get)
    result = model_metadata.fetch_endpoint_model_metadata(
        "http://127.0.0.1:18743/v1", api_key="x", force_refresh=True)
    assert result == {}
    assert calls, "the /models fetch was not attempted"
    assert calls[0].endswith("/models")


# -- the direct Ollama probes and the auxiliary title call --------------------


def test_opt_in_suppresses_the_ollama_api_show_probes(opt_in):
    assert model_metadata._query_ollama_api_show(
        "glm-4.6", LOOPBACK, api_key="x") is None
    assert model_metadata.query_ollama_num_ctx(
        "glm-4.6", LOOPBACK, api_key="x") is None
    assert model_metadata.query_ollama_supports_vision(
        "glm-4.6", LOOPBACK, api_key="x") is None


def test_without_the_opt_in_the_ollama_api_show_probe_still_posts(opt_out):
    model_metadata._query_ollama_api_show(
        "glm-4.6", "http://127.0.0.1:18744/v1", api_key="x")
    assert _RecordingClient.calls, "the /api/show probe was not attempted"
    assert _RecordingClient.calls[0].endswith("/api/show")


def test_opt_in_disables_auxiliary_title_generation(monkeypatch):
    from agent import title_generator

    monkeypatch.setenv(stagea_text_only.OPT_IN_ENV, stagea_text_only.OPT_IN_VALUE)
    assert title_generator._auto_title_enabled() is False

    class _UntouchableSessionDb:
        def __getattr__(self, name):
            raise AssertionError("session store touched under the opt-in: %s" % name)

    # Opening turn, titleable message: without the opt-in this would write an
    # instant title and fork the model upgrade; under it nothing happens.
    assert title_generator.maybe_auto_title(
        _UntouchableSessionDb(), "session-1",
        "Please summarize the quarterly report for the board meeting",
        conversation_history=[]) is None


def test_without_the_opt_in_title_generation_keeps_its_default(monkeypatch):
    from agent import title_generator

    monkeypatch.delenv(stagea_text_only.OPT_IN_ENV, raising=False)
    assert title_generator._auto_title_enabled() is True
