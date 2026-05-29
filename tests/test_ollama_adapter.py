from unittest.mock import Mock, call, patch

import pytest

import adapters.ollama_adapter as adapter
from core.settings import Settings


@pytest.fixture(autouse=True)
def _clear_cache():
    adapter.clear_model_cache()
    yield
    adapter.clear_model_cache()


def _fake_listing(names):
    return {"models": [{"name": n} for n in names]}


def test_exact_match_via_show():
    with patch.object(adapter.ollama, "show", return_value={"details": {}}):
        ok, note = adapter.ensure_model_available("gemma4:latest")
    assert ok is True
    assert note is None


def test_exact_match_via_list():
    with patch.object(adapter.ollama, "show", side_effect=Exception("no")):
        with patch.object(
            adapter.ollama,
            "list",
            return_value=_fake_listing(["gemma4:latest", "llama3.2-vision"]),
        ):
            ok, note = adapter.ensure_model_available("gemma4:latest")
    assert ok is True
    assert note is None


def test_tag_fallback_warns():
    with patch.object(adapter.ollama, "show", side_effect=Exception("no")):
        with patch.object(
            adapter.ollama,
            "list",
            return_value=_fake_listing(["gemma4:e4b"]),
        ):
            ok, note = adapter.ensure_model_available("gemma4:latest")
    assert ok is True
    assert note is not None and "gemma4:e4b" in note


def test_resolve_model_name_returns_matching_tag():
    with patch.object(adapter.ollama, "show", side_effect=Exception("no")):
        with patch.object(
            adapter.ollama,
            "list",
            return_value=_fake_listing(["gemma4:e4b"]),
        ):
            ok, resolved, note = adapter.resolve_model_name("gemma4:latest")
    assert ok is True
    assert resolved == "gemma4:e4b"
    assert note is not None and "gemma4:e4b" in note


def test_missing_model_reports_candidates():
    with patch.object(adapter.ollama, "show", side_effect=Exception("no")):
        with patch.object(
            adapter.ollama,
            "list",
            return_value=_fake_listing(["llama3.2-vision", "granite3.2-vision"]),
        ):
            ok, note = adapter.ensure_model_available("gemma4:latest")
    assert ok is False
    assert note is not None
    assert "gemma4" in note
    assert "llama3.2-vision" in note


def test_list_models_cache_ttl():
    calls = {"n": 0}

    def _fake_list():
        calls["n"] += 1
        return _fake_listing(["m1"])

    with patch.object(adapter.ollama, "list", side_effect=_fake_list):
        adapter.list_models()
        adapter.list_models()
        adapter.list_models()
    assert calls["n"] == 1
    adapter.clear_model_cache()
    with patch.object(adapter.ollama, "list", side_effect=_fake_list):
        adapter.list_models()
    assert calls["n"] == 2


def test_list_models_with_status_distinguishes_empty_success_from_failure():
    with patch.object(adapter.ollama, "list", return_value=_fake_listing([])):
        inventory = adapter.list_models_with_status()

    assert inventory.models == []
    assert inventory.reachable is True

    adapter.clear_model_cache()
    with patch.object(adapter.ollama, "list", side_effect=Exception("down")):
        inventory = adapter.list_models_with_status()

    assert inventory.models == []
    assert inventory.reachable is False


def test_list_models_uses_settings_ttl_for_cache_expiry(monkeypatch):
    settings = Settings(model_list_ttl=1.0, request_timeout=None)
    calls = {"n": 0}

    def _fake_list():
        calls["n"] += 1
        return _fake_listing([f"m{calls['n']}"])

    clock = iter([10.0, 10.5, 11.1])
    monkeypatch.setattr(adapter.time, "monotonic", lambda: next(clock))

    with patch.object(adapter.ollama, "list", side_effect=_fake_list):
        assert adapter.list_models(settings=settings) == _fake_listing(["m1"])["models"]
        assert adapter.list_models(settings=settings) == _fake_listing(["m1"])["models"]
        assert adapter.list_models(settings=settings) == _fake_listing(["m2"])["models"]

    assert calls["n"] == 2


def test_list_show_and_chat_use_settings_request_timeout():
    settings = Settings(request_timeout=9.5)
    client = Mock()
    client.list.return_value = _fake_listing(["gemma4:latest"])
    client.show.return_value = {"details": {}}
    client.chat.return_value = {"message": {"content": "done"}}

    with patch.object(adapter.ollama, "Client", return_value=client) as client_factory:
        assert adapter.list_models(settings=settings) == _fake_listing(["gemma4:latest"])["models"]
        ok, resolved, note = adapter.resolve_model_name("gemma4:latest", settings=settings)
        content = adapter.query_ollama("prompt", "img", "gemma4:latest", settings=settings)

    assert (ok, resolved, note) == (True, "gemma4:latest", None)
    assert content == "done"
    assert client_factory.call_args_list == [
        call(timeout=9.5),
        call(timeout=9.5),
        call(timeout=9.5),
    ]
    client.list.assert_called_once_with()
    client.show.assert_called_once_with("gemma4:latest")
    client.chat.assert_called_once()


def test_query_ollama_forwards_schema_dict_format():
    schema = {
        "type": "object",
        "properties": {"total": {"type": "string"}},
        "required": ["total"],
    }
    client = Mock()
    client.chat.return_value = {"message": {"content": '{"total": "42"}'}}

    with patch.object(adapter.ollama, "Client", return_value=client):
        content = adapter.query_ollama(
            "prompt",
            "img",
            "gemma4:latest",
            format=schema,
            timeout=1.0,
        )

    assert content == '{"total": "42"}'
    assert client.chat.call_args.kwargs["format"] is schema


def test_query_ollama_stream_uses_shared_messages_and_timeout():
    client = Mock()
    client.chat.return_value = [
        {"message": {"content": "hello"}},
        {"message": {"content": ""}},
        {"message": {"content": " world"}},
    ]

    with patch.object(adapter.ollama, "Client", return_value=client) as client_factory:
        chunks = list(
            adapter.query_ollama_stream(
                "prompt",
                "img",
                "gemma4:latest",
                system_prompt="system",
                timeout=2.0,
            )
        )

    assert chunks == ["hello", " world"]
    client_factory.assert_called_once_with(timeout=2.0)
    assert client.chat.call_args.kwargs["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "prompt", "images": ["img"]},
    ]
    assert client.chat.call_args.kwargs["stream"] is True


def test_substring_no_longer_matches():
    """Regression: 'gemma4' should NOT match 'gemma4:26b' without explicit tag."""
    with patch.object(adapter.ollama, "show", side_effect=Exception("no")):
        with patch.object(
            adapter.ollama,
            "list",
            return_value=_fake_listing(["gemma4:26b"]),
        ):
            # target 'gemma4:latest' has different base-tag than candidate,
            # but base() of 'gemma4' (no tag) should match 'gemma4:26b' base.
            ok, note = adapter.ensure_model_available("gemma4")
    assert ok is True  # base match
    assert note and "gemma4:26b" in note
