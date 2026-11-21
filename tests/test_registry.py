"""Tests for tracking-URI resolution (src/registry.py)."""

from src.registry import DEFAULT_TRACKING_URI, is_remote_tracking_uri, resolve_tracking_uri

CONFIG = {"mlflow": {"tracking_uri": "sqlite:///from_config.db"}}


def test_config_uri_is_used_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert resolve_tracking_uri(CONFIG) == "sqlite:///from_config.db"
    assert resolve_tracking_uri({}) == DEFAULT_TRACKING_URI


def test_env_overrides_config(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    assert resolve_tracking_uri(CONFIG) == "http://127.0.0.1:5000"


def test_remote_uris():
    assert is_remote_tracking_uri("http://127.0.0.1:5000")
    assert is_remote_tracking_uri("https://mlflow.example.com")
    assert not is_remote_tracking_uri("sqlite:///mlflow_tracking.db")
