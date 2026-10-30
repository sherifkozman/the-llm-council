"""Tests for the bounded, fixed-origin Codex compatibility record download."""

import asyncio
import importlib
import io
import ssl

import certifi
import pytest


def worker():
    return importlib.import_module("llm_council.providers.cli._codex_record_worker")


class Response:
    def __init__(self, body=b'{"schema":1}', status=200, headers=None):
        self.body = body
        self.status = status
        self.headers = headers or {}
        self.read_limit = None

    def getheader(self, name):
        return self.headers.get(name)

    def read(self, amount):
        self.read_limit = amount
        return self.body[:amount]


class Connection:
    response = None
    request_args = None
    options = None

    def __init__(self, host, **kwargs):
        type(self).options = host, kwargs

    def request(self, *args, **kwargs):
        type(self).request_args = args, kwargs

    def getresponse(self):
        return self.response

    def close(self):
        pass


def test_fetch_uses_fixed_verified_tls_origin_and_bounded_body(monkeypatch):
    module = worker()
    response = Response()
    Connection.response = response
    monkeypatch.setattr(module.http.client, "HTTPSConnection", Connection)
    assert module.fetch_record(certifi.where()) == b'{"schema":1}'
    host, options = Connection.options
    assert host == "raw.githubusercontent.com"
    assert options["context"].verify_mode == ssl.CERT_REQUIRED
    assert options["context"].check_hostname
    assert options["timeout"] > 0
    args, kwargs = Connection.request_args
    assert args == ("GET", "/sherifkozman/the-llm-council/main/compatibility/codex-v1.json")
    assert kwargs["headers"] == {"Accept": "application/json", "Accept-Encoding": "identity"}
    assert response.read_limit == 256 * 1024 + 1


@pytest.mark.parametrize(
    "response",
    [
        Response(status=302, headers={"Location": "https://evil.example/record"}),
        Response(status=404),
        Response(body=b"a" * (256 * 1024 + 1)),
        Response(headers={"Content-Length": str(256 * 1024 + 1)}),
        Response(headers={"Content-Length": "garbage"}),
        Response(headers={"Content-Encoding": "gzip"}),
        Response(body=b"\xff"),
    ],
)
def test_fetch_rejects_untrusted_response(monkeypatch, response):
    module = worker()
    Connection.response = response
    monkeypatch.setattr(module.http.client, "HTTPSConnection", Connection)
    with pytest.raises((ValueError, UnicodeDecodeError)):
        module.fetch_record(certifi.where())


@pytest.mark.asyncio
async def test_fetch_launcher_uses_only_sanitized_environment(monkeypatch, tmp_path):
    module = importlib.import_module("llm_council.providers.cli.codex")
    provider = module.CodexCLIProvider(cli_path="synthetic")
    calls = []

    async def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return 0, module._LiveCodexState(stdout_parts=['{"schema":1}'])

    monkeypatch.setattr(provider, "_run_cli", fake_run)
    deadline = asyncio.get_running_loop().time() + 5
    result = await provider._fetch_compatibility_record(deadline=deadline)
    assert result == b'{"schema":1}'
    cmd, kwargs = calls[0]
    assert cmd[1] == "-I"
    assert cmd[2].endswith("/_codex_record_worker.py")
    assert cmd[3] == certifi.where()
    assert kwargs["env"] == {"PATH": module.os.defpath}
    assert kwargs["deadline"] == deadline


@pytest.mark.asyncio
async def test_fetch_launcher_fails_closed_on_worker_failure(monkeypatch):
    module = importlib.import_module("llm_council.providers.cli.codex")
    provider = module.CodexCLIProvider(cli_path="synthetic")

    async def fake_run(cmd, **kwargs):
        return 1, module._LiveCodexState()

    monkeypatch.setattr(provider, "_run_cli", fake_run)
    assert (
        await provider._fetch_compatibility_record(deadline=asyncio.get_running_loop().time() + 5)
        is None
    )


def test_fetch_rejects_partial_http_body_even_if_it_is_valid_json(monkeypatch):
    module = worker()

    class FakeSocket:
        def makefile(self, *args, **kwargs):
            return io.BytesIO(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{}")

    response = module.http.client.HTTPResponse(FakeSocket())
    response.begin()
    Connection.response = response
    monkeypatch.setattr(module.http.client, "HTTPSConnection", Connection)
    with pytest.raises(ValueError, match="size"):
        module.fetch_record(certifi.where())


@pytest.mark.asyncio
async def test_fetch_launcher_bounds_its_own_deadline_and_rejects_oversized_output(monkeypatch):
    module = importlib.import_module("llm_council.providers.cli.codex")
    provider = module.CodexCLIProvider(cli_path="synthetic")
    observed = {}

    async def fake_run(cmd, **kwargs):
        observed.update(kwargs)
        return 0, module._LiveCodexState(stdout_parts=["x" * (256 * 1024 + 1)])

    monkeypatch.setattr(provider, "_run_cli", fake_run)
    start = asyncio.get_running_loop().time()
    assert await provider._fetch_compatibility_record(deadline=start + 100) is None
    assert start <= observed["deadline"] <= start + 5.1


def test_record_worker_can_start_without_site_packages(tmp_path):
    """The isolated child must not need a user-site certifi import."""
    import subprocess
    import sys

    module = worker()
    result = subprocess.run(
        [sys.executable, "-I", "-S", module.__file__, str(tmp_path / "missing-ca.pem")],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 1
    assert result.stderr.strip() == "Codex compatibility record unavailable"
