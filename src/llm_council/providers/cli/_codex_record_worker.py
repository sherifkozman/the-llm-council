"""Download the public Codex compatibility record from the fixed Council origin."""

import http.client
import ssl
import sys

_HOST = "raw.githubusercontent.com"
_PATH = "/sherifkozman/the-llm-council/main/compatibility/codex-v1.json"
_MAX_BYTES = 256 * 1024


def fetch_record(ca_bundle: str) -> bytes:
    connection = http.client.HTTPSConnection(
        _HOST, context=ssl.create_default_context(cafile=ca_bundle), timeout=5.0
    )
    try:
        connection.request(
            "GET", _PATH, headers={"Accept": "application/json", "Accept-Encoding": "identity"}
        )
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("record source returned an unexpected status")
        encoding = response.getheader("Content-Encoding")
        if encoding is not None and encoding.lower() != "identity":
            raise ValueError("encoded record is not supported")
        length = response.getheader("Content-Length")
        if length is not None and (
            not length.isascii() or not length.isdecimal() or int(length) > _MAX_BYTES
        ):
            raise ValueError("record size is invalid")
        data = response.read(_MAX_BYTES + 1)
        if len(data) > _MAX_BYTES or (length is not None and len(data) != int(length)):
            raise ValueError("record size does not match response")
        data.decode("utf-8", errors="strict")
        return data
    finally:
        connection.close()


if __name__ == "__main__":
    try:
        sys.stdout.buffer.write(fetch_record(sys.argv[1]))
    except (IndexError, OSError, ValueError, http.client.HTTPException):
        print("Codex compatibility record unavailable", file=sys.stderr)
        sys.exit(1)
