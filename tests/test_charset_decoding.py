"""
Tests for decoding responses that declare no charset.

requests decodes text/* without a charset as ISO-8859-1, so a UTF-8 page
served as plain `text/html` (common on Cloudflare Workers) came back garbled:
Arabic became mojibake and "m²" became "mÂ²". decoded_text() detects the
charset from the bytes instead, and keeps any charset the server declares.
"""

import sys
import os
from unittest.mock import patch

import pytest
import requests

# Ensure scripts/ is importable from the worktree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import citability_scorer  # noqa: E402
import fetch_page  # noqa: E402
import llmstxt_generator  # noqa: E402

SAMPLE = "ماكينات تعبئة وتغليف القهوة: 20–70 bags/min, 15,000+ m²"


def _make_response(body: bytes, content_type: str) -> requests.Response:
    """Build a real requests.Response decoded the way requests decodes it."""
    resp = requests.models.Response()
    resp._content = body
    resp.status_code = 200
    resp.url = "https://example.com/"
    resp.headers["Content-Type"] = content_type
    resp.encoding = requests.utils.get_encoding_from_headers(resp.headers)
    return resp


@pytest.mark.parametrize("module", [fetch_page, citability_scorer, llmstxt_generator])
class TestDecodedText:
    def test_utf8_without_charset_is_not_garbled(self, module):
        html = f"<html><body><p>{SAMPLE}</p></body></html>".encode("utf-8")
        resp = _make_response(html, "text/html")
        assert SAMPLE not in resp.text  # the requests default this guards against
        assert SAMPLE in module.decoded_text(resp)

    def test_declared_charset_is_kept(self, module):
        resp = _make_response("café".encode("iso-8859-1"), "text/plain; charset=ISO-8859-1")
        assert module.decoded_text(resp) == "café"


class TestFetchLlmsTxtDecoding:
    def test_llms_txt_without_charset_keeps_utf8(self):
        body = f"# Example\n\n> {SAMPLE}\n".encode("utf-8")
        with patch("fetch_page.requests.get", return_value=_make_response(body, "text/plain")):
            result = fetch_page.fetch_llms_txt("https://example.com/")
        assert SAMPLE in result["llms_txt"]["content"]
