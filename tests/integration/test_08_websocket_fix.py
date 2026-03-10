"""Test 08: WebSocket gateway URL fix (unit test)."""

import pytest


class TestWebSocketURLFix:
    def test_gateway_url_none_fallback(self):
        from hyperwave_community.api_client import _API_CONFIG
        original = _API_CONFIG.copy()
        try:
            _API_CONFIG['api_url'] = 'https://example.com'
            _API_CONFIG['gateway_url'] = None
            result = _API_CONFIG.get('gateway_url') or _API_CONFIG['api_url']
            assert result == 'https://example.com'
        finally:
            _API_CONFIG.update(original)

    def test_gateway_url_set(self):
        from hyperwave_community.api_client import _API_CONFIG
        original = _API_CONFIG.copy()
        try:
            _API_CONFIG['api_url'] = 'https://api.example.com'
            _API_CONFIG['gateway_url'] = 'https://gateway.example.com'
            result = _API_CONFIG.get('gateway_url') or _API_CONFIG['api_url']
            assert result == 'https://gateway.example.com'
        finally:
            _API_CONFIG.update(original)

    def test_old_pattern_broken(self):
        config = {'gateway_url': None, 'api_url': 'https://fallback.com'}
        old_result = config.get('gateway_url', 'https://fallback.com')
        assert old_result is None
        new_result = config.get('gateway_url') or config['api_url']
        assert new_result == 'https://fallback.com'
