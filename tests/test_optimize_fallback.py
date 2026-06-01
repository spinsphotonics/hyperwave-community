"""Tests for WebSocket fallback logic in optimize().

The optimize() function tries the unified /pipeline_optimize_ws endpoint
first. If that fails with a connection error (OSError, WebSocketException,
ConnectionError, TimeoutError), it falls back to a 2-step POST+WS flow:
  1. POST /pipeline_optimize_start -> session_id
  2. WS   /inverse_design_ws?session_id=<id>

These tests mock the WebSocket and HTTP layers to verify all paths.
"""
from __future__ import annotations

import json
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import websocket as _real_ws_module


# ---------------------------------------------------------------------------
# Helper: mock WebSocket
# ---------------------------------------------------------------------------

class MockWebSocket:
    """Fake websocket.WebSocket that replays a list of JSON messages."""

    def __init__(self, messages: List[dict]):
        self._messages = list(messages)
        self._sent: list = []
        self._closed = False

    def recv(self):
        if not self._messages:
            raise Exception("No more messages")
        return json.dumps(self._messages.pop(0))

    def send(self, data):
        self._sent.append(data)

    def send_binary(self, data):
        self._sent.append(data)

    def settimeout(self, t):
        pass

    def close(self):
        self._closed = True


# ---------------------------------------------------------------------------
# Minimal valid inputs for optimize()
# ---------------------------------------------------------------------------

_LAYERS = [
    {"name": "box", "thickness": 2.0, "index": 1.44},
    {"name": "etch", "thickness": 0.22, "index": 3.47, "design": True,
     "density_radius": 4, "initial_value": 0.5},
    {"name": "clad", "thickness": 2.0, "index": 1.44},
]
_NX, _NY = 20, 20
_THETA = {"etch": np.full((_NX, _NY), 0.5, dtype=np.float32)}
_GRID = 0.035
_WAVELENGTH = 1.55

# Minimal source / mode arrays -- shape (1, 6, 1, ny, 1)
_SOURCE = np.zeros((1, 6, 1, _NY, 1), dtype=np.complex64)
_SOURCE[0, 1, :, :, :] = 1.0  # Ey
_SOURCE[0, 5, :, :, :] = 1.0  # Hz

_MODE = np.zeros((1, 6, 1, _NY, 1), dtype=np.complex64)
_MODE[0, 1, :, :, :] = 1.0
_MODE[0, 5, :, :, :] = 1.0

# Properly encoded b64 theta for step messages (must be decodable)
def _encode_theta():
    from hyperwave_community.api_client import encode_array
    return encode_array(np.full((_NX, _NY), 0.5, dtype=np.float32))

_THETA_B64 = _encode_theta()

# Standard messages for a successful run
_STARTED_MSG = {"type": "started"}
_STEP_MSG = {"type": "step", "step": 1, "efficiency": 0.5, "loss": 0.1,
             "theta_b64": _THETA_B64, "step_time": 1.0}
_DONE_MSG = {"type": "done"}


# ---------------------------------------------------------------------------
# Fixture: patch _API_CONFIG and _build_device_from_specs
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_config_and_device():
    """Set deterministic _API_CONFIG and stub _build_device_from_specs."""
    fake_config = {
        "api_key": "test-key-123",
        "api_url": "https://api.example.com",
        "gateway_url": "https://gateway.example.com",
    }

    # Minimal DeviceConfig-like object returned by _build_device_from_specs
    fake_device = MagicMock()
    fake_device.design_layers_info = [
        {"name": "etch", "theta": np.full((_NX, _NY), 0.5, dtype=np.float32),
         "density_radius": 4, "eps_range": [1.44**2, 3.47**2],
         "z_range": [10, 20]},
    ]
    fake_device.freq_band = [0.1, 0.1, 1]
    fake_device.recipe_params = {"grid_shape": [_NX, _NY]}
    fake_device.shape = [_NX, _NY, 40]
    fake_device.grid = _GRID

    fake_abs = {"absorption_widths": (10, 5, 5), "abs_coeff": 0.001}

    with patch("hyperwave_community.api_client._API_CONFIG", new=fake_config), \
         patch("hyperwave_community.device._build_device_from_specs",
               return_value=fake_device), \
         patch("hyperwave_community.absorption.absorber_params",
               return_value=fake_abs):
        yield


def _call_optimize(**kwargs):
    """Call optimize() with minimal valid arguments, merging any overrides."""
    from hyperwave_community.pipeline import optimize
    defaults = dict(
        layers=_LAYERS,
        theta=_THETA,
        grid=_GRID,
        wavelength=_WAVELENGTH,
        source=_SOURCE,
        mode=_MODE,
        n_steps=1,
    )
    defaults.update(kwargs)
    return optimize(**defaults)


# ===================================================================
# 1. Unified WS succeeds (happy path)
# ===================================================================

@patch("requests.post")
@patch("websocket.create_connection")
def test_unified_ws_happy_path(mock_create_conn, mock_post):
    """When unified WS works, fallback POST is never called."""
    ws = MockWebSocket([_STARTED_MSG, _STEP_MSG, _DONE_MSG])
    mock_create_conn.return_value = ws

    result = _call_optimize()

    # Only one create_connection call (the unified WS)
    assert mock_create_conn.call_count == 1
    url_arg = mock_create_conn.call_args[0][0]
    assert "/pipeline_optimize_ws" in url_arg

    # POST never called
    mock_post.assert_not_called()

    # Result sanity
    assert len(result.history) == 1
    assert result.history[0]["efficiency"] == 0.5
    assert result.best_efficiency == 0.5


# ===================================================================
# 2. Unified WS fails with TimeoutError, fallback succeeds
# ===================================================================

@patch("websocket.create_connection")
def test_fallback_on_timeout(mock_create_conn):
    """TimeoutError triggers fallback; POST + second WS succeed."""
    fallback_ws = MockWebSocket([_STEP_MSG, _DONE_MSG])

    # First call: unified WS fails. Second call: fallback WS succeeds.
    mock_create_conn.side_effect = [
        TimeoutError("connect timed out"),
        fallback_ws,
    ]

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"session_id": "test-session-id"}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = _call_optimize()

    # Two create_connection calls
    assert mock_create_conn.call_count == 2

    # POST was called with /pipeline_optimize_start
    assert mock_post.call_count == 1
    post_url = mock_post.call_args[0][0]
    assert "/pipeline_optimize_start" in post_url

    # Fallback WS URL contains session_id
    fb_url = mock_create_conn.call_args_list[1][0][0]
    assert "/inverse_design_ws" in fb_url
    assert "session_id=test-session-id" in fb_url

    assert len(result.history) == 1


# ===================================================================
# 3. Unified WS fails with OSError, fallback succeeds
# ===================================================================

@patch("websocket.create_connection")
def test_fallback_on_oserror(mock_create_conn):
    """OSError('Connection refused') triggers fallback."""
    fallback_ws = MockWebSocket([_STEP_MSG, _DONE_MSG])
    mock_create_conn.side_effect = [
        OSError("Connection refused"),
        fallback_ws,
    ]

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"session_id": "sess-abc"}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = _call_optimize()

    assert mock_create_conn.call_count == 2
    assert mock_post.call_count == 1
    assert len(result.history) == 1


# ===================================================================
# 4. Unified WS fails, gateway POST fails, api_url POST succeeds
# ===================================================================

@patch("websocket.create_connection")
def test_fallback_gateway_post_fails_api_succeeds(mock_create_conn):
    """Gateway POST raises ConnectionError, api_url POST succeeds."""
    fallback_ws = MockWebSocket([_STEP_MSG, _DONE_MSG])
    mock_create_conn.side_effect = [
        TimeoutError("ws timeout"),
        fallback_ws,
    ]

    mock_resp_ok = MagicMock()
    mock_resp_ok.json.return_value = {"session_id": "sess-api"}
    mock_resp_ok.raise_for_status = MagicMock()

    def _post_side_effect(url, **kwargs):
        if "gateway.example.com" in url:
            raise ConnectionError("gateway down")
        return mock_resp_ok

    with patch("requests.post", side_effect=_post_side_effect) as mock_post:
        result = _call_optimize()

    # Two POST calls: first to gateway (fails), second to api_url (succeeds)
    assert mock_post.call_count == 2
    first_post_url = mock_post.call_args_list[0][0][0]
    second_post_url = mock_post.call_args_list[1][0][0]
    assert "gateway.example.com" in first_post_url
    assert "api.example.com" in second_post_url

    # Fallback WS uses api_url base
    fb_url = mock_create_conn.call_args_list[1][0][0]
    assert "api.example.com" in fb_url
    assert "session_id=sess-api" in fb_url

    assert len(result.history) == 1


# ===================================================================
# 5. Unified WS fails, both POSTs fail -> RuntimeError
# ===================================================================

@patch("websocket.create_connection")
def test_both_endpoints_fail_raises(mock_create_conn):
    """When all connection attempts fail, RuntimeError is raised."""
    mock_create_conn.side_effect = OSError("ws refused")

    def _post_always_fails(url, **kwargs):
        raise ConnectionError(f"cannot reach {url}")

    with patch("requests.post", side_effect=_post_always_fails):
        with pytest.raises(RuntimeError, match="All endpoints failed"):
            _call_optimize()


# ===================================================================
# 6. Server returns error on unified WS (no fallback)
# ===================================================================

@patch("requests.post")
@patch("websocket.create_connection")
def test_server_error_no_fallback(mock_create_conn, mock_post):
    """A server error message should raise immediately, not trigger fallback."""
    error_msg = {"type": "error", "message": "bad request: invalid phase"}
    ws = MockWebSocket([error_msg])
    mock_create_conn.return_value = ws

    with pytest.raises(RuntimeError, match="bad request: invalid phase"):
        _call_optimize()

    # POST should NOT be called -- this is a server-level error, not a
    # connection error, so fallback is not appropriate.
    mock_post.assert_not_called()
    # Only one create_connection (the unified WS)
    assert mock_create_conn.call_count == 1


# ===================================================================
# 7. Cancellation via KeyboardInterrupt returns partial results
# ===================================================================

@patch("requests.post")
@patch("websocket.create_connection")
def test_keyboard_interrupt_partial_results(mock_create_conn, mock_post):
    """KeyboardInterrupt sends cancel and returns partial history."""
    step1 = {"type": "step", "step": 1, "efficiency": 0.3,
             "loss": 0.2, "theta_b64": _THETA_B64, "step_time": 1.0}
    step2 = {"type": "step", "step": 2, "efficiency": 0.4,
             "loss": 0.15, "theta_b64": _THETA_B64, "step_time": 1.0}

    class InterruptAfterTwoSteps(MockWebSocket):
        """WS that raises KeyboardInterrupt after yielding two steps."""
        def __init__(self):
            super().__init__([_STARTED_MSG, step1, step2])
            self._count = 0

        def recv(self):
            self._count += 1
            if self._count > 3:  # started + 2 steps, then interrupt
                raise KeyboardInterrupt()
            return super().recv()

    ws = InterruptAfterTwoSteps()
    mock_create_conn.return_value = ws

    result = _call_optimize(n_steps=10)

    # Should have 2 steps of history
    assert len(result.history) == 2
    assert result.history[0]["efficiency"] == 0.3
    assert result.history[1]["efficiency"] == 0.4
    assert result.best_efficiency == 0.4

    # Cancel message should have been sent
    cancel_msgs = [s for s in ws._sent if "cancel" in str(s)]
    assert len(cancel_msgs) >= 1

    # WS should be closed
    assert ws._closed


# ===================================================================
# 8. Heartbeat timeout after max silent heartbeats
# ===================================================================

@patch("websocket.create_connection")
def test_heartbeat_timeout_raises(mock_create_conn):
    """If ws.recv() times out 41 consecutive times, raise RuntimeError."""
    import websocket as _ws_lib

    class TimeoutWebSocket(MockWebSocket):
        def __init__(self):
            super().__init__([_STARTED_MSG])
            self._ack_sent = False

        def recv(self):
            if not self._ack_sent:
                self._ack_sent = True
                return super().recv()
            raise _ws_lib.WebSocketTimeoutException("timed out")

    ws = TimeoutWebSocket()
    mock_create_conn.return_value = ws

    with pytest.raises(RuntimeError, match="No response from GPU"):
        _call_optimize(n_steps=3)

    assert ws._closed


# ===================================================================
# 9. Heartbeat resets counter on real message
# ===================================================================

@patch("websocket.create_connection")
def test_heartbeat_resets_on_message(mock_create_conn):
    """Heartbeat counter resets on a real message. Without reset, 35+35=70 > 40 threshold."""
    import websocket as _ws_lib

    class TwoBatchTimeoutWebSocket(MockWebSocket):
        def __init__(self):
            super().__init__([_STARTED_MSG])
            self._ack_sent = False
            self._timeout_count = 0
            self._phase = 0  # 0=timeouts, 1=step, 2=timeouts, 3=done

        def recv(self):
            if not self._ack_sent:
                self._ack_sent = True
                return super().recv()
            self._timeout_count += 1
            if self._phase == 0 and self._timeout_count <= 35:
                raise _ws_lib.WebSocketTimeoutException("timed out")
            if self._phase == 0:
                self._phase = 1
                self._timeout_count = 0
                return json.dumps(_STEP_MSG)
            if self._phase == 1:
                self._phase = 2
            if self._phase == 2 and self._timeout_count <= 35:
                raise _ws_lib.WebSocketTimeoutException("timed out")
            return json.dumps(_DONE_MSG)

    ws = TwoBatchTimeoutWebSocket()
    mock_create_conn.return_value = ws

    result = _call_optimize(n_steps=1)
    assert len(result.history) == 1
    assert result.history[0]["efficiency"] == 0.5


# ===================================================================
# 10. Multi-layer theta_b64 dict in step messages
# ===================================================================

@patch("websocket.create_connection")
def test_multilayer_theta_b64_dict(mock_create_conn):
    """Step messages with theta_b64 as dict should decode all layers."""
    theta_b64_dict = {
        "etch": _THETA_B64,
        "slab": _THETA_B64,
    }
    step_msg = {"type": "step", "step": 1, "efficiency": 0.5,
                "loss": 0.1, "theta_b64": theta_b64_dict, "step_time": 1.0}

    ws = MockWebSocket([_STARTED_MSG, step_msg, _DONE_MSG])
    mock_create_conn.return_value = ws

    result = _call_optimize(n_steps=1)
    assert "etch" in result.design.thetas
    assert "slab" in result.design.thetas
    expected = np.full((_NX, _NY), 0.5, dtype=np.float32)
    np.testing.assert_array_equal(result.design.thetas["etch"], expected)
    np.testing.assert_array_equal(result.design.thetas["slab"], expected)


# ===================================================================
# 11. Startup ack is a transient "timed out" error -> retry, then succeed
# ===================================================================

@patch("requests.post")
@patch("websocket.create_connection")
def test_startup_timeout_retries_then_succeeds(mock_create_conn, mock_post):
    """A startup ack of {'type':'error','message':'...timed out'} is a transient
    gateway/cold-start timeout. optimize() should reconnect and retry the full
    connect+send+ack sequence (with backoff) rather than failing the call."""
    from hyperwave_community.pipeline import _MAX_STARTUP_RETRIES

    timeout_ack = {"type": "error", "message": "Request to Cloud Function timed out"}
    first_ws = MockWebSocket([dict(timeout_ack)])
    good_ws = MockWebSocket([_STARTED_MSG, _STEP_MSG, _DONE_MSG])
    mock_create_conn.side_effect = [first_ws, good_ws]

    # Patch sleep so the test does not actually wait for backoff.
    with patch("time.sleep") as mock_sleep:
        result = _call_optimize()

    # Reconnected once: two create_connection calls, both to the unified WS.
    assert mock_create_conn.call_count == 2
    assert "/pipeline_optimize_ws" in mock_create_conn.call_args_list[0][0][0]
    assert "/pipeline_optimize_ws" in mock_create_conn.call_args_list[1][0][0]

    # Backoff slept exactly once with the first exponential delay (2s), and the
    # timed-out socket was closed.
    from hyperwave_community.pipeline import _STARTUP_BACKOFF_BASE
    mock_sleep.assert_called_once_with(_STARTUP_BACKOFF_BASE)
    assert first_ws._closed

    # The original payload was resent verbatim, exactly once, on the retry.
    assert len(first_ws._sent) == 1
    assert len(good_ws._sent) == 1
    assert first_ws._sent == good_ws._sent

    # No POST fallback was used (this is a startup retry, not a connection error).
    mock_post.assert_not_called()

    # The retry produced a normal result.
    assert len(result.history) == 1
    assert result.history[0]["efficiency"] == 0.5
    assert _MAX_STARTUP_RETRIES >= 1


# ===================================================================
# 12. Startup timeout that never clears -> raise after exhausting retries
# ===================================================================

@patch("requests.post")
@patch("websocket.create_connection")
def test_startup_timeout_exhausts_retries_raises(mock_create_conn, mock_post):
    """If every startup attempt times out, optimize() must give up after
    _MAX_STARTUP_RETRIES retries and raise (not loop forever)."""
    from hyperwave_community.pipeline import _MAX_STARTUP_RETRIES

    timeout_ack = {"type": "error", "message": "Request to Cloud Function timed out"}
    # Enough timed-out sockets to cover every attempt (1 initial + N retries).
    mock_create_conn.side_effect = [
        MockWebSocket([dict(timeout_ack)]) for _ in range(_MAX_STARTUP_RETRIES + 1)
    ]

    from hyperwave_community.pipeline import _STARTUP_BACKOFF_BASE
    with patch("time.sleep") as mock_sleep:
        with pytest.raises(RuntimeError, match="timed out"):
            _call_optimize()

    # Exactly initial attempt + N retries.
    assert mock_create_conn.call_count == _MAX_STARTUP_RETRIES + 1
    # Backoff is exponential 2/4/8s: one sleep per retry, none after the final
    # failed attempt.
    expected = [_STARTUP_BACKOFF_BASE * (2 ** i) for i in range(_MAX_STARTUP_RETRIES)]
    assert [c.args[0] for c in mock_sleep.call_args_list] == expected
    # Startup retry never falls back to POST.
    mock_post.assert_not_called()


# ===================================================================
# 13. Genuine (non-timeout) server error still raises immediately, no retry
# ===================================================================

@patch("requests.post")
@patch("websocket.create_connection")
def test_genuine_server_error_does_not_retry(mock_create_conn, mock_post):
    """A non-timeout error (e.g. bad request) must NOT be retried -- it should
    raise immediately on the first attempt. Guards the retry from masking real
    server errors. (Complements test_server_error_no_fallback.)"""
    error_ack = {"type": "error", "message": "bad request: invalid phase"}
    mock_create_conn.side_effect = [
        MockWebSocket([dict(error_ack)]),
        MockWebSocket([_STARTED_MSG, _STEP_MSG, _DONE_MSG]),  # must never be used
    ]

    with patch("time.sleep") as mock_sleep:
        with pytest.raises(RuntimeError, match="bad request: invalid phase"):
            _call_optimize()

    # Only one connection attempt -- no retry, no backoff.
    assert mock_create_conn.call_count == 1
    mock_sleep.assert_not_called()
    mock_post.assert_not_called()


# ===================================================================
# 14. Transient PRE-DISPATCH siblings (not the literal "timed out") also retry
# ===================================================================

@pytest.mark.parametrize("transient_msg", [
    "Error calling Cloud Function: upstream request timeout",   # upstream 504 body
    "All connection attempts failed",                            # httpx ConnectError
    "4 DEADLINE_EXCEEDED: Deadline exceeded",                    # gRPC deadline
    "Server disconnected without sending a response.",           # httpx disconnect
    "Timed out waiting for request payload",                     # payload-receive timeout
])
@patch("requests.post")
@patch("websocket.create_connection")
def test_startup_retries_on_transient_siblings(mock_create_conn, mock_post, transient_msg):
    """Pre-dispatch transients other than the literal 'Request ... timed out'
    string are equally safe to retry (same pre-dispatch billing/validate path)."""
    err_ack = {"type": "error", "message": transient_msg}
    first_ws = MockWebSocket([dict(err_ack)])
    good_ws = MockWebSocket([_STARTED_MSG, _STEP_MSG, _DONE_MSG])
    mock_create_conn.side_effect = [first_ws, good_ws]

    with patch("time.sleep") as mock_sleep:
        result = _call_optimize()

    assert mock_create_conn.call_count == 2           # reconnected and retried
    assert mock_sleep.call_count == 1
    mock_post.assert_not_called()                     # not a connection-error fallback
    assert len(result.history) == 1


# ===================================================================
# 15. Non-transient billing rejection raises immediately, no retry
# ===================================================================

@patch("requests.post")
@patch("websocket.create_connection")
def test_insufficient_credits_not_retried(mock_create_conn, mock_post):
    """A billing rejection carries no transient signature -> raise on attempt 1,
    no backoff, no second connection (don't burn 14s retrying a hard 'no')."""
    err_ack = {"type": "error",
               "message": "Insufficient credits. Purchase at spinsphotonics.com"}
    mock_create_conn.side_effect = [
        MockWebSocket([dict(err_ack)]),
        MockWebSocket([_STARTED_MSG, _STEP_MSG, _DONE_MSG]),  # must never be used
    ]

    with patch("time.sleep") as mock_sleep:
        with pytest.raises(RuntimeError, match="Insufficient credits"):
            _call_optimize()

    assert mock_create_conn.call_count == 1
    mock_sleep.assert_not_called()
    mock_post.assert_not_called()


# ===================================================================
# 16. Malformed (non-JSON) startup ack closes the socket and fails fast
# ===================================================================

@patch("requests.post")
@patch("websocket.create_connection")
def test_malformed_startup_ack_raises_and_closes(mock_create_conn, mock_post):
    """A non-JSON first frame must not leak the socket and must not be retried
    (it is a protocol error, not a transient)."""
    class BadAckWebSocket(MockWebSocket):
        def recv(self):
            return "<html>502 Bad Gateway</html>"  # not JSON

    ws = BadAckWebSocket([])
    mock_create_conn.side_effect = [ws,
                                    MockWebSocket([_STARTED_MSG, _STEP_MSG, _DONE_MSG])]

    with patch("time.sleep") as mock_sleep:
        with pytest.raises(RuntimeError, match="Malformed startup ack"):
            _call_optimize()

    assert mock_create_conn.call_count == 1   # not retried
    assert ws._closed                          # socket closed, not leaked
    mock_sleep.assert_not_called()
    mock_post.assert_not_called()
