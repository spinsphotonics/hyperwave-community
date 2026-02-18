"""Fixtures for integration tests against mock server."""

import os
import sys
import time
import subprocess

import pytest
import requests

MOCK_SERVER_PORT = 8765
MOCK_SERVER_URL = f"http://localhost:{MOCK_SERVER_PORT}"
TEST_API_KEY = "test-api-key-12345"

MOCK_SERVER_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "hyperwave-cloud", "tests", "mock_server.py"
)


@pytest.fixture(scope="session")
def mock_server():
    """Start mock server for the test session, kill after."""
    # Check if server already running
    try:
        r = requests.get(f"{MOCK_SERVER_URL}/health", timeout=1)
        if r.status_code == 200:
            yield MOCK_SERVER_URL
            return
    except requests.ConnectionError:
        pass

    server_path = os.path.abspath(MOCK_SERVER_PATH)
    if not os.path.exists(server_path):
        pytest.skip(f"Mock server not found at {server_path}")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "mock_server:app",
         "--host", "0.0.0.0", "--port", str(MOCK_SERVER_PORT)],
        cwd=os.path.dirname(server_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    for _ in range(30):
        try:
            r = requests.get(f"{MOCK_SERVER_URL}/health", timeout=1)
            if r.status_code == 200:
                break
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("Mock server failed to start")

    yield MOCK_SERVER_URL

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture(autouse=True)
def reset_billing(mock_server):
    requests.post(f"{mock_server}/billing/reset")
    yield


@pytest.fixture
def configure_sdk(mock_server):
    import hyperwave_community as hwc
    from hyperwave_community.api_client import _API_CONFIG
    _API_CONFIG['api_key'] = TEST_API_KEY
    _API_CONFIG['api_url'] = mock_server
    _API_CONFIG['gateway_url'] = None
    return hwc
