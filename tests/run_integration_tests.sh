#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMMUNITY_DIR="$(dirname "$SCRIPT_DIR")"
CLOUD_DIR="$(dirname "$COMMUNITY_DIR")/hyperwave-cloud"
MOCK_SERVER_DIR="$CLOUD_DIR/tests"
MOCK_PORT=8765

echo "=== Integration Test Runner ==="
echo "Community SDK: $COMMUNITY_DIR"
echo "Mock server:   $MOCK_SERVER_DIR/mock_server.py"
echo ""

# Check dependencies
python3 -c "import fastapi, uvicorn, numpy, requests" 2>/dev/null || {
    echo "Installing test dependencies..."
    pip install fastapi uvicorn numpy requests pytest 2>/dev/null
}

# Kill any existing mock server
pkill -f "uvicorn mock_server:app" 2>/dev/null || true
sleep 1

# Start mock server
echo "Starting mock server on port $MOCK_PORT..."
cd "$MOCK_SERVER_DIR"
python3 -m uvicorn mock_server:app --host 0.0.0.0 --port $MOCK_PORT &
MOCK_PID=$!
echo "Mock server PID: $MOCK_PID"

# Wait for server
for i in $(seq 1 20); do
    if curl -s "http://localhost:$MOCK_PORT/health" > /dev/null 2>&1; then
        echo "Mock server ready."
        break
    fi
    sleep 0.5
done

# Run tests
echo ""
echo "=== Running Tests ==="
cd "$COMMUNITY_DIR"

# Install the SDK in editable mode if not already
pip install -e . 2>/dev/null || true

EXIT_CODE=0
python3 -m pytest tests/integration/ -v --tb=short -x 2>&1 || EXIT_CODE=$?

# Cleanup
echo ""
echo "=== Cleanup ==="
kill $MOCK_PID 2>/dev/null || true
wait $MOCK_PID 2>/dev/null || true
echo "Mock server stopped."

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "ALL TESTS PASSED"
else
    echo ""
    echo "TESTS FAILED (exit code $EXIT_CODE)"
fi

exit $EXIT_CODE
