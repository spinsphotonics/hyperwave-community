#!/bin/bash
# Install git hooks for this repository
set -e
REPO_ROOT=$(git rev-parse --show-toplevel)
HOOKS_DIR="$REPO_ROOT/.git/hooks"
mkdir -p "$HOOKS_DIR"

cat > "$HOOKS_DIR/pre-push" << 'HOOK'
#!/bin/bash
PROTECTED_BRANCH="main"
while read local_ref local_sha remote_ref remote_sha; do
  if echo "$remote_ref" | grep -q "refs/heads/$PROTECTED_BRANCH"; then
    echo ""
    echo "BLOCKED: Direct push to '$PROTECTED_BRANCH' is not allowed."
    echo ""
    echo "Use a feature branch and open a PR instead."
    echo "CI checks must pass before merging."
    echo ""
    echo "To force push (emergency only): git push --no-verify"
    echo ""
    exit 1
  fi
done
exit 0
HOOK
chmod +x "$HOOKS_DIR/pre-push"
echo "Git hooks installed successfully."
