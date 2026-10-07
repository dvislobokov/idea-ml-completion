#!/usr/bin/env bash
# Render docs/talk/slides.md with marp-cli (Node 22, network for the first npx run).
# HTML always; PDF only when a Chromium/Chrome binary is available (marp needs a browser for PDF).
set -euo pipefail
cd "$(dirname "$0")"

# charts (optional, needs matplotlib): PY=/root/work/nn/.venv/bin/python ./build.sh
if [ -n "${PY:-}" ]; then "$PY" -I make_charts.py; fi

MARP="npx -y @marp-team/marp-cli@latest"
$MARP slides.md -o slides.html --html

# find a browser for the PDF export
BROWSER="${CHROME_PATH:-}"
for c in chromium chromium-browser google-chrome google-chrome-stable; do
  [ -z "$BROWSER" ] && command -v "$c" >/dev/null 2>&1 && BROWSER="$(command -v "$c")"
done
if [ -z "$BROWSER" ] && [ -d "$HOME/.cache/puppeteer" ]; then
  BROWSER="$(find "$HOME/.cache/puppeteer" -type f -name chrome -perm -u+x 2>/dev/null | head -1 || true)"
fi
if [ -z "$BROWSER" ] && [ "${INSTALL_CHROME:-0}" = "1" ]; then
  npx -y @puppeteer/browsers install chrome@stable --path "$HOME/.cache/puppeteer" >/dev/null
  BROWSER="$(find "$HOME/.cache/puppeteer" -type f -name chrome -perm -u+x 2>/dev/null | head -1 || true)"
fi
if [ -n "$BROWSER" ]; then
  CHROME_PATH="$BROWSER" $MARP slides.md -o slides.pdf --pdf --allow-local-files --html
  echo "built slides.html and slides.pdf (browser: $BROWSER)"
else
  echo "built slides.html; no Chromium found — PDF skipped (INSTALL_CHROME=1 ./build.sh downloads one)"
fi
