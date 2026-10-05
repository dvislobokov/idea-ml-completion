#!/usr/bin/env bash
# Update the server's checkout from GitHub without the git smart protocol (which some providers break): downloads the
# tarball of the branch from codeload.github.com, unpacks it over the working copy and rebuilds the CLI.
# usage: tools/server/update.sh [branch]      (default main)
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
branch="${1:-main}"
tgz="$(mktemp --suffix=.tgz)"
curl -sfL --retry 3 -o "$tgz" "https://codeload.github.com/dvislobokov/idea-ml-completion/tar.gz/$branch"
tar -xzf "$tgz" -C "$root" --strip-components=1
rm -f "$tgz"
chmod +x "$root/gradlew" "$root"/tools/*/*.sh
cd "$root" && ./gradlew :ml-train:installDist -q --console=plain
echo "updated to $branch: $(grep -m1 '^## ' CHANGELOG.md | head -1)"
