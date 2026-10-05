#!/usr/bin/env bash
# Update the server's checkout from GitHub without the git smart protocol (which some providers break): resolves the
# branch to a commit through the GitHub API (codeload caches branch tarballs for minutes, commit tarballs are exact),
# downloads that commit's tarball from codeload.github.com, unpacks it over the working copy and rebuilds the CLI.
# usage: tools/server/update.sh [branch-or-commit]      (default main)
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
ref="${1:-main}"
repo="dvislobokov/idea-ml-completion"
if [ ${#ref} -lt 40 ]; then
  sha="$(curl -sfL --retry 3 "https://api.github.com/repos/$repo/commits/$ref" | jq -r .sha)"
  [ -n "$sha" ] && [ "$sha" != "null" ] || { echo "cannot resolve $ref through the GitHub API" >&2; exit 1; }
else
  sha="$ref"
fi
tgz="$(mktemp --suffix=.tgz)"
curl -sfL --retry 3 -o "$tgz" "https://codeload.github.com/$repo/tar.gz/$sha"
tar -xzf "$tgz" -C "$root" --strip-components=1
rm -f "$tgz"
chmod +x "$root/gradlew" "$root"/tools/*/*.sh
echo "$sha" > "$root/.server-commit"
cd "$root" && ./gradlew :ml-train:installDist -q --console=plain
echo "updated to $ref ($sha)"
