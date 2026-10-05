#!/usr/bin/env bash
# One-time server setup (Ubuntu/Debian or RHEL-family): JDK 21, git, jq, build the CLI, create the data directory.
# usage: tools/server/setup.sh          (run from the repository root; sudo is used only for package installation)
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
if command -v apt-get >/dev/null; then
  sudo apt-get update -qq && sudo apt-get install -y -qq openjdk-21-jdk-headless git jq curl unzip time >/dev/null
elif command -v dnf >/dev/null; then
  sudo dnf install -y -q java-21-openjdk-devel git jq curl unzip time >/dev/null
else
  echo "no apt-get/dnf: install JDK 21, git, jq, time manually" >&2
fi
mkdir -p "$root/../ml-data"
cd "$root" && ./gradlew :ml-core:test :ml-train:installDist -q --console=plain
echo "== environment"
echo "host: $(hostname)  kernel: $(uname -r)"
echo "cpu:  $(nproc) cores, $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
echo "ram:  $(free -g | awk '/Mem:/ {print $2}') GB total, $(free -g | awk '/Mem:/ {print $7}') GB available"
echo "disk: $(df -h "$root" | awk 'NR==2 {print $4 " free on " $6}')"
echo "java: $(java -version 2>&1 | head -1)"
echo "git:  $(git --version); jq: $(jq --version)"
echo "cli:  $root/ml-train/build/install/ml-train/bin/ml-train"
echo "github reachability: $(curl -s -o /dev/null -w '%{http_code}' --max-time 10 https://github.com/ || echo unreachable)"
echo "setup done"
