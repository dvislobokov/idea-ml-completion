#!/usr/bin/env bash
# Render data/<lang>-repos.csv as Markdown into data/<lang>-repos.md
set -euo pipefail
lang="$1"; title="$2"
in="data/${lang}-repos.csv"; out="data/${lang}-repos.md"
n=$(($(wc -l < "$in")-1))
{
  echo "# Корпус $title: кандидаты для предобучения"
  echo
  echo "Дата отбора: $(date +%F). Критерии: GitHub stars ≥ 1000, push после 2025-06-01, не форк, не архив,"
  echo "лицензия MIT / Apache-2.0 / BSD-2 / BSD-3, доля $title ≥ 60%, размер 1 МБ – 3 ГБ, квота на владельца"
  echo "(5, у крупных организаций больше), исключены списки, туториалы, шаблоны и примеры."
  echo
  echo "Всего: $n репозиториев. Источник: \`$in\`."
  echo
  duckdb -markdown -c "select row_number() over () as '#', '[' || repo || '](https://github.com/' || repo || ')' as repo, stars as '★', license, size_mb as 'MB', share_pct as '%', pushed, description from read_csv('$in', header=true, all_varchar=true)"
} > "$out"
echo "$out" >&2
