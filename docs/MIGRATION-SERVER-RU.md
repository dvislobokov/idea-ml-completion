# Переезд на новый сервер (2 × H200): что копировать, что пересобирать

Состояние на 2026-10-07. Старый сервер: 32 vCPU / 176 ГБ / RTX PRO 6000, `/` 400 ГБ (`sda`), `/mnt/corpus` 2 ТБ (`sdb`).

## 1. Что обязательно скопировать (невоспроизводимо или дорого)

| Путь на старом сервере | Размер | Зачем |
|---|---|---|
| `~/work/ml-data/catalog/` | 17 МБ | каталоги GitHub-репозиториев (списки, звёзды, лицензии) |
| `~/work/ml-data/go/prepared/`, `csharp/prepared/` | 1,1 + 1,6 ГБ | манифесты и фолды lm/rank/test — **основа воспроизводимости**; пересчитываются только с тем же корпусом |
| `~/work/ml-data/tokenizer/` | 85 МБ | словари BPE `go-16384.bpe`, `cs-16384.bpe` (+32k), фикстуры паритета |
| `~/work/ml-data/go/bpe16k/`, `csharp/bpe16k/` | 20 + 16 ГБ | BPE-шарды для обучения (пересчёт ~10–15 мин на язык, но нужен корпус) |
| `~/work/ml-data/go/models/`, `csharp/models/` | 290 + 140 МБ | все модели: n-граммы e14/e15, ранкеры, `go-nn-31m-e1.cml`, `cs-nn-31m-e1.cml` |
| `~/work/ml-data/go/nn/`, `csharp/nn/` | 1,9 + 0,8 ГБ | чекпоинты обучения (`ckpt-latest.pt`, `ckpt-before-fimfix.pt`), логи, результаты `eval-inline-*`, паритетные фикстуры, отчёты исследований |
| `~/work/ml-data/go/psi/`, `csharp/psi/` | 220 + 140 МБ | реальные списки автодополнения (`.cmlx`), PSI-контекст, прототипы Roslyn |
| `~/work/nn/` **без** `.venv` | ~100 МБ | канонические Python-скрипты (копия в репо `tools/nn/`) + `psi/`, `psi-cs/`, `research/`, `clean/out` |

Итого ~45 ГБ. Команда (с нового сервера, после настройки SSH на старый):

```sh
mkdir -p ~/work/ml-data && cd ~/work/ml-data
rsync -a --info=progress2 old:~/work/ml-data/{catalog,tokenizer} .
rsync -a --info=progress2 --exclude 'repos' --exclude 'repos.*' --exclude 'shards*' --exclude 'prepared-*' old:~/work/ml-data/go ./
rsync -a --info=progress2 --exclude 'repos' --exclude 'repos.*' --exclude 'shards*' --exclude 'prepared-*' old:~/work/ml-data/csharp ./
rsync -a --exclude '.venv' --exclude '__pycache__' old:~/work/nn ~/work/
```

## 2. Корпус: копировать или скачать заново

| | Копировать (`go/repos` 190 ГБ с `sda`, `/mnt/corpus/csharp/repos` 113 ГБ) | Скачать заново (`tools/corpus/fetch-catalog.sh`, ~25 + 40 мин) |
|---|---|---|
| Плюс | манифесты и фолды остаются в точности валидными | быстро, не грузит старый сервер |
| Минус | 300 ГБ по сети | снимки `HEAD` сдвинутся → `prepare` надо перепрогнать (26 мин/язык), числа e14–e16 станут не байт-в-байт воспроизводимыми |

Рекомендация: **копировать**, если между серверами ≥ 100 МБ/с (≈ 1 час); иначе — скачать заново и перепрогнать `prepare`,
`shard`, `encode` (всё автоматизировано, ~2 часа суммарно). Шарды лексера (`go/shards` 23 ГБ, `csharp/shards` 17 ГБ) не копировать —
`ml-train shard` делает их за 8–15 мин.

## 3. Что ставится заново (ничего не копировать)

```sh
# репозитории
git clone git@github.com:dvislobokov/idea-ml-completion.git ~/work/idea-ml-completion
git clone --branch migration git@github.com:dvislobokov/idea-golang-support.git ~/work/idea-golang-support
git clone --branch master   git@github.com:dvislobokov/idea-dotnet-support.git ~/work/idea-dotnet-support
# JDK 21, Node, uv, zig, make, gcc, tmux, nvtop
apt install -y openjdk-21-jdk zig make gcc tmux nvtop golang-go   # golang-go только ради apt-зависимостей; сам Go — ниже
# Go 1.27.1 (GOROOT для go-psi), .NET SDK 10 (Roslyn-оракул для C#)
curl -sL https://go.dev/dl/go1.27.1.linux-amd64.tar.gz | tar -xz -C /usr/local      # PATH=/usr/local/go/bin:$PATH
apt install -y dotnet-sdk-10.0
# IntelliJ IDEA Community 2026.1.4 (headless сборка плагинов и экспорт PSI-датасетов)
mkdir -p ~/work/idea && curl -sL 'https://github.com/JetBrains/intellij-community/releases/download/idea%2F2026.1.4/idea-2026.1.4.tar.gz' | tar -xz -C ~/work/idea --strip-components=1
# Python для обучения: uv + venv с torch под CUDA нового сервера (H200: cu12x/cu13x по версии драйвера)
cd ~/work/nn && ~/.local/bin/uv venv --python 3.12 .venv && ~/.local/bin/uv pip install --python .venv/bin/python torch numpy   # индекс torch под нужную CUDA
# движок
cd ~/work/idea-ml-completion && ./gradlew :ml-core:test :ml-train:installDist -q
```

Диски: смонтировать корпусный диск с `noatime` (`tools/server/prepare-disk.sh` делает это и чинит atime на `/`).
Claude Code: `tmux new -s ml`, `claude --resume` не переносится — новая сессия начнёт с `CLAUDE.md` и `docs/NEURAL-RU.md §8`;
`~/.claude/settings.json` (токен, прокси) и SSH-ключ для GitHub переносит пользователь вручную.

## 4. Первое, что сделать на 2 × H200

1. **DDP в `tools/nn/train/train.py`** (`torchrun --nproc_per_node 2`): пока обучение однокарточное. ~30 строк: `init_process_group`,
   `DistributedDataParallel`, деление групп данных по rank (поток `PackedStream` детерминирован по `(seed, epoch, gi)` — раздать
   группы по `gi % world_size`), `all_reduce` метрик, чекпоинт с rank 0. Проверить resume.
2. Прогнать смоук go31m 10 минут и сравнить tok/s с 771k на PRO 6000 (ожидание ×3–4 на две карты).
3. Очередь, согласованная в `docs/NEURAL-RU.md §8` и `CLAUDE.md` «Plan»: Go на чистом корпусе с `fim_rate 0.7`; cs31m-e2 / cs50m;
   затем формат с PSI-контекстом (данные в `go/psi/context-*.jsonl`, `csharp/psi/roslyn/*.ctx.jsonl`) и учитель для дистилляции.

## 5. Что НЕ переносить

`go/shards`, `csharp/shards` (пересчёт), `csharp/prepared-v1`, `-v2`, `go/prepared-prev`, `-crashed`, `-300` (старые манифесты),
`~/work/nn/.venv` (torch под другую CUDA), `~/.gradle` (кэш), `~/work/idea` (скачать), `.claude/worktrees/*` (агентские worktree —
всё нужное слито в `main`), `csharp/repos.sda-old` (удалён), `ml-train/build`.
