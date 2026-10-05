# Сервер обучения: пошаговая инструкция

Сервер: `109.236.57.144`, AMD EPYC 7443P 24 ядра, 64 ГБ RAM, 2 × 960 ГБ SSD. Ожидается Linux (Ubuntu/Debian или
RHEL-семейство). Все команды ниже выполняются на сервере, если не сказано иное. После каждого шага пришлите мне то,
что указано в блоке «Что прислать».

Сейчас SSH с моей машины не проходит: TCP-соединение открывается напрямую, минуя VPN, но sshd не отвечает баннером
и сбрасывает соединение (`Connection reset`). Это похоже на фильтр на стороне сервера: fail2ban, ufw/iptables с
ограничением по странам или адресам, либо sshd слушает только VPN-интерфейс. Поэтому шаг 0 делается через консоль
провайдера (VNC / serial / web-терминал в панели).

---

## Шаг 0. Доступ по SSH (через консоль провайдера)

### 0.1. Диагностика, почему SSH не отвечает

```sh
sudo systemctl status ssh sshd 2>/dev/null | head -20
sudo ss -tlnp | grep ':22 '                      # на каком адресе слушает sshd
sudo sshd -T 2>/dev/null | grep -Ei 'listenaddress|allowusers|denyusers|usedns|maxstartups|passwordauth|pubkeyauth'
sudo ufw status verbose 2>/dev/null
sudo iptables -S 2>/dev/null | head -40
sudo nft list ruleset 2>/dev/null | head -60
sudo fail2ban-client status sshd 2>/dev/null
sudo journalctl -u ssh -u sshd -n 50 --no-pager
```

**Что прислать:** весь вывод этого блока.

Типовые исправления (применять по результату диагностики):

```sh
# fail2ban забанил мой адрес — разбанить и добавить в игнор
sudo fail2ban-client set sshd unbanip <IP_моей_машины>
# ufw закрыт — открыть 22 порт
sudo ufw allow 22/tcp && sudo ufw reload
# sshd слушает только один адрес — убрать ListenAddress или добавить 0.0.0.0
sudo sed -i 's/^ListenAddress/#ListenAddress/' /etc/ssh/sshd_config && sudo systemctl restart ssh
```

Адреса, с которых идут подключения с моей машины: напрямую через провайдера `46.138.243.124` (именно так сейчас идёт
трафик к серверу после `no-route` на ocserv), через VPN `72.56.21.195`. Если в `journalctl` видны отказы для
`46.138.243.124`, это фильтр на сервере; если записей о подключении нет вовсе, пакеты режутся раньше sshd
(провайдер или геофильтр), и тогда проще снять `no-route` и пускать трафик через VPN.

### 0.2. Пользователь для обучения и мой ключ

```sh
sudo useradd -m -s /bin/bash -G sudo ml 2>/dev/null || sudo useradd -m -s /bin/bash -G wheel ml
echo 'ml ALL=(ALL) NOPASSWD: ALL' | sudo tee /etc/sudoers.d/ml >/dev/null && sudo chmod 440 /etc/sudoers.d/ml
sudo -u ml mkdir -p /home/ml/.ssh && sudo chmod 700 /home/ml/.ssh
echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBWWU20mWsaeLStUJv3Q2CZIiD12zZN3R8IC1Ar9ZB4R dvislobokov@DESKTOP-PUVEQGK' | sudo tee -a /home/ml/.ssh/authorized_keys >/dev/null
sudo chmod 600 /home/ml/.ssh/authorized_keys && sudo chown -R ml:ml /home/ml/.ssh
```

Это публичный ключ `~/.ssh/id_ed25519.pub` с вашей рабочей машины, тот же, которым вы ходите на ocserv. После этого
я подключаюсь как `ssh ml@109.236.57.144` и дальше всё делаю сам. Если хотите, чтобы я работал под `root`, добавьте
ту же строку в `/root/.ssh/authorized_keys`.

**Что прислать:** «ключ добавлен», имя пользователя.

---

## Шаг 1. Окружение и сборка

```sh
sudo apt-get update -qq && sudo apt-get install -y git tmux          # Debian/Ubuntu; на RHEL: sudo dnf install -y git tmux
mkdir -p ~/work && cd ~/work
git clone https://github.com/dvislobokov/idea-ml-completion.git
cd idea-ml-completion
tools/server/setup.sh
```

`setup.sh` ставит JDK 21, git, jq, собирает CLI, прогоняет тесты и печатает блок `== environment`.

**Что прислать:** блок `== environment` целиком. В нём важна строка `github reachability`: должно быть `200`.

Длинные шаги ниже запускайте внутри `tmux`, чтобы обрыв SSH их не убил:

```sh
tmux new -s ml        # войти; выйти, не останавливая: Ctrl+B, затем D; вернуться: tmux attach -t ml
```

---

## Шаг 2. Корпус (30–60 минут)

```sh
cd ~/work/idea-ml-completion
tools/server/fetch.sh > ../ml-data/fetch.log 2>&1
tools/server/stats.sh
```

Скачиваются 737 C# и 900 Go репозиториев без истории, только исходники, около 10 ГБ. Повторный запуск продолжает с
места обрыва. Прогресс: `tail -f ../ml-data/fetch.log`.

**Что прислать:** вывод `tools/server/stats.sh` (три строки).

---

## Шаг 3. Разбиение на доли

```sh
tools/server/sets.sh go && tools/server/sets.sh csharp
```

По 30 репозиториев откладывается на тест, остальные делятся 2:1 на долю для языковой модели и долю для ранкера.

**Что прислать:** две напечатанные строки.

---

## Шаг 4. Обучение n-грамм и ранкера (20–40 минут на язык)

```sh
tools/server/train.sh go full-mkn4
tools/server/train.sh csharp full-mkn4
```

Результат каждого запуска: `../ml-data/<lang>/exp/<tag>/summary.txt`, модели `lm.cml` и `rank.cml` рядом, подробные
логи `l2.log`, `l1.log`.

**Что прислать:** оба `summary.txt`. Если в них есть `FAILED`, то и хвост упомянутого лога.

Память: языковая модель запускается с кучей 40 ГБ, ранкер с 20 ГБ. При `OutOfMemoryError` в логе:

```sh
XMX_LM=52g tools/server/train.sh go full-mkn4          # больше памяти языковой модели
PER_FILE=10 tools/server/train.sh go full-mkn4         # меньше примеров ранкера на файл
```

---

## Шаг 5. Серия экспериментов (после шага 4, по моим подсказкам)

Каждый эксперимент — одна строка с новым тегом, `train.sh` принимает параметры языковой модели:

```sh
tools/server/train.sh go prune-1122 --min-count 1,1,2,2
tools/server/train.sh go prune-repos3 --min-count 1,1,2,2 --min-repos 1,1,1,3
tools/server/train.sh go order5 --order 5 --min-count 1,1,2,2,2
```

---

## Шаг 6. Сводный отчёт

```sh
tools/server/report.sh
cat ../ml-data/report.txt
```

**Что прислать:** `../ml-data/report.txt` целиком.

---

## Шаг 7. Датасет из PSI Go-плагина (позже, когда будут модели)

Нужны клон `idea-golang-support`, дистрибутив IntelliJ IDEA Community 2026.1.4 и Go toolchain:

```sh
cd ~/work
git clone <ваш репозиторий idea-golang-support>
# IDEA Community (около 1,5 ГБ); путь затем в gradle.properties: localIdePath
curl -L -o idea.tar.gz 'https://download.jetbrains.com/idea/ideaIC-2026.1.4.tar.gz' && mkdir -p idea && tar -xzf idea.tar.gz -C idea --strip-components=1
sudo apt-get install -y golang-go                      # для GOROOT со стандартной библиотекой
```

Скрипт запуска экспорта я добавлю в `tools/server/` отдельным шагом, когда дойдём: он гоняет
`./gradlew :go-psi-ide:mlDataset` по долям `sets/rank.txt` и `sets/test.txt` с языковой моделью из шага 4.
