# Обзор литературы: локальный ML-движок автодополнения (n-gram + ranker + gray text) для C#/Go плагинов IntelliJ

Дата: 2026-10-05. Все числа ниже — из первоисточников (ссылки в конце); там, где это мой вывод/экстраполяция, это помечено явно словами «моя оценка» / «вывод».

Контекст нашей задачи: общий Kotlin/JVM-движок, без Python и сети, CPU-only; бюджет <5 мс на ранжирование списка до 500 элементов, <30 мс на gray text; модель ≤15 МБ на язык. Прототип: 4-gram modified Kneser-Ney (MKN) по лексерным токенам, top-50k идентификаторов (остальное `<ID>`), хэшированные таблицы с 24-битным fingerprint и 8-битными log-prob (~5 байт/запись; 19 МБ без прунинга, 8 МБ с прунингом синглтонов 3/4-грамм), listwise логистическая регрессия над ~6 признаками. Метрики: perplexity 8–10/токен, next-identifier top-1 ≈ 0.30–0.33, ranker MRR 0.72–0.74, OOV идентификаторов 9 % (Go) / 17.5 % (C#).

---

## 1. Что реально дают n-gram + cache на коде; где граница с нейросетями

**Измерено (Hellendoorn & Devanbu, FSE'17, Java-корпус Allamanis & Sutton, оценка по всем токенам, MRR по топ-10):**

| Модель (JM-6 сглаживание) | Entropy, бит | MRR |
|---|---|---|
| Plain n-gram, flat корпус, dynamic | 5.54 | ≈0.56–0.60 |
| + file cache (Tu et al. 2014) | 3.43 | ≈0.69–0.70 |
| Nested (по иерархии директорий проекта), без cache | 3.65 / 2.94 (maintenance) | 0.67 / 0.71 |
| Nested + cache | 2.57 / 2.23 (maintenance) | 0.745 / 0.788 |
| LSTM/650 static (vocab 74k) | 3.03 | 0.68 |
| Nested cache, **без ограничения словаря** | 1.92 | **0.818** (top-1 0.759) |

Ключевые выводы из той же работы:
- Cache-компонент даёт ~2 бита энтропии и ~+10 п.п. MRR; «nested» (счётчики по вложенным директориям проекта) — ещё ~1 бит и ~+5 п.п. Информация из соседних файлов проекта «почти так же полезна, как из того же файла».
- **Modified Kneser-Ney хуже Jelinek-Mercer на коде**: KN-3 отстаёт от JM-6 на 0.56 бита (0.40 с cache); у MKN качество cache-модели деградирует с ростом порядка. Это прямо релевантно нашему прототипу (мы используем MKN).
- Увеличение обучающего корпуса с 1 % до 75 % (~1 млрд токенов) дало nested-cache модели лишь +0.06 бита и +0.4 % MRR — n-gram **насыщается** быстро; наши 13 репозиториев → 700–900 репозиториев дадут немного для LM-части (но много для словаря и для фич ranker'а).
- Смешивание с LSTM: энтропия падает до 1.17–1.25 бита, но MRR в dynamic-режиме почти не растёт (cache доминирует).

**Karampatsis et al., ICSE'20 (та же Java test, SLP-Core как baseline, Table 2):** nested cache n-gram на small train: dynamic 2.57 бит / MRR 74.6; на large train 2.49 / 75.0. Открытословарная BPE-NLM (GRU, 2048 units, BPE 10k) на large train: 1.23 бит / MRR 82.4 (+cache 83.3); **по идентификаторам** MRR 59.1 (n-gram) vs 62.9–65.5 (NLM). Т.е. практический отрыв NLM от хорошо сделанной n-gram — ~5–9 п.п. MRR на идентификаторах, при этом n-gram на большом корпусе занимала 6–8.5 ГБ диска и 50–60 ГБ RAM (без компрессии!), NLM — 15–45 МБ.

**Svyatkovskiy et al., MSR'21 (IntelliCode, member-completion по кандидатам от статического анализа, Python):** переформулировка «генерация → ранжирование кандидатов» критична: модель ~3 МБ ⟨BPE, GRU, StaticAnalysis⟩ даёт Recall@1 66.4 / Recall@5 88.0 / MRR 0.757 за 5.4 мс CPU; ~50 МБ — 70.1 / 89.8 / 0.786 за 7.7 мс. Baseline «частота использования» — Recall@1 41.8 / MRR 0.547. В продакшене Recall@5 упал с 90 % offline до ~70 % online.

**Выводы для нас.** (a) Наш perplexity 8–10 (≈3.0–3.3 бита) без cache — согласуется с литературой; (b) добавление file-cache + project-level nested counts должно дать порядка 1.5–2 бита и +8–12 п.п. MRR на всех токенах (моя экстраполяция с Java на C#/Go; для Go, с короткими именами и большим числом локальных переменных, выигрыш cache обычно больше); (c) смена сглаживания MKN → JM (или интерполяция JM/absolute discounting) — дешёвая проверка с ожидаемыми 0.3–0.5 бита.

## 2. OOV идентификаторов

Измерено:
- Karampatsis et al.: закрытый словарь 75k даёт OOV ~10–15 % на Java-тесте; BPE с 2–10k merges убирает OOV полностью, при этом **разницы качества между 2k/5k/10k практически нет** (MRR 77.5–78.7 maintenance на small train). CamelCase/snake-разбиение («Heuristic NLM») хуже BPE и даже хуже closed-vocab по MRR (64.1 vs 71.0 dynamic), т.к. длинные последовательности субтокенов размывают контекст.
- Hellendoorn & Devanbu: лучший практический приём для n-gram — **не ограничивать словарь вовсе** и опираться на cache/nested counts: именно nested cache с открытым словарём даёт MRR 0.818, теряя лишь 4.4 п.п. относительно режима «UNK засчитывается».
- Li et al., IJCAI'18 (pointer mixture, PY150/JS150): при словаре 50k OOV 7–11 %, pointer-копирование из локального контекста добавляет лишь +0.4–0.5 п.п. к attention-LSTM (больше — при словаре 1k: +1.5 п.п.).
- Aye & Kaiser, ICSE'20 (Dart, Google): 3.3 % всех токенов — повторы OOV-токенов, уже встречавшихся в окне контекста; детектор повторов (precision 0.905/recall 0.907) закрывает их. Субтокенная модель лучше токенной на +3–4 п.п. top-1 (0.714 vs 0.678).

**Применение к нам.** В сценарии ранжирования списка кандидаты известны из PSI, поэтому OOV для LM — это не «не можем предложить», а «нет вероятности для score». Рекомендуемая комбинация:
1. Сохранить `<ID>` класс как fallback, но считать для OOV-кандидата **backoff-score из субтокенов** (camelCase/snake split → unigram/bigram по субтокенам, top-5–10k субтокенов). Это заменяет 17.5 % «слепых» кандидатов в C# на информативный score; дёшево (одна таблица ≤1 МБ).
2. Основной источник вероятностей для OOV — **проектная/файловая cache-модель** (п. 7): любой идентификатор, объявленный в проекте, после первого появления уже не OOV для cache. По данным Hellendoorn именно этот компонент важнее глобальной LM.
3. Полный BPE для n-gram не рекомендуется: удлиняет контекст в 2–3 раза при фиксированном порядке 4 (Karampatsis прямо пишут, что для n-gram BPE «неприменим»). BPE имеет смысл только если/когда появится transformer.
4. Для gray text (генерация) OOV критичен: там ограничивать выход **PSI-видимыми идентификаторами** (как делаем) и для них использовать субтокенный/cache score — это решает проблему без открытого словаря.

## 3. Ранжирование списка

**JetBrains, «All You Need Is Logs» (FSE'22 Industry, Bibaev et al.):**
- Модель: CatBoost с listwise-лоссом **QuerySoftMax**; переход с RMSE на QuerySoftMax дал +10 % success rate (blog, episode 4). Пробовали LightGBM, MLP, Transformer — «ни один не приблизился» при ограничении размера модели; деревья конвертируются в if-else и исполняются внутри JVM.
- Ограничения: модель ≤2 МБ (реально 366 КБ), inference «десятки мс» (измерено +27 мс латентности 92.3 → 119.3 мс, т.к. ранжируется на каждый символ).
- Признаки: ~146 (Java) / 188 (Python) после отбора по permutation importance из «нескольких сотен». Группы: **prefix** (длина, число совпавших символов, case-sensitive/exact match), **синтаксический контекст** (keyword? тот же файл/модуль? third-party?), **история навигации** (ходил ли пользователь к определению), **история сессии** (выбирал ли этот элемент раньше, длительность), языковые (тип элемента, частота использования, блок if/for).
- Offline (Python): Recall@1 all 0.761 → 0.870; **Recall@1 на первом показе списка 0.634 → 0.799** (+16.5 п.п.) — это была целевая метрика. Online A/B (231 vs 246 пользователей): typing actions 2.073 → 1.832, typed-select 0.416 → 0.346, prefix length 2.63 → 2.28, explicit select 0.247 → 0.292 (не значимо), manual start 0.047 → 0.079.
- «Jumping»: явного механизма в статье нет; в blog (episode 1) описан freeze первых позиций: поздно пришедшие элементы ставятся **ниже выделенного**, чтобы «вариант не уехал из-под пальцев». Ранжирование повторяется на каждый символ, и ML-модель с prefix-фичами как раз сглаживает скачки.

**Bruch et al., FSE'09 (BMN, kNN по контексту вызовов)**: для метод-вызовов top-3 — recall 68 % / precision 77 % против ~Eclipse-алфавитного порядка; **Proksch et al., TOSEM'15 (PBN, байесовские сети)** — качество сравнимо с BMN при лучшем масштабировании размера модели/скорости, прирост от дополнительного контекста (тип получателя, окружающий метод) заметен для запросов без вызовов. Для нас это аргумент за фичи «что уже вызывалось на этом объекте/типе в текущем методе».

**Выбор модели для нас.** Логистическая регрессия над ~6 признаками — хороший старт, но литература однозначно показывает, что основной выигрыш идёт от **количества и качества признаков** (десятки–сотни), а не от класса модели; при 20+ признаках с нелинейностями (prefix match × длина, recency × kind) GBDT (CatBoost/LightGBM, 100–300 деревьев глубины 4–6) обычно даёт +3–6 п.п. R@1 (моя оценка по аналогичным LTR-задачам; у JetBrains CatBoost vs heuristics +11 п.п.). Инференс 300–500 деревьев на 500 кандидатов — ~0.5–2 мс на JVM при предвычисленных фичах (вывод). Переранжировать только top-N (N≈20–50 по эвристике/прочитанному prefix) — стандартная практика для латентности; потери минимальны, если recall@50 эвристики >0.98 (у JetBrains R@5 эвристики уже 0.957).

Метрики для shipping (по JetBrains): R@1 на первом показе списка, typing actions, typed-select rate, explicit-cancel; значимость через bootstrap по пользователям, p<0.01.

## 4. Компрессия n-gram LM

- **Entropy pruning (Stolcke 2000):** Hub4 4-gram, порог 1e-8 → модель 26 % от исходной, +5.7 % perplexity (163.0 → 172.3), WER без изменений; порог 1e-7 → ~5 % размера, perplexity +24 % и WER +1.3 п.п. Для KN-моделей известно (мой комментарий, общеизвестная оговорка Chelba et al. 2010), что entropy pruning хуже сочетается с KN, чем с Katz/JM — ещё один аргумент за JM.
- **KenLM (Heafield 2011):** PROBING — хэш с линейным probing, в 2.4× быстрее SRILM при 57 % памяти; TRIE с bit-packing + quantization 8 бит prob/7 бит backoff (`-q 8 -b 7`) — 21 % памяти SRILM; KenLM 5-gram ~16–20 байт/n-gram (probing) и ~4–6 (quantized trie). Pauls & Klein 2011: COMPRESSED-трие ~1.5–3 байта/n-gram при скорости на уровне SRILM (33 байт/n-gram).
- **Randomised / fingerprint LMs:** Talbot & Osborne 2007 (Bloom filter + log-quantised counts); Guthrie & Hepple 2010 (minimal perfect hash, 2.07 бит/ключ): **2.47 байт/n-gram с 8-бит fingerprint и 8-бит значением**, 1.41 байта при меньшем числе error-bits; false positive rate 2^-m для m-битного fingerprint (0.0027 при 8 бит; у нас 24 бита → ~6e-8, избыточно).

**Применение.** Наши ~5 байт/запись (24 бита fingerprint + 8 бит prob + накладные) уже близки к Guthrie-Hepple. Пути к ≤15 МБ при 50–100× большем корпусе:
1. Entropy pruning по Stolcke (не count-cut) — ожидаемо ×3–4 сжатие при ≤5 % PPL (измерено на речи; на коде из-за более пикового распределения, вероятно, сравнимо или лучше — вывод).
2. 16-битный fingerprint (FP 1.5e-5 на запрос; при 500 кандидатов × 4 порядка ≈ 3 % списков получат один ложный score — приемлемо, если fingerprint проверяется на всех порядках, как у Talbot & Osborne) — экономит 1 байт/запись (−20 %).
3. Backoff-веса квантовать 4–5 бит, prob 7 бит (KenLM по умолчанию 8/7; Stolcke/Whittaker показывали, что 8 бит достаточно без потери PPL).
4. Хранить `<ID>`-классовые n-граммы отдельно от лексических: они плотные и хорошо жмутся.
Реалистичный итог (моя оценка): ~5 М записей × 4 байта ≈ 20 МБ → после entropy pruning 8–12 МБ на язык.

## 5. Обучение в масштабе на CPU

- **lmplz (Heafield et al., ACL'13):** потоковый подсчёт с merge-sort на диске; фиксированный RAM (`-S 50%`), 302 М токенов за 7.7 % RAM и 14 % времени SRILM; 126 млрд токенов на 140 ГБ RAM за 2.8 дня. Для 6 ГБ C# (~1–1.5 млрд токенов) и 3 ГБ Go — часы на одной машине с 16–32 ГБ RAM. Алгоритм (один поток на порядок, adjusted counts, interpolated MKN) воспроизводим в Kotlin, но проще вызывать lmplz на сервере и конвертировать ARPA в наш формат; ARPA-выход также нужен для entropy pruning (SRILM `ngram -prune`).
- **Дедупликация (Allamanis, Onward!'19):** в Java GitHub Corpus 24.8 % файлов — near-duplicates; при split по файлам до 24 % теста — cross-set дубликаты, при split **по проектам** — 8.9 %. Метод: токен-множество идентификаторов/литералов, Jaccard T0≥0.8 (set) и T1≥0.7 (multiset), файлы <20 идентификаторов игнорируются. Эффект: для нейро-LM на JS150k дубликаты завышали Acc-ID на 51 % и MRR-ID на 39 %; PHOG-автодополнение идентификаторов — 61.4 % vs 48.9 %. Для C# в датасете C#-19 дубликатов было 10.6 %.
- Также: Hellendoorn и Karampatsis нормализуют литералы, non-ASCII → `<UNK>`; FLCC (JetBrains) делает import-dropout 50 % и заменяет отступы на `<SCOPE_IN>/<SCOPE_OUT>`.

**Рекомендации:** split train/dev/test строго по репозиториям; near-dup dedup на уровне файлов до подсчёта n-грамм (реализация с MinHash/LSH на 10^6 файлов — часы на CPU); для C# дополнительно фильтровать generated code (`*.Designer.cs`, `*.g.cs`, `obj/`) и vendored/`vendor/` для Go — наши текущие OOV/PPL измерения на 13 репозиториях почти наверняка завышены, если dedup не делался.

## 6. Gray text (inline completion)

**JetBrains FLCC (Semenkin et al., 2024; blog 2024–2025):**
- Модель: LLaMA-подобный decoder, **100M параметров**, контекст 1536 токенов, BPE 16 384 (YouTokenToMe, без merge через `\n`), int4 через llama.cpp (400 → ~100 МБ; ×2 быстрее на M2, ×4 на i9), нативный C++-сервер + gRPC из Kotlin; ONNX Runtime с GPT-2 тестировали, но выбрали llama.cpp. Обучение: 8×H100, несколько дней (для 100M — «около недели» на одном узле 8×H100 по blog Mellum).
- Латентность: **~150 мс** среднее от вызова до показа; >90 % вызовов переиспользуют кэш hidden states; память — 20–40 % бюджета IDE.
- Декодирование: beam search с token healing, приоритет гипотез, завершённых `\n`, стоп при 20 итерациях или когда незавершённые гипотезы хуже лучшей завершённой в k=3 раза.
- **Фильтры показа:** (1) low confidence score, (2) синтаксическая/семантическая проверка через PSI/инспекции (Rider: «no chance of a compilation error in scope»), (3) safety/secrets; теряется ~1 % «ценных» подсказок. Переход на сбор гипотез по `\n` поднял «perfect lines» с 13 % до 21 %.
- Online: acceptance rate ≈ 35–38 %, explicit cancel ≈ 5 %, Ratio of Completed Code (RoCC) 17 % → 22 % в Python. В 2025 добавили **CatBoost-фильтр 2.5 МБ, 1–2 мс в Kotlin** по признакам «язык, число импортов, скорость набора, время с последнего нажатия, resolve ссылок, похожесть на окружающий код, token scores, энтропия»: acceptance +≈50 %, explicit cancel −≈40 % при неизменном RoCC. Mellum paper: локальная 100M-модель — RoCC 0.25/0.42 (Python/Java), AR 0.26/0.34; cloud 4B — 0.39/0.49, 0.34/0.34.
- Copilot: acceptance ≈ 30 % (28.9 → 34 % за 6 месяцев), «contextual filter» дал +6 % acceptance; по телеметрии до вызова (без кода) можно подавить 35 % вызовов и поднять AR с 18.4 до 34.2 % (Pre-Filtering, 2025). Smart Invocation (de Moor et al., AIWARE'24): маленький transformer-классификатор «вызывать ли» превосходит telemetry-baseline при низкой латентности.

**Применение к нам (n-gram gray text).** n-gram-генерация 3–8 токенов beam'ом из 4-граммы — десятки микросекунд, бюджет 30 мс не проблема. Решающее — **фильтр показа**: (a) порог на суммарную log-prob/длину и на отношение p(top1)/p(top2) первого токена; (b) обязательное ограничение идентификаторов PSI-видимыми и проверка синтаксиса PSI-парсером; (c) показывать только до конца «логической единицы» (`;`, `)`, `\n`); (d) telemetry-гейт (пауза после последнего нажатия >150–300 мс, не в комментарии/строке). Реалистичная цель для n-gram gray text — AR 20–25 % на коротких (2–5 токенов) подсказках (моя оценка; у 100M-трансформера 35 %); оптимизировать RoCC и explicit-cancel, а не AR в чистом виде (GitHub прямо предупреждает, что оптимизация AR приводит к коротким тривиальным подсказкам).

## 7. On-device персонализация

Доказательные источники:
- File cache (Tu et al. 2014; воспроизведено Hellendoorn): −2 бита, +10 п.п. MRR. Nested project counts: ещё −1 бит, +5 п.п. (maintenance-режим, когда модель видит остальные файлы проекта — наш случай с индексом IDE).
- Dynamic adaptation NLM (Karampatsis): дообучение на проекте даёт −0.5 бита / +2–3 % MRR даже для большой нейросети; для n-gram «dynamic» (добавление файлов по мере просмотра) — тот же механизм, что cache.
- Логи пользователей для ranker (Aye et al., ICSE-SEIP'21, Facebook): обучение на реальных принятых дополнениях вместо коммитов → +12.8–13.8 % accuracy и +6.2 % использования автодополнения в A/B. JetBrains — фичи «выбирал ли раньше», «ходил к определению».
- Смешивание: Hellendoorn использует простую интерполяцию с весом, зависящим от «уверенности» (число наблюдений контекста) — λ = n/(n+k); классический рецепт NLP — EM-подбор λ на held-out внутри проекта (Kuhn & De Mori 1990 для cache; типичные веса cache 0.1–0.3). 

**Рекомендация:** три уровня счётчиков — файл (экспоненциальное затухание по расстоянию в токенах/строках), проект (n-gram counts по индексу, пересчёт инкрементально при изменении файла), глобальная LM; веса по confidence-интерполяции, затем 1–2 итерации EM по последним 10k токенам пользователя. Accept-log использовать как (a) фичу ranker'а (сколько раз принимался этот элемент/в этом контексте), (b) данные для периодического дообучения ranker'а на устройстве (логрегрессия/маленький GBDT тренируются за секунды). Ожидаемый суммарный эффект на ranker MRR: +0.05–0.10 (моя оценка).

## 8. Рекомендованная дорожная карта

| # | Шаг | Ожидаемый эффект | Стоимость | Риски |
|---|---|---|---|---|
| 1 | Dedup по Allamanis + split по репозиториям; пересчитать baseline | честные метрики (текущие, вероятно, завышены на 5–20 % по ID-метрикам) | 1–2 дня | — |
| 2 | File-cache + project nested counts с confidence-интерполяцией | −1.5…2 бита; ranker MRR +0.05–0.1; OOV-кандидаты получают score | 3–5 дней | память при больших проектах — ограничить top-K файлов/директорий |
| 3 | Сглаживание JM (или AD) вместо MKN; порядок 5–6 | −0.3…0.5 бита | 1–2 дня | у нас квантованный формат — нужны отдельные таблицы для λ |
| 4 | Ranker: 20–40 признаков (prefix-matching детализировано, kind, scope distance, «вызывалось на этом типе в методе», cache-score, accept-log, LM-score по субтокенам) + CatBoost/LightGBM listwise (QuerySoftMax/LambdaRank), экспорт в if-else Kotlin; rerank top-50 | R@1 первого показа +5–10 п.п. | 1–2 недели | нужен сбор размеченных сессий (прокси по истории правок + реальные логи EAP) |
| 5 | Субтокенная backoff-таблица для OOV идентификаторов | закрыть 9/17.5 % слепых кандидатов | 2–3 дня | слабый сигнал, но дешёвый |
| 6 | Pretraining на 700–900 репо через lmplz → ARPA → entropy pruning (Stolcke) → наш формат; fingerprint 16 бит, backoff 4–5 бит | словарь/покрытие ↑, размер ≤10–12 МБ | 1 неделя | LM-часть насыщается — не ждать >0.1 бита от роста данных |
| 7 | Gray text из n-gram: beam 3–5, PSI-ограничение, фильтр показа (confidence + telemetry + синтаксис), метрики AR/RoCC/explicit-cancel | AR 20–25 % на коротких подсказках | 1–2 недели | UX-раздражение при плохом фильтре; начать с консервативного порога |
| 8 | CatBoost-фильтр «показывать ли» на логах (аналог JetBrains 2.5 МБ/1–2 мс) | AR +30–50 %, cancel −40 % | после накопления логов | нужны логи |
| 9 | Малый transformer (10–50M, BPE 8–16k, int8) для gray text, LM-score в ranker'е | AR до ~30–35 %, MRR +3–5 п.п. на ID | 1–2 мес + GPU-обучение (неделя на 8×H100 для 100M; для 30M — дни на 1–2 GPU) | латентность в JVM |

**Когда transformer оправдан и что реалистично на CPU в JVM.** Критерии: (1) шаги 1–7 выполнены и AR gray text упёрлось в ~20 %; (2) есть GPU на неделю обучения и телеметрия для оценки. Размер: 10–50M параметров, 6–8 слоёв, d=384–512, контекст 512–1024. Латентность (моя оценка на основе FLCC-данных: 100M int4 через llama.cpp ≈ 150 мс end-to-end с beam и KV-cache; Svyatkovskiy: GRU 50 МБ — 7.7 мс/подсказка в PyTorch CPU): для 30M int8 с KV-cache префикса ~5–10 мс/токен на одном ядре AVX2, т.е. 5 токенов beam=3 ≈ 50–100 мс — **не укладывается в 30 мс без KV-cache префикса и ограничения beam=1–2**; как **scorer** 500 кандидатов (один forward на контекст + скоринг кандидатов через logits/субтокены) — 10–20 мс, т.е. вне бюджета 5 мс для ranker'а, но пригодно для gray text и для «ленивого» уточнения top-20. Среда: ONNX Runtime Java (JNI, overhead измерен в ~+58 % латентности vs Python в batch-1 из-за конверсий, нативные библиотеки ~30–60 МБ на платформу) vs llama.cpp через JNI (путь JetBrains; gguf int4/int8, лучше CPU-кернелы, но отдельный процесс/нативный бинарь на 3 ОС) vs рукописный инференс на Kotlin (Vector API/ FP32 — для ≤20M параметров реалистично 20–40 мс на 5 токенов, zero-dependency, но int8-GEMM на JVM в 3–5 раз медленнее нативного). Рекомендация: начинать с ONNX Runtime (int8, dynamic quant) для оценки качества; в продакшен — только если выигрыш AR ≥ +10 п.п. над n-gram на A/B.

## Сводная таблица подходов

| Подход | Измерено в литературе | Применимость к нам | Усилия |
|---|---|---|---|
| n-gram JM-6 + file cache | 5.54 → 3.43 бит; MRR 0.56 → 0.69 (Java) | прямо; заменяет MKN | низкие |
| + nested project counts | → 2.57/2.23 бит; MRR 0.745/0.788; без лимита словаря 0.818 | прямо (есть индекс IDE) | низкие–средние |
| BPE-NLM GRU 2048 (Karampatsis) | 1.23 бит; MRR 0.824 (+cache 0.833); ID-MRR +5–9 п.п. vs n-gram; 45–240 МБ | только как будущий transformer, не для n-gram | высокие |
| Neural rerank над static-analysis кандидатами (IntelliCode) | 3 МБ: R@1 66 / R@5 88 / 5 мс; 50 МБ: 70 / 90 / 8 мс | архитектурно = наш ranker с нейро-score | средние |
| CatBoost LTR на логах (JetBrains) | R@1 init 0.634 → 0.799; typing actions −12 %; 366 КБ, +27 мс | прямо; главный рычаг качества списка | средние |
| Entropy pruning (Stolcke) | 26 % размера, +5.7 % PPL | прямо, после lmplz | низкие |
| MPH/fingerprint LM (Guthrie-Hepple) | 1.41–2.47 байт/n-gram | у нас уже ~5; цель 3.5–4 | низкие |
| FLCC 100M transformer int4 (JetBrains) | 150 мс, ~100 МБ, AR 35–38 %, cancel 5 % | ориентир; наш бюджет 15 МБ/30 мс допускает 10–30M | высокие |
| Фильтр показа CatBoost (JetBrains 2025) | 2.5 МБ, 1–2 мс; AR +50 %, cancel −40 % | прямо для gray text | средние (нужны логи) |
| Dedup (Allamanis) | метрики по ID завышены до 39–51 % | обязательно | низкие |

---

## Источники

1. Hellendoorn, Devanbu. Are Deep Neural Networks the Best Choice for Modeling Source Code? FSE'17. https://www.cs.ucdavis.edu/~devanbu/isDLgood.pdf ; SLP-Core: https://github.com/SLP-team/SLP_Core
2. Tu, Su, Devanbu. On the localness of software. FSE'14. https://dl.acm.org/doi/10.1145/2635868.2635875
3. Hindle, Barr, Gabel, Su, Devanbu. On the naturalness of software. ICSE'12 / CACM'16. https://discovery.ucl.ac.uk/id/eprint/1505799/
4. Karampatsis, Babii, Robbes, Sutton, Janes. Big Code != Big Vocabulary. ICSE'20. https://arxiv.org/abs/2003.07914
5. Karampatsis, Sutton. Maybe Deep Neural Networks are the Best Choice for Modeling Source Code. 2019. https://arxiv.org/abs/1903.05734
6. Svyatkovskiy et al. Fast and Memory-Efficient Neural Code Completion. MSR'21. https://arxiv.org/abs/2004.13651
7. Aye, Kaiser. Sequence Model Design for Code Completion in the Modern IDE. 2020. https://arxiv.org/abs/2004.05249
8. Aye, Kim, Li. Learning Autocompletion from Real-World Datasets. ICSE-SEIP'21. https://arxiv.org/abs/2011.04542
9. Li, Wang, Lyu, King. Code Completion with Neural Attention and Pointer Networks. IJCAI'18. https://arxiv.org/abs/1711.09573
10. Bruch, Monperrus, Mezini. Learning from Examples to Improve Code Completion Systems. FSE'09. https://www.monperrus.net/martin/bmn-fse2009.pdf
11. Proksch, Lerch, Mezini. Intelligent Code Completion with Bayesian Networks. TOSEM 25(1), 2015. https://dl.acm.org/doi/10.1145/2744200
12. Raychev, Vechev, Yahav. Code Completion with Statistical Language Models. PLDI'14. https://csaws.cs.technion.ac.il/~yahave/papers/pldi14-statistical.pdf
13. Bibaev et al. (JetBrains). All You Need Is Logs. FSE'22 Industry. https://arxiv.org/abs/2205.10692
14. JetBrains blog, Code Completion Episodes 1–4 (2021). https://blog.jetbrains.com/blog/2021/05/28/code-completion-episode-1-scenarios-and-requirements/ ; https://blog.jetbrains.com/blog/2021/06/04/code-completion-episode-2-why-machine-learning/ ; https://blog.jetbrains.com/blog/2021/08/20/code-completion-episode-4-model-training/
15. Semenkin et al. (JetBrains). Full Line Code Completion: Bringing AI to Desktop. 2024. https://arxiv.org/abs/2405.08704
16. JetBrains blog. Full Line Code Completion in JetBrains IDEs: All You Need to Know (2024). https://blog.jetbrains.com/blog/2024/04/04/full-line-code-completion-in-jetbrains-ides-all-you-need-to-know/ ; Rider: https://blog.jetbrains.com/dotnet/2024/07/10/full-line-code-completion-in-jetbrains-rider/
17. JetBrains blog. Complete the Un-Completable (2024). https://blog.jetbrains.com/ai/2024/10/complete-the-un-completable-the-state-of-ai-completion-in-jetbrains-ides/
18. JetBrains blog. AI Code Completion: Less Is More (2025). https://blog.jetbrains.com/ai/2025/03/ai-code-completion-less-is-more/
19. JetBrains blog. Mellum: How We Trained a Model to Excel in Code Completion (2025). https://blog.jetbrains.com/ai/2025/04/mellum-how-we-trained-a-model-to-excel-in-code-completion/ ; paper: https://arxiv.org/abs/2510.05788
20. de Moor et al. A Transformer-Based Approach for Smart Invocation of Automatic Code Completion. AIWARE'24. https://arxiv.org/abs/2405.14753
21. Control Models for In-IDE Code Completion. 2026. https://arxiv.org/abs/2601.20223
22. Pre-Filtering Code Suggestions using Developer Behavioral Telemetry. 2025. https://arxiv.org/abs/2511.18849
23. GitHub blog. Smarter, more efficient coding: Copilot goes beyond Codex (2023). https://github.blog/news-insights/product-news/smarter-more-efficient-coding-github-copilot-goes-beyond-codex-with-improved-ai-model/ ; acceptance ≈30 %: https://itpro.com/technology/artificial-intelligence/github-30-of-copilot-coding-suggestions-are-accepted
24. Allamanis. The Adverse Effects of Code Duplication in Machine Learning Models of Code. Onward!'19. https://arxiv.org/abs/1812.06469
25. Stolcke. Entropy-based Pruning of Backoff Language Models. 2000. https://arxiv.org/abs/cs/0006025
26. Heafield. KenLM: Faster and Smaller Language Model Queries. WMT'11. https://kheafield.com/papers/avenue/kenlm.pdf ; структуры данных: https://kheafield.com/code/kenlm/structures/
27. Heafield, Pouzyrevsky, Clark, Koehn. Scalable Modified Kneser-Ney Language Model Estimation. ACL'13. https://aclanthology.org/P13-2121/ ; https://kheafield.com/code/kenlm/estimation/
28. Pauls, Klein. Faster and Smaller N-Gram Language Models. ACL'11. https://aclanthology.org/P11-1027/
29. Talbot, Osborne. Smoothed Bloom Filter Language Models: Tera-Scale LMs on the Cheap. EMNLP'07. https://aclanthology.org/D07-1049/
30. Guthrie, Hepple. Storing the Web in Memory: Space Efficient Language Models with Constant Time Retrieval. EMNLP'10. https://aclanthology.org/D10-1026/
31. Tabnine local model (GPT-2, 355M, 1 ГБ RAM) — https://dev.to/tabnine/major-announcement-tabnine-significantly-reduces-memory-footprint-by-using-a-single-shared-process-for-local-inference-g6e
32. ONNX Runtime Java overhead (batch-1, +58 % vs Python) — https://dev.to/e_b680bbca20c348/cross-language-model-inference-without-python-an-engineering-perspective-1i12 ; Fast DistilBERT on CPUs (int8, 2.76 мс при seq 32) — https://arxiv.org/abs/2211.07715
