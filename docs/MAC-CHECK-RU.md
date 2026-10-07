# Проверка нейроинференса на своей машине (macOS / Windows / Linux)

Что проверяем: (1) загружается ли наша нативная библиотека ядер и проходит ли её самопроверка на этом CPU (на Apple Silicon —
NEON, на x86 — AVX2/AVX-512); (2) совпадает ли Kotlin-инференс с PyTorch на реальной модели; (3) какая скорость.
Всё, что нужно, лежит в репозитории: модель `models/cs-nn-31m-e1.cml`, словарь `models/cs-16384.bpe`, эталонные ответы
PyTorch `models/parity-cs/`. Нужны JDK 21 и ~2 ГБ свободной памяти; интернет не нужен (кроме первого прогона Gradle).

## 1. Нативные ядра и базовые тесты (~2 мин)

```sh
git clone git@github.com:dvislobokov/idea-ml-completion.git && cd idea-ml-completion
./gradlew :ml-core:test --tests '*NativeNnKernelsTest*' --tests '*NnModelTest*' -i 2>&1 | grep -E 'native:|SKIP|PASSED|FAILED|tests completed'
```

Ожидаемый вывод содержит строку вида
`native: loaded libcmlkernels-macos-arm64.dylib, ISA NEON+DotProd (detected NEON+DotProd)` — библиотека загрузилась, самопроверка
прошла, ядра NEON работают. Если вместо этого `SKIP native kernels unavailable: …` — пришлите строку целиком: там причина
(не загрузилась библиотека, Gatekeeper, самопроверка упала на ISA …). Тесты `NnModelTest` при этом всё равно должны быть
зелёными — они проходят на скалярных Kotlin-ядрах.

Если macOS блокирует библиотеку (сообщение про «developer cannot be verified»): `xattr -dr com.apple.quarantine ml-core/` и повторить;
это именно то, что нужно знать перед релизом.

## 2. Паритет с PyTorch на реальной модели (~3 мин; каталог `models/parity-cs/` появится отдельным коммитом — если его ещё нет, пропустите шаг)

```sh
export CML_NN_MODEL=$PWD/models/cs-nn-31m-e1.cml
export CML_BPE_VOCAB=$PWD/models/cs-16384.bpe
export CML_NN_PARITY=$PWD/models/parity-cs
./gradlew :ml-core:test --tests '*NnParityTest*' -i 2>&1 | grep -E 'parity|kernel|argmax|identical|PASSED|FAILED'
```

Тест прогоняет 32 промпта и 40 позиций через три пути (скалярный Kotlin, native f32, native q8) и сравнивает с эталоном:
ожидается argmax 32/32 и идентичные жадные продолжения для scalar/f32, для q8 — почти все. Падение теста на q8 при зелёных
scalar/f32 означает, что на этой архитектуре int8-ядро считает иначе — тоже важный результат, пришлите вывод.

Windows: те же команды в PowerShell (`$env:CML_NN_MODEL = "$PWD\models\cs-nn-31m-e1.cml"` и т. д., `.\gradlew.bat`).

## 3. Скорость (по желанию, ~2 мин)

```sh
./gradlew :ml-core:printBenchClasspath -q > /tmp/cp.txt
java -cp "$(cat /tmp/cp.txt)" io.github.completionml.core.nn.NnBench --file models/cs-nn-31m-e1.cml --threads 4 --kernels scalar
java -cp "$(cat /tmp/cp.txt)" io.github.completionml.core.nn.NnBench --file models/cs-nn-31m-e1.cml --threads 4 --kernels best
```

Печатает prefill / decode / line / reuse8 в мс (протокол как в `docs/NATIVE-KERNELS.md`). На сервере (EPYC, 8 потоков) было
530 / 167 мс на строку для скалярного / native q8; интересно, сколько даст M-серия с 4–8 потоками.

## Результаты на MacBook Pro M1 Pro, 32 ГБ (2026-10-07)

Шаг 1: `native: loaded libcmlkernels-macos-arm64.dylib, ISA neon-dotprod (detected neon-dotprod)` — Gatekeeper не вмешался,
NEON+DotProd прошли самопроверку. Шаг 3, 4 потока, реальная модель cs31m, строка целиком / при наборе: скалярный Kotlin 252 / 40 мс,
native f32 164 / 49 мс, native q8 68 / 17 мс (`--kernels best` выбирает q8 сам). Шаг 2 (паритет) — ожидается.

## Что прислать

Три фрагмента вывода: строку `native: …` (или `SKIP …`), итог `NnParityTest`, и таблицу `NnBench`. Плюс модель Mac
(чип, число ядер) и версию macOS.
