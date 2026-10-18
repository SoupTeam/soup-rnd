# D1: исходные измерения и архив

Исходные данные вынесены из diff PR #16 в отдельную архивную ветку.
Текущий результат: [итоговый отчёт](d1-final-result-20261016.md).
Код и тесты исследования остаются в PR; архивирование не меняет результаты
и не закрывает открытый критерий скорости (**NO_VERDICT**).

## Сохранность

- Исходный commit: `94bf6605a5a64d59177a019a4f837e01f56f4bd9`.
- Сохранены **240 файлов evidence и 25 исторических документов** (265 файлов).
- ZIP: **278115525 байт**, шесть частей не больше 48 MiB.
- SHA-256 ZIP: `0c07c69bf24b077fd98aaa58b715b84a3c59eed1a0346e8e4002203c00eb1335`.
- Полное восстановление проверено: размер, SHA-256 и Git blob ID каждого файла
  совпадают с исходным commit. Байты JSON не нормализованы.

Сохранены CUDA tensors, телеметрия, исходники выполненных запусков, команды,
версии, данные и их notices, отрицательные контроли, ошибки, отклонённые замеры,
исходные manifests и независимые audits. Вложенные копии внутри старых JSON и
сырых архивов сохранены, чтобы не нарушать их контрольные суммы. Повторяющиеся
копии и прежние status reports убраны из PR, а не из истории исследования.
Двенадцать preregistration amendments остаются также в checkout: harness читает
и хеширует их как входы протокола. Их удаление изменило бы воспроизводимость.

## Скачать и проверить

[Архив и инструкция восстановления](https://github.com/SoupTeam/soup-rnd/tree/fa551398c457fbe52bcb148db90c75c3868b4afd)
закреплены на отдельном commit; архивная ветка не предназначена для merge.

```bash
git clone --depth 1 --single-branch --branch koko/d1-evidence-archive-20261018 \
  https://github.com/SoupTeam/soup-rnd.git d1-evidence-archive
cd d1-evidence-archive
git rev-parse HEAD
# Expected: fa551398c457fbe52bcb148db90c75c3868b4afd
python restore.py --output ../d1-evidence-restored
```

Нужен только Python 3.10+, GPU и обучение не нужны. Скрипт проверяет части,
полный ZIP и каждый файл до восстановления; существующий каталог не перезаписывает.
В архивной ветке также лежат `manifest.json` с полным перечнем файлов и `SHA256SUMS`.
После восстановления пути начинаются с `benchmarks/`. Для прежних replay-команд
скопируйте восстановленный `benchmarks/evidence/` в checkout исследования.

## Где искать результаты

| Путь внутри восстановленного `benchmarks/evidence/` | Содержимое |
|---|---|
| `d1-final-f1-20261016/` | Финальные correctness gates, полный шаг, VRAM, отклонённый ABBA, raw CUDA tensors, audit и manifest |
| `d1-native-20261016/` | Штатная BF16-арифметика, checkpoint/negative control, split-K traces, ранние пилоты |
| `d1-wmma-20261016/` | Предыдущий WMMA-этап с отдельной настройкой накопления и его неуспехи |
| `d1-kernel-screen-20261016/` | Первичный kernel screen и mismatch witnesses |
| `d1-a100-k-20261016/`, `d1-k-precoverage-replay-20261016/`, `d1-efg-replays-20261015/` | Прежняя K-сетка, производные audits и replay |
| `d1-rtx3090-20261015/` | Ранний projection/QA study, raw parts, dataset и notices |
| `d1-preparation-20261015/` | CPU-подготовка и её контроли |

`manifest.json` и `audit.json` финального этапа лежат в первой строке таблицы;
`raw-final-f1.tgz` сохраняет исходный SHA-256
`ec4074e873980c43a0269fa1841789cebf9f1b420bfffc5734c8aed103210675`.
Прежние отчёты и preregistration amendments восстановятся рядом, в `benchmarks/`.
Они фиксируют историю, а текущий вывод нужно читать в итоговом отчёте PR.
