# Исходная BF16 против NF4: короткая проверка качества

На 32 заранее выбранных фрагментах WikiText-2 test для полной pretrained
Mistral-7B-v0.3 **у NF4 есть небольшое ухудшение метрики**:

| Метрика, меньше лучше | Исходная BF16 | Nested NF4 / BF16 compute |
|---|---:|---:|
| Mean next-token NLL | 1.8874408620 | 1.9187500774 |
| Perplexity | 6.6024504606 | 6.8124381251 |

Delta NLL **+0.0313092154**, относительное изменение perplexity **+3.18045%**.
Во всех32 фрагментах NLL у NF4 выше. Это **не «потеря3.18% качества» вообще**:
измерена конкретная метрика предсказания следующего токена на небольшой английской
выборке. Top1 predictions совпали в93.0406% позиций; это agreement, не accuracy.
Идентичность качества и качество после fine-tuning этим тестом не установлены.

## Что именно сравнили

- Полные исходные веса `mistralai/Mistral-7B-v0.3`, revision
  `caa1feb0e54d415e2df31207e5f4e273e33509b1`; все три shard SHA проверены
  против upstream identities. Это не случайные веса и не дообученная модель.
- [WikiText-2 raw test](https://huggingface.co/datasets/Salesforce/wikitext), revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`, parquet SHA256
  `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`.
  Dataset attribution/license: Salesforce/WikiText, CC BY-SA3.0 / GFDL.
- По [правилу, записанному до аренды](d1-quant-quality-rule-20261019.md):32
  непересекающихся окна по512 токенов равномерно по всему test token stream,
  без добавленных BOS/EOS, всего **16352 scored targets**. Контекст каждого окна
  сбрасывается; это не perplexity полного корпуса с большим sliding context.
- Два свежих процесса: resident BF16 и обычный bitsandbytes nested NF4,
  все224 decoder linears квантизованы; embedding/head остаются BF16.
  Одинаковые токены, SDPA, eval/inference mode, use_cache=False, TF32off, seed842.
  BF16 compute в обоих путях. LoRA и обучения нет.
- RTX3090 sm_86, UUID `GPU-c9f9c922-5bab-5b73-1635-a5c4f933a11e`, driver565.57.01,
  Torch2.11.0+cu126, Transformers5.17.0, bitsandbytes0.50.2, accelerate1.13.0,
  NumPy2.4.3, Python3.12. Частоты не фиксировали: timing verdict здесь не нужен.

## Связь с D1 и польза Soup

Этот прогон измеряет **влияние самой квантизации BF16→NF4**. Он выполнен resident
через standard Transformers/bitsandbytes, **без RAM-стриминга и без D1 kernel**.
Его нельзя выдавать за проверку всего streaming-тренинга или дополнительный
kernel parity gate. Он даёт команде конкретную оценку компромисса NF4 на этой7B.

Отдельно [предыдущий тест Soup RAM streaming](d1-real7b-result-20261019.md)
показал экономию D1 kernel **63.5 MiB allocated /32 MiB reserved** против штатного
NF4 и три bit-exact обновления. Измеренное здесь отличие BF16/NF4 не следует
приписывать kernel: он в этом сравнении не выполнялся. Длительное обучение,
downstream tasks, другие модели/контексты и streaming-versus-resident correctness
остаются открытыми. Полная D1/#842 требует решения владельца о покрытии.

## Воспроизводимость, попытки и бюджет

Executed source **`dc6fa5cc2ef17205320d3b3a2e6408d225751790`**; source archive
содержит только реально выполняемый standalone harness и его правило. Независимый
CPU replay сверил эти два файла с Git commit, model/data/input hashes, все16352
finite NLL/predictions обоих путей и пересчитал NLL/PPL/дельту/agreement.

[Неизменяемый архив](https://github.com/SoupTeam/soup-rnd/tree/33a68aac8c6fec6274866aee836850a4ebb473d4/followups/quant-quality-20261019)
содержит source, rule, derived token IDs, per-token NLL/predictions, environment,
logs, независимый replay/manifest и cleanup receipts. Raw221647 B, SHA256:
`bd03c8e378304acad5c4dcaa89b5a51198de291ff56f21be86289b34396d566a`.
Исходные веса и plaintext датасета не публикуются. Прежние archive blobs неизменны.

Первая аренда56674839 остановилась **до GPU evaluation** на системном pip/PEP668;
предшествующий CRLF wrapper error тоже сохранён. Медленная передача полного repo
была заменена передачей только standalone source до выполнения. Аренду удалили,
ошибки/raw/receipts сохранили. Перед повторной арендой записано инфраструктурное
изменение: venv с Torch из образа, общий лимит обеих попыток$1.50. Численный код,
правило выборки и метрики не менялись, результатов качества до повтора не было.

Вторая аренда56675853, тот же machine22784, прошла оба прогона и удалена после
проверенного экспорта. Guest network guard:14.706 GB из24 GB. Новых личных SSH
регистраций не создавали; доступ на уровне контейнера исчез вместе с ним.
Все D1 ресурсы освобождены. Последний видимый счёт: **D1$9.014/$15**, обе новые
попытки **$0.127**, остаток **$5.986**. Provider billing может обновиться; резерв$1
на прежний трафик сохранён, новые расходы не продолжаются.
