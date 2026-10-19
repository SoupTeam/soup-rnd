# D1 — текст для передачи тимлиду

Передаю scoped D1 research result на review:
[PR #16](https://github.com/SoupTeam/soup-rnd/pull/16),
[итоговый валидный F1](d1-vm-f1-result-20261019.md).

Сделан checkpoint-safe fused NF4 forward/dX backend для ограниченного BF16 scope.
Packed weights и scales видимы checkpoint, полный временный BF16 вес не создаётся.
Большой recycled-buffer gate: forward/dX exact, два recompute, отрицательный
контроль в том же процессе обнаруживает ожидаемое конечное отличие. Три полных
обновления синтетического decoder hidden4096/4layers/seq128 точно совпадают с
native streaming и отдельно с непатченным resident NF4: logits, loss, все16
ненулевых LoRA gradients и16 обновлённых параметров на каждом шаге.

**Польза для Soup:** peak allocated ниже на **157 MiB**, reserved — на **140 MiB**
на полном RAM-tier streaming-шаге этого fixture. **Цена:** на RTX 3090 при
фиксированных1200 MHz native **51.568 ms**, кандидат **83.512 ms** за шаг,
то есть кандидат занимает в1.619 раза больше времени. Автоматически включать
его ради ускорения оснований нет; это измеренный компромисс памяти и времени.
На pretrained7B/8B эти числа не переносятся.

**Формальный F1 для этого scope выполнен:** пять fresh-process ABBA-блоков,
20 warmup /200 timed steps на arm,4000 timed steps; ноль отклонённых блоков,
sizing_extra=0. Все20 same-session GEMM references и обязательная телеметрия
валидны; primary/reference SM1200–1200 MHz, power limit420 W неизменен.
Исходные clock-spread блоки и их NO_VERDICT сохранены отдельно без пересмотра.

Независимый CPU replay проверил609 файлов выполненного source `a7ee402c`,
CUDA witness hashes,204 exact finite full-update величины, большой alias control,
timings/медианы и пересчёт исходного validator. На новой VM прошли12 физических
GPU-регрессий. [Raw evidence, команды и receipts](D1_EVIDENCE.md) вынесены из PR;
changelog и [точный список покрытия](D1_ACCEPTANCE.md) приложены.

Все D1 аренды удалены, clocks сброшены, persistence выключен, временный SSH-ключ
отозван. Снимок расходов: новая VM **$0.137**, D1 суммарно **$8.653 из $15**;
billing может обновиться. Дальнейших платных прогонов для этой cohort не нужно.

**Предлагаемый статус: «На проверке — research result».** Осталось решение
владельца: принять ограниченный BF16/RTX 3090 результат, явно оставив остальные
shapes/dtypes/GPU и pretrained target-model coverage открытыми, либо определить
следующий отдельный scope. A100 dX gate не пройден, FP16 и остальные формы не
объявлены поддержанными. Production dispatch не переключён; полное закрытие
D1/#842 без этого решения не заявляется.
