# CUDA graphs для сегментного expert GEMM: исследование по первичным источникам

Дата: 2026-11-19. Версия PyTorch в окружении: 2.14.1+cu130
(`/Users/Asus/Documents/soup-rnd/.venv/lib/python3.11/site-packages/torch`,
`torch/version.py:4`). Документация cuBLAS и CUDA Programming Guide: v13.5.

Контекст в одном абзаце. Harness MBR на Moonlight-16B-A3B (RTX 3060, sm_86,
bf16). Режим `--expert-gemm segmented` вызывает native cuBLAS GEMM для каждого
сегмента микробатча отдельно, чтобы формы M совпали с эталоном A0 и биты
совпали. Цена: около 8 вызовов `F.linear` на эксперта в forward и в VJP, шаг
7,9 с против 5,5-6,1 с без сегментов. Остаток разрыва это число запусков ядер.
Вопрос: можно ли снять накладные расходы запуска через CUDA graphs и сохранить
побитовое равенство с eager-вызовами той же формы.

Обозначения. "Документ говорит" это цитата или близкий пересказ. "Вывод" это
моё рассуждение. "Документация не говорит" значит, что в просмотренных
первичных источниках ответа нет.

## Короткий вывод

Технически можно. Граф, снятый для одной формы (M, N, K), при replay запускает
ровно те ядра с теми аргументами, что были записаны при capture, и cuBLAS заново
ядро не выбирает (вывод из документов PyTorch и CUDA, прямой фразы про cuBLAS
нет). Равенство битов с eager той же формы NVIDIA нигде не гарантирует для
capture: гарантия "те же биты" дана только для той же архитектуры, того же
числа SM и без нескольких активных потоков. Capture идёт на побочном потоке с
другим буфером workspace, и это самое слабое место. Главный риск: число
графов. Граф нужен на каждый точный M (паддинг до бакета меняет форму, а по
документации cuBLAS результат может зависеть от формы), то есть до сотни
графов на тип GEMM, и документация PyTorch сама советует не делать так при
часто меняющихся формах. Решает только эксперимент на 3060.

## 1. Ограничения CUDAGraph, torch.cuda.graph, make_graphed_callables

Источник A: PyTorch docs, "CUDA semantics", раздел "CUDA Graphs"
(https://docs.pytorch.org/docs/2.14/notes/cuda.html, версия 2.14, и
https://docs.pytorch.org/docs/main/notes/cuda.html, версия 2.16 unstable).
Разделы "Constraints" и "Non-constraints" в обеих версиях совпадают.

Документ говорит:

- Capture это запись работы потока. "CUDA work issued to a capturing stream
  doesn't actually run on the GPU. Instead, the work is recorded in a graph."
- Replay: "Each replay runs the same kernels with the same arguments. For
  pointer arguments this means the same memory addresses are used."
- Warmup и поток: "Before capture, warm up the workload to be captured by
  running a few eager iterations. Warmup must occur on a side stream."
- Статические буферы: "you must maintain long-lived references to tensors that
  hold input and output data during capture. To run the graph on new input
  data, copy new data to the capture's input tensor(s), replay the graph, then
  read the new output from the capture's output tensor(s)."
- Capture требует не-default поток. `torch.cuda.graph` ставит побочный поток
  сам.
- Ошибка времени выполнения "likely": "Ops that synchronize the CPU with the
  GPU (e.g., .item() calls) are prohibited." И: "Dynamic control flow (based on
  CPU or GPU data) is prohibited, unless ... torch.cond()".
- Тихие ошибки или неопределённое поведение: "Within a process, only one
  capture may be underway at a time." "No non-captured CUDA work may run in this
  process (on any thread) while capture is underway." "CPU work is not captured.
  If the captured ops include CPU work, that work will be elided during
  replay." "Every replay reads from and writes to the same (virtual) memory
  addresses." "Dynamic shapes are prohibited. The graph assumes every tensor in
  the captured op sequence has the same size and layout in every replay."
- Non-constraint: "Once captured, the graph may be replayed on any stream."
- Память графа, раздел "Graph memory management": аллокатор при capture берёт
  память из "graph-private memory pool". "The private pool stays alive until
  its CUDAGraph object and all tensors created during capture go out of
  scope." По умолчанию отдельный пул на каждый capture. Графы можно связать
  общим пулом (`pool=g1.pool()`), "if you know they'll always be replayed in
  the same order they were captured, and never be replayed concurrently". Там же:
  "It's also safe to share a memory pool across separate graphs that do not
  depend on each other's outputs, provided they never run concurrently ...
  replaying one graph can clobber another graph's outputs when they share a pool,
  unless clone() is called on the outputs beforehand. This pattern is frequently
  used in inference servers that accept variable batch sizes at runtime."
- `make_graphed_callables`: "creates separate graphs of the forward-pass and
  backward-pass work" и "maintains static inputs and outputs as needed"
  (раздел "Partial-network capture").

Исходник, версия 2.14.1, `torch/cuda/graphs.py`:

- `class graph.__enter__`, строки 1159, 1169, 1171: каждый вход в capture
  вызывает `torch.cuda.synchronize()`, `torch.cuda.empty_cache()` и
  `torch._C._host_emptyCache()`. Вывод: capture это тяжёлая операция с полной
  синхронизацией, много captures подряд на горячем пути недопустимы.
- `capture_begin`, строки 576-612: параметр `capture_error_mode`, по умолчанию
  "global". Документ предупреждает не менять его без знания
  `cudaStreamCaptureMode`.
- `make_graphed_callables`, строки 1293-1325: `requires_grad` у sample_args
  должен совпадать с реальными входами; буферы модуля должны иметь
  `requires_grad=False`; нельзя менять набор параметров и буферов; нет double
  backward; "sample_args ... must contain only Tensors".
- `Graphed.forward`, строки 1506-1511: если `data_ptr` входа не совпадает со
  статическим, делается `static_input_surface[i].copy_(inputs[i])`, затем
  `fwd_graph.replay()`, возвращается `o.detach()` от статических выходов.

Источник B: NVIDIA, CUDA Programming Guide v13.5, раздел 4.2 "CUDA Graphs"
(https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html):

- Зачем: "overhead costs can be paid once for the entire graph during
  instantiation, and the graph itself can then be launched repeatedly with very
  little overhead" (введение раздела).
- Раздел 4.2.2.1.2.2 "Prohibited and Unhandled Operations": "It is invalid to
  synchronize or query the execution status of a stream which is being
  captured". Также синхронные API вроде `cudaMemcpy()` недопустимы.
- Раздел 4.2.3 "Updating Instantiated Graphs": можно менять параметры узлов
  (`cudaGraphExecKernelNodeSetParams`) без нового instantiate, но топология
  должна остаться той же. Вывод: форму GEMM так не поменять, потому что ядро
  и сетка выбраны cuBLAS при capture.

## 2. Совпадают ли биты графа и eager для cuBLAS

### Что говорит NVIDIA

Источник C: cuBLAS v13.5, https://docs.nvidia.com/cuda/cublas/index.html.

Раздел 2.1.4 "Results Reproducibility":

- "By design, all cuBLAS API routines from a given toolkit version, generate the
  same bit-wise results at every run when executed on GPUs with the same
  architecture and the same number of SMs. However, bit-wise reproducibility is
  not guaranteed across toolkit versions".
- "This guarantee no longer holds when multiple CUDA streams are active or
  fixed-point emulation is used. If multiple concurrent streams are active, the
  library may optimize total performance by picking different internal
  implementations."
- Причина по документу: "library optimizations in selecting internal workspace
  for the routines running in parallel streams". Обходы: отдельный workspace на
  поток через `cublasSetWorkspace()`, или handle на поток, или
  `cublasLtMatmul` с workspace пользователя, или `CUBLAS_WORKSPACE_CONFIG=:16:8`
  или `:4096:8`. "Any of those settings will allow for deterministic behavior
  even with multiple concurrent streams sharing a single cuBLAS handle."

Раздел 2.1.10 "GEMM Algorithms Numerical Behavior": часть алгоритмов делит
вычисление по K (split-K). "When this type of algorithm is chosen by the cuBLAS
heuristics ... the results of each split is summed deterministically". То есть
порядок редукции зависит от выбранного алгоритма, а выбор делают эвристики.

Раздел 2.1.12 "CUDA Graphs Support": "cuBLAS routines can be captured in CUDA
Graph stream capture without restrictions in most situations." Исключение:
вывод в буфер хоста. "In the case of CUBLAS(LT)_POINTER_MODE_HOST, coefficient
values are captured in the graph." Заметка там же: при capture cuBLAS может
создавать memory nodes через `cudaMallocAsync`; для child graphs это может
сорваться, и совет такой: дать workspace через `cublasSetWorkspace()`.

Раздел 3.1.5 "Experimental: Batch Invariance" (cuBLASLt). Важный текст про
зависимость от формы: "By default, matmul results may depend on the problem
shape: splitting a problem along the M or N dimension and computing the parts
separately may produce results that are not bitwise-identical to the
corresponding parts of the unsplit problem, because the library is free to select
different kernels or different reduction schemes (for example, split-K) for
different shapes." Гарантия режима: "identical only when the matmul descriptor,
the K dimension, the workspace size, and the split-K configuration (if specified)
are identical." Режим только для compute capability 10.x. На sm_86 он вернёт
`CUBLAS_STATUS_NOT_SUPPORTED`. Для нас режим не применим, но он подтверждает
исходное предположение проекта: форма M влияет на ядро, и размер workspace
входит в условия воспроизводимости.

Раздел 2.4.8 `cublasSetWorkspace()`: `cublasSetStream()` "unconditionally resets
the cuBLAS library workspace back to the default workspace pool". Пользовательский
workspace должен быть выровнен хотя бы на 256 байт.

### Что делает PyTorch 2.14.1

Источник D: `aten/src/ATen/cuda/CublasHandlePool.cpp`, тег v2.14.1
(https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/aten/src/ATen/cuda/CublasHandlePool.cpp):

- Строки 165-209, `parseChosenWorkspaceSize`: по умолчанию
  `:4096:2:16:8`, то есть 4096*2 KiB + 16*8 KiB = 8,125 MiB. 32 MiB только для
  compute capability 9, 10, 11, 12. Для sm_86 это 8,125 MiB. Env
  `CUBLAS_WORKSPACE_CONFIG` переопределяет.
- Строки 315-345, `setWorkspaceForHandle`: workspace хранится в карте с ключом
  (handle, stream) и выделяется из caching allocator. Если на потоке workspace
  ещё нет, он выделяется заново.
- Строки 402-446, `getCurrentCUDABlasHandle`: на каждый вызов делается
  `cublasSetStream(handle, stream)` и `setWorkspaceForHandle`. Комментарий
  (437-445): workspace задаётся явно, потому что с CUDA 12.2 cuBLAS иначе
  аллоцирует через `cudaMallocAsync` вне caching allocator.

Документация CUDA semantics 2.14, раздел "cuBLAS workspaces" совпадает с этим
кодом: "For each combination of cuBLAS handle and CUDA stream, a cuBLAS workspace
will be allocated ... not deallocated unless torch._C._cuda_clearCublasWorkspaces()
is called." Документация `main` (2.16 unstable) говорит иначе: по умолчанию
workspace на операцию из caching allocator, есть `TORCH_CUBLAS_WORKSPACE_CACHE=1`,
и "Persistent workspaces must not be used when capturing multiple CUDA graphs on
the same stream." В исходнике v2.14.1 переменной `TORCH_CUBLAS_WORKSPACE_CACHE`
нет (поиск по файлу пуст). Значит, для 2.14.1 верна версия 2.14, а текст `main`
к нашему окружению не относится. В `main` у `CublasHandlePool.cpp` отдельные
capture-handle и workspace на (capture id, handle, stream) (комментарии в
строках 309-356 файла в `main`). Это показывает, что PyTorch сам считает
workspace при capture отдельным ресурсом.

Источник E: `aten/src/ATen/cuda/CUDABlas.cpp`, тег v2.14.1:

- Строки 1306-1318, `gemm_internal<at::BFloat16>`: по умолчанию (backend не
  Cublaslt) вызывается `gemm_internal_cublas<at::BFloat16>`. Это
  `cublasGemmEx` с `CUBLAS_GEMM_DEFAULT_TENSOR_OP` (строки 1153-1172), с флагом
  `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`, если
  `allowBF16ReductionCuBLAS` не разрешает split-K (строки 1134-1146).
  Алгоритм выбирает эвристика cuBLAS внутри вызова, явного алгоритма PyTorch не
  передаёт.
- Если включён backend Cublaslt, путь другой (`bgemm_internal_cublaslt`,
  строки 255-450): там PyTorch сам вызывает `cublasLtMatmulAlgoGetHeuristic` с
  `CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES` и выравниваниями указателей A, B, C
  (строки 355-372, 395-420). Вывод: в Lt-режиме и выравнивание указателей, и
  размер workspace входят в запрос эвристики.

### Ответ на ключевой вопрос раздела

Документ говорит: replay запускает записанные ядра с записанными аргументами
(источник A), а cuBLAS можно ловить в capture (источник C, 2.1.12).

Вывод: эвристика cuBLAS выполняется один раз при capture, при replay выбора
нет. Поэтому вопрос "даёт ли capture тот же алгоритм, что eager той же формы"
сводится к вопросу, одинаков ли контекст вызова при capture и в eager.
Различия контекста, которые я нашёл:

1. Поток. Capture идёт на побочном потоке, eager на основном. Workspace на
   (handle, stream) другой буфер (источник D, 315-345), размер тот же (8,125
   MiB).
2. Выравнивание указателей A, B, C. Входы eager это срезы отсортированных
   строк, в графе это статические буферы пула. Документация не говорит, влияет ли
   выравнивание на выбор ядра в `cublasGemmEx`. Для Lt-пути PyTorch передаёт
   выравнивание в эвристику явно (CUDABlas.cpp 355-363).
3. Активность других потоков во время capture. cuBLAS 2.1.4 предупреждает о
   смене реализации при нескольких активных потоках.

Документация не говорит, что capture и eager дают то же ядро и тот же порядок
редукции. Прямой гарантии нет. Утверждение "биты совпадут" не подтверждено
первичным источником. Его надо проверять на железе (раздел "Что нужно проверить
на железе", эксперименты 1 и 2).

## 3. Динамические формы

Документ говорит (PyTorch, "CUDA semantics", раздел "Constraints"): "Dynamic
shapes are prohibited." Граф привязан к одной форме и к одним адресам.

Рекомендации документов:

- PyTorch, CUDAGraph Trees, раздел "Dynamic Shape Support"
  (https://docs.pytorch.org/docs/2.14/torch.compiler_cudagraph_trees.html):
  "CUDAGraph Trees re-record CUDAGraph for every unique shape of an input
  tensor." "When there are limited shapes (e.g., batch sizes in inference), it is
  profitable to re-record CUDAGraphs. However, if input tensor shapes change
  frequently or even on every invocation, re-recording CUDAGraph may not be
  profitable." Совет: "padding input tensors to a few fixed tensor shapes" и
  `torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True`.
- Тот же раздел о памяти: "Nvidia uses 64 KB of device memory per kernel launch
  in CUDAGraph, up until CUDA 12.4 and Driver Version 550+." У нас CUDA 13.0,
  и документация не даёт числа для новых версий.
- Паддинг в практике из раздела "Sharing memory across captures": серверы
  инференса с переменным batch size держат набор графов и общий пул (там
  названа vLLM как пример).

Почему паддинг меняет биты. Документ говорит (cuBLAS 3.1.5, приведено выше):
результат "may depend on the problem shape", потому что библиотека "is free to
select different kernels or different reduction schemes (for example, split-K)
for different shapes". Вывод: паддинг M=37 до M=128 вызывает GEMM другой формы.
Он может выбрать другое ядро или другую схему редукции, и строки результата
могут отличаться на биты от eager M=37. Для вашей цели (равенство с A0 на
точных M) паддинг в бакеты 8, 32, 128, 512 небезопасен по умолчанию. Исключение
только если эксперимент 3 покажет, что на этом GPU и этих K, N ядро для
паддинг-формы даёт те же биты строк. Прямой гарантии в документации нет.

Исходник, как PyTorch сам поступает с формами. Тег 2.14.1:

- `torch/_inductor/cudagraph_trees.py`, строки 429-488, `cudagraphify_impl`:
  `fn_cache: dict[tuple[int, ...], ...]` (строка 439). Ключ это значения
  int-входов (symint). На каждый новый ключ вызывается `cudagraphify(...)`
  (строка 479), то есть записывается новый граф, и результат кладётся в кэш
  (строка 486). Это подтверждает: отдельный граф на каждый уникальный ключ формы.
- `torch/_inductor/cudagraph_utils.py`, строки 519-541,
  `maybe_warning_due_to_dynamic_shape`: сообщение "CUDAGraph supports dynamic
  shapes by recording a new graph for each distinct input size. Recording too
  many CUDAGraphs may lead to extra overhead ... a) padding inputs to a few fixed
  number of shapes; or b) set ...cudagraph_skip_dynamic_graphs=True".
- `torch/_inductor/config.py`, строки 1950-1956 и 1993-1995:
  `cudagraph_skip_dynamic_graphs = False` ("If False, we will re-record a graph
  for each unique set of shape inputs"), `cudagraph_capture_sizes` (ограничить
  набор записываемых размеров, прочие идут eager), предупреждение после 8 форм
  (`cudagraph_dynamic_shape_warn_limit = 8`). Строка 1991 даёт лимит
  перезаписей по адресам: `cudagraph_unexpected_rerecord_limit = 128`.

`torch.compile(mode="reduce-overhead")`: документ
(https://docs.pytorch.org/docs/2.14/generated/torch.compile.html, параметр
`mode`) говорит, что режим снижает накладные расходы Python через CUDA graphs,
стоит памяти (кэш workspace), и "we only reduce overhead for CUDA only graphs
which do not mutate inputs". Вывод: он не решает задачу "тот же cuBLAS-вызов
по форме". Он компилирует граф через Inductor и записывает граф на каждый
уникальный набор форм. Вызовет ли Inductor для `F.linear` тот же ATen/cuBLAS
вызов, что eager, не проверено первичным источником. Это надо проверять
(эксперимент 6).

Другие способы, найденные в документах:

- `cudaGraphExecKernelNodeSetParams` и `cudaGraphExecUpdate` (CUDA Guide 4.2.3):
  меняют параметры узлов без re-instantiate. Для смены M это не подходит:
  ядро и сетка выбираются cuBLAS при capture. Вывод, без прямого источника
  про cuBLAS.
- Условные узлы графа (CUDA Guide 4.2.4, PyTorch `torch.cond`, раздел "Data
  Dependent Control Flow") дают ветвление по данным GPU. Документация PyTorch
  говорит, что поддержка в backend inductor "is not available yet". Вывод:
  как замену выбору по M рассматривать нельзя, потому что набор ветвей это
  снова набор форм.
- Документация не даёт способа заставить cuBLAS выполнить GEMM переменной формы
  внутри одного графа.

## 4. Autograd, autocast, caching allocator, замороженные веса

Autograd. Документ (раздел "Partial-network capture"): `make_graphed_callables`
строит отдельные графы forward и backward, "autograd-aware, and can be used in
the training loop as direct replacements". Требования (graphs.py 1293-1325):
совпадение `requires_grad` у sample_args с реальными входами, буферы без
`requires_grad`, только тензорные аргументы, нет double backward. Реализация это
`torch.autograd.Function` (graphs.py 1503-1530): forward копирует вход в
статический буфер, если `data_ptr` другой, затем replay. Вывод для проекта: VJP,
который заново считает forward, можно оформить как два отдельных графа или как
ручной `torch.autograd.Function` со своими графами. Backward внутри capture не
запрещён: в `torch.cuda.graph` можно ловить `loss.backward()`
(пример "Whole-network capture" в документации).

Autocast. Документ (graphs.py 1323-1330): "automatic mixed precision is supported
in make_graphed_callables only with disabled caching. The context manager
torch.cuda.amp.autocast() must have cache_enabled=False", код бросает
`RuntimeError` при включённом кэше приведения. Если harness использует bf16
напрямую без autocast, пункта нет. Документация не говорит про кэш приведения
весов внутри ручного `torch.cuda.graph`. Вывод: при autocast внутри capture кэш
привязывал бы приведённую копию к адресу из пула, поэтому безопаснее
`cache_enabled=False` или заранее приведённые веса.

Caching allocator и пул. Документ (раздел "Graph memory management"): аллокации
capture идут из graph-private pool, который живёт, пока жив CUDAGraph и все
тензоры, созданные при capture. Освобождённая или переназначенная память даёт
illegal memory access или порчу значений. Общий пул допустим только для графов,
которые "never be replayed concurrently". Для набора графов по формам, которые
гоняются последовательно, это разрешено, но "replaying one graph can clobber
another graph's outputs when they share a pool, unless clone() is called on the
outputs beforehand".

Копирование весов эксперта в буфер пула. Документ говорит (раздел "CUDA
Graphs"): "By filling input memory with new data (e.g., from a new batch) before
each replay, you can rerun the same work on new data." Адрес буфера постоянный,
содержимое меняется: это штатный сценарий. Вывод: если веса эксперта i копируются
в статический входной буфер перед replay графа, это допустимо. Условия: копия
идёт в том же потоке до replay, буфер не освобождается, и граф не работает
параллельно с чем-то, что пишет в буфер. CUDAGraph Trees различают три типа
входов (раздел "Input mutation ..."): входы из eager копируются в статический
адрес; параметры и буферы "we assume (and runtime-check) have the same tensor
addresses on every execution", и их содержимое не копируется. Эта проверка
(`check_static_inputs_are_stable`, cudagraph_trees.py 1228-1244) требует, чтобы
адрес совпал. Значит, веса можно либо оставить на их родном адресе (граф на
эксперта, адрес привязан к экспертам), либо копировать в общий статический
буфер (один граф на форму, одна копия на вызов). Копия весов это лишняя
трафик-нагрузка: документация чисел не даёт.

Замороженные веса. Документация не говорит об особых правилах для весов с
`requires_grad=False`. Для `make_graphed_callables` буферы обязаны быть без
grad. Вывод: у замороженных весов во backward нужен только dgrad, и граф
backward для них короче (нет wgrad), это уменьшает число ядер на граф.

## 5. Стоимость capture и число графов

Чисел в документах мало. Что есть:

- Время capture: документация PyTorch и NVIDIA не даёт. Известно только
  устройство: каждый вход `torch.cuda.graph` делает `synchronize` и
  `empty_cache` (graphs.py 1159-1171), плюс инстанцирование графа
  (`cudaGraphInstantiate`, CUDA Guide 4.2.2.2, "one-time cost" в примере кода).
  Время одного capture надо мерить (эксперимент 4).
- Память графа: PyTorch, CUDAGraph Trees, "Dynamic Shape Support": "Nvidia uses
  64 KB of device memory per kernel launch in CUDAGraph, up until CUDA 12.4 and
  Driver Version 550+". Для CUDA 13.0 числа в первичных источниках нет. Пул для
  выходов и workspace считается отдельно (см. раздел 4).
- Workspace cuBLAS: 8,125 MiB на (handle, stream) для sm_86 (источник D). Один
  дополнительный поток capture создаёт ещё один такой workspace.

Оценка числа графов (формула, не измерение). Ключ графа = (форма сегмента M,
тип операции). Для одного типа GEMM число графов равно числу различных M,
встреченных в шаге. Верхняя граница по условию задачи: M от 1 до около 240, то
есть до 240 графов на тип (gate, up, down, и отдельно forward и dgrad в VJP).
Если типов GEMM g, то до 240*g графов. Число вызовов в шаге (около 1570 вызовов
экспертов, до 8 сегментов на вызов) задаёт число replay, а не число графов.
Паддинг в бакеты 8, 32, 128, 512 дал бы 4*g графов, но меняет формы GEMM и по
разделу 3 небезопасен для равенства битов без проверки. Сколько различных M
реально встречается в шаге, надо взять из лога harness (эксперимент 5).

Граф всего эксперта целиком (gate, up, silu, down) на один M уменьшает число
запусков сильнее, чем граф на один GEMM. Тогда ключ это M, а graph содержит
все ядра эксперта. Это вывод, а не рекомендация источников. Он совместим с
правилом "статические адреса": веса копируются в буфер пула, вход копируется в
статический буфер.

## Что нужно проверить на железе

Все эксперименты на RTX 3060, torch 2.14.1+cu130, bf16, те же формы, что в
режиме segmented. Начинать стоит с экспериментов 1 и 2: они дешёвые и решают
вопрос "можно ли вообще".

1. Бит-равенство capture против eager на одной форме. Взять реальные K, N
   эксперта и M из {1, 7, 37, 128, 240}. Eager: `F.linear(x, w)`. Граф: capture
   на побочном потоке (warmup 3 итерации), replay с копией x и w в статические
   буферы. Сравнить `torch.equal` для forward и dgrad (backward по x). Успех:
   полное совпадение на всех M, 100 случайных входов на M. Любое различие
   закрывает идею (или ставит условие, см. 2).
2. Влияние контекста capture. Повторить 1 с (a) `CUBLAS_WORKSPACE_CONFIG`
   `:4096:2:16:8` по умолчанию и с `:4096:8`; (b) входом с другим выравниванием
   (срез со смещением в 2 байта против выровненного на 512); (c) capture на
   потоке, который уже использовался для eager. Успех: те же биты во всех
   вариантах. Различия по (b) покажут, что выравнивание влияет на ядро.
3. Паддинг. Для M из {1, 5, 37, 100, 240} посчитать GEMM с M, паддингом до
   ближайшего бакета (8, 32, 128, 512) и сравнить первые M строк с eager точной
   формы. Успех: биты равны на всех M и в обеих GEMM (forward и dgrad). Одно
   различие означает, что паддинг запрещён и нужен граф на точный M.
4. Стоимость capture и памяти. Для 20 разных M измерить время
   `torch.cuda.graph` (с синхронизацией), `torch.cuda.memory_allocated` и
   `mem_get_info` до и после, размер пула (`torch.cuda.memory_stats` по пулу
   графа). Успех: время и память на граф, по которым считается оценка раздела 5.
   Отдельно: прирост памяти от второго потока capture (workspace 8,125 MiB).
5. Распределение M в реальном шаге. Залогировать M каждого сегмента за один
   шаг без изменения вычислений. Успех: число различных M и доля replay на
   самые частые M. Если различных M больше около 100 на тип, граф на точный M
   не окупится.
6. Выигрыш по времени. Для эксперта целиком (gate, up, silu, down, forward) на
   M из эксперимента 5 сравнить eager-цикл с replay графа за 1000 повторов.
   Успех: время replay существенно меньше eager (оценка по остатку разрыва шага:
   7,9 с против 5,5-6,1 с). Отдельно учесть копию весов в буфер пула.
7. `torch.compile(mode="reduce-overhead")`. Для той же функции записать, вызывает
   ли Inductor тот же ATen mm, и сравнить биты с eager (как в 1). Успех:
   совпадение. Иначе путь закрыт. Дополнительно посмотреть
   `cudagraph_skip_dynamic_graphs` и `cudagraph_capture_sizes`.
8. Весь шаг. Включить граф-путь в harness и сравнить биты потерь, градиентов и
   обновлений адаптера с A0 на тех же данных, что в диагностике MBR. Успех:
   равенство, как у режима segmented без графов.

## Источники

Документация PyTorch:

- https://docs.pytorch.org/docs/2.14/notes/cuda.html, "CUDA Graphs"
  (Constraints, Non-constraints, Graph memory management, Sharing memory across
  captures, Partial-network capture), "cuBLAS workspaces".
- https://docs.pytorch.org/docs/main/notes/cuda.html (2.16 unstable): тот же
  раздел; отличается текстом про `TORCH_CUBLAS_WORKSPACE_CACHE`.
- https://docs.pytorch.org/docs/2.14/torch.compiler_cudagraph_trees.html,
  "Dynamic Shape Support", "Input mutation support".
- https://docs.pytorch.org/docs/2.14/generated/torch.compile.html, параметр
  `mode`.

Документация NVIDIA:

- https://docs.nvidia.com/cuda/cublas/index.html (v13.5): 2.1.4 Results
  Reproducibility, 2.1.10 GEMM Algorithms Numerical Behavior, 2.1.12 CUDA
  Graphs Support, 2.4.8 cublasSetWorkspace(), 3.1.5 Experimental: Batch
  Invariance.
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
  (v13.5): 4.2.2.1.2 Stream Capture, 4.2.2.1.2.2 Prohibited and Unhandled
  Operations, 4.2.2.2 Graph Instantiation, 4.2.3 Updating Instantiated Graphs.

Исходники PyTorch:

- `torch/cuda/graphs.py`, 2.14.1 (локально, `torch/version.py:4`):
  строки 576-612, 691-712, 1081-1171, 1222-1330, 1492-1530.
- `torch/_inductor/cudagraph_trees.py`, 2.14.1: строки 429-488, 1228-1244.
- `torch/_inductor/cudagraph_utils.py`, 2.14.1: строки 519-541.
- `torch/_inductor/config.py`, 2.14.1: строки 1950-1995.
- https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/aten/src/ATen/cuda/CublasHandlePool.cpp
  строки 165-209, 315-345, 402-446.
- https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/aten/src/ATen/cuda/CUDABlas.cpp
  строки 355-420, 1126-1172, 1187-1189, 1306-1318.
- https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/cuda/CublasHandlePool.cpp
  (ветка main, только как сравнение; строки 63-113 и 309-356).

## Где источники молчат

- Даёт ли capture то же ядро и тот же порядок редукции, что eager той же
  формы: прямой гарантии нет (разделы 2, эксперименты 1 и 2).
- Влияние выравнивания указателей на выбор ядра в `cublasGemmEx` (не в Lt):
  документация не говорит.
- Время capture одного графа и память графа на CUDA 13.0: чисел нет.
- Поведение autocast-кэша внутри ручного `torch.cuda.graph`: документация не
  говорит (есть только запрет для `make_graphed_callables`).
- Вызывает ли Inductor для `F.linear` в `reduce-overhead` тот же ATen mm:
  не подтверждено первичным источником.
