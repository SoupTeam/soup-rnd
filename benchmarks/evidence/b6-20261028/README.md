# B6 evidence

CUDA tests here use tiny synthetic models; they do not establish real giant
training. The selected GPU tests cover frozen expert forward/backward, NF4,
autocast and adapter save/reload. RTX 3070 and RTX 5060 each passed 9 selected
cases. The real-checkpoint pilot is running and will be recorded separately.

The complete CPU test collection attempt on Windows stopped in unchanged E4
modules importing Unix-only `resource`. The separate streaming setup regressions
passed 138 tests with 8 platform skips. See raw logs for scope.
