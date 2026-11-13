**Prepared cache footprint (file bytes, not training VRAM)**

| Format | Complete cache + index GiB | Layer file MiB, min–max | Files | Double quantization |
| --- | ---: | ---: | ---: | --- |
| nf4 | 5.656 | 94.939–94.939 | 40 | yes |
| none | 15.256 | 368.017–368.017 | 40 | no |

Cache bytes include unquantized large tensors and metadata; they do not imply a corresponding VRAM reduction.

**BF16 RAM fixed-count sweep**

| Layers | Capacity screen | Accuracy | Primary / semantic bitwise | Capture allocated / reserved GiB | Sampled free min GiB | Selection only, s/step |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 0 | 3 updates fit | QUALIFIED_TENSORS_REPLAYED | yes / yes | 4.142 / 4.709 | 2.638 | 4.0395 |
| 4 | 3 updates fit | QUALIFIED_TENSORS_REPLAYED | yes / yes | 5.579 / 6.162 | 1.185 | 4.0315 |
| 5 | 3 updates fit | QUALIFIED_TENSORS_REPLAYED | yes / yes | 5.939 / 6.525 | 0.821 | 4.0388 |
| 6 | 3 updates fit | QUALIFIED_TENSORS_REPLAYED | yes / yes | 6.298 / 6.889 | 0.458 | 4.0300 |
| 7 | cap_refusal | NOT_IN_INDEPENDENT_REPLAY | — / — | — / — | — | — |

Capacity screens and selection times are not held-out performance evidence.

**NF4 RAM capacity sweep**

| Layers | Capacity screen | Accuracy | Primary / semantic bitwise | Capture allocated / reserved GiB | Sampled free min GiB | Selection only, s/step |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 0 | 3 updates fit | QUALIFIED_TENSORS_REPLAYED | yes / yes | 3.527 / 4.000 | 3.366 | 3.9943 |
| 4 | 3 updates fit | NOT_IN_INDEPENDENT_REPLAY | — / — | — / — | — | — |
| 8 | 3 updates fit | NOT_IN_INDEPENDENT_REPLAY | — / — | — / — | — | — |
| 12 | 3 updates fit | NOT_IN_INDEPENDENT_REPLAY | — / — | — / — | — | — |
| 16 | 3 updates fit | NOT_IN_INDEPENDENT_REPLAY | — / — | — / — | — | — |
| 20 | 3 updates fit | NOT_IN_INDEPENDENT_REPLAY | — / — | — / — | — | — |
| 24 | 3 updates fit | NOT_IN_INDEPENDENT_REPLAY | — / — | — / — | — | — |
| 28 | 3 updates fit | NOT_IN_INDEPENDENT_REPLAY | — / — | — / — | — | — |
| 32 | 3 updates fit | NOT_IN_INDEPENDENT_REPLAY | — / — | — / — | — | — |
| 33 | 3 updates fit | NOT_IN_INDEPENDENT_REPLAY | — / — | — / — | — | — |
| 34 | 3 updates fit | QUALIFIED_TENSORS_REPLAYED | yes / yes | 6.679 / 7.150 | 0.216 | 3.9409 |
| 35 | cap_refusal | NOT_IN_INDEPENDENT_REPLAY | — / — | — / — | — | — |
| 36 | cap_refusal | NOT_IN_INDEPENDENT_REPLAY | — / — | — / — | — | — |

Capacity screens and selection times are not held-out performance evidence.

**Independent held-out comparisons**

| Comparison | Baseline → candidate s/step | Paired time cut, 95% CI | ≥2% gate | Allocated GiB, baseline → candidate | Reserved GiB, baseline → candidate |
| --- | ---: | --- | --- | ---: | ---: |
| dense-best | 4.0735 → 4.0596 | 0.25% [-0.21, 0.52] | no | 4.142 → 6.298 | 4.709 → 6.889 |
| nf4-maxfit | 4.0528 → 4.0076 | 0.92% [-0.15, 2.63] | no | 3.527 → 6.679 | 4.000 → 7.150 |
| nf4-disk-vs-ram | 4.0281 → 14.0073 | -248.03% [-340.59, -3.21] | no | 3.527 → 3.527 | 4.000 → 4.000 |
| dense-four-reference | 4.0875 → 4.0811 | 0.13% [-1.08, 1.44] | no | 4.142 → 5.579 | 4.709 → 6.162 |

Step medians summarize four fresh processes per arm, eight measured updates each. Time cut is the median of paired reductions; 95% bootstrap resamples whole pairs.

| Comparison | CPU process HWM GiB, baseline → candidate | Measured read GiB/step, baseline → candidate | Disk / RAM step multiplier | NVMe training reads proven |
| --- | ---: | ---: | ---: | --- |
| dense-best | 20.430 → 20.423 | 0.000 → 0.000 | — | — |
| nf4-maxfit | 10.430 → 10.430 | 0.000 → 0.000 | — | — |
| nf4-disk-vs-ram | 10.432 → 8.973 | 0.000 → 12.609 | 3.480× | yes |
| dense-four-reference | 20.422 → 20.431 | 0.000 → 0.000 | — | — |

| Comparison | getrusage HWM GiB, baseline → candidate | /proc VmHWM GiB, baseline → candidate | /proc VmRSS setup GiB, baseline → candidate | /proc VmRSS exit GiB, baseline → candidate |
| --- | ---: | ---: | ---: | ---: |
| dense-best | 20.430 → 20.423 | 20.430 → 20.423 | 19.316 → 19.316 | 19.784 → 19.784 |
| nf4-maxfit | 10.430 → 10.430 | 10.430 → 10.430 | 9.318 → 9.318 | 9.782 → 9.782 |
| nf4-disk-vs-ram | 10.432 → 8.973 | 10.432 → 5.811 | 9.318 → 5.318 | 9.780 → 5.781 |
| dense-four-reference | 20.422 → 20.431 | 20.422 → 20.431 | 19.316 → 19.317 | 19.781 → 19.765 |

CPU columns are per-arm maxima across process snapshots, not whole-machine RAM. The earlier CPU HWM column retains the conservative maximum of both counters. getrusage can retain a pre-exec high-water mark; /proc VmHWM describes the current memory map. Their difference does not establish which earlier code allocated memory.

| Comparison | Decoder loads/step, median [range], baseline → candidate | Large-tensor loads/step, median [range], baseline → candidate | Logical decoder copy GiB/step, baseline → candidate |
| --- | --- | --- | ---: |
| dense-best | 138 [138, 138] → 114 [114, 114] | 4 [4, 4] → 4 [4, 4] | 49.596 → 40.971 |
| nf4-maxfit | 138 [138, 138] → 2 [2, 2] | 4 [4, 4] → 4 [4, 4] | 12.794 → 0.185 |
| nf4-disk-vs-ram | 138 [138, 138] → 138 [138, 138] | 4 [4, 4] → 4 [4, 4] | 12.794 → 12.794 |
| dense-four-reference | 138 [138, 138] → 122 [122, 122] | 4 [4, 4] → 4 [4, 4] | 49.596 → 43.846 |

Load/copy counters use only the last eight updates per process. Private resident layer count alone does not describe additional weights retained in rotating buffers.

**Descriptive complete-run phases (exploratory, not preregistered or gating)**

| Comparison | Phase | Baseline seconds, median [range] | Candidate seconds, median [range] |
| --- | --- | ---: | ---: |
| dense-best | Dataset/model/trainer setup + adapter initialization/digest | 42.161 [36.469, 43.871] | 36.550 [33.201, 43.742] |
| dense-best | All 14 optimizer updates + final adapter save | 58.981 [58.756, 59.341] | 58.627 [58.525, 58.944] |
| dense-best | Streaming setup + private-slot installation | 41.995 [36.277, 43.645] | 36.416 [33.069, 43.532] |
| dense-best | Benchmark entry (cached model) | 107.928 [102.813, 109.793] | 100.282 [98.249, 109.164] |
| dense-best | Worker child-process wall (includes Python launch/imports) | 120.143 [113.129, 121.350] | 111.330 [110.877, 119.535] |
| dense-best | Descriptive 14-update training+save time cuts by pair | 0.149%, 0.685%, 1.244%, 0.179% | median 0.432% |
| nf4-maxfit | Dataset/model/trainer setup + adapter initialization/digest | 33.443 [32.209, 36.972] | 33.286 [32.514, 34.414] |
| nf4-maxfit | All 14 optimizer updates + final adapter save | 58.085 [57.769, 58.254] | 57.597 [57.235, 57.663] |
| nf4-maxfit | Streaming setup + private-slot installation | 33.315 [32.084, 36.834] | 33.160 [32.390, 34.288] |
| nf4-maxfit | Benchmark entry (cached model) | 97.087 [94.718, 99.164] | 95.191 [94.621, 96.569] |
| nf4-maxfit | Worker child-process wall (includes Python launch/imports) | 103.099 [101.089, 105.949] | 102.081 [101.927, 103.191] |
| nf4-maxfit | Descriptive 14-update training+save time cuts by pair | 0.678%, 1.750%, 1.003%, 0.184% | median 0.840% |
| nf4-disk-vs-ram | Dataset/model/trainer setup + adapter initialization/digest | 42.100 [41.538, 43.730] | 31.622 [30.018, 36.278] |
| nf4-disk-vs-ram | All 14 optimizer updates + final adapter save | 58.083 [57.975, 58.569] | 126.801 [86.052, 161.012] |
| nf4-disk-vs-ram | Streaming setup + private-slot installation | 41.958 [41.315, 43.530] | 31.485 [29.824, 36.138] |
| nf4-disk-vs-ram | Benchmark entry (cached model) | 106.683 [105.853, 107.670] | 163.279 [128.740, 195.532] |
| nf4-disk-vs-ram | Worker child-process wall (includes Python launch/imports) | 113.946 [113.108, 114.821] | 168.472 [134.255, 200.852] |
| nf4-disk-vs-ram | Descriptive 14-update training+save time cuts by pair | -48.430%, -141.837%, -93.607%, -176.709% | median -117.722% |
| dense-four-reference | Dataset/model/trainer setup + adapter initialization/digest | 29.370 [28.609, 33.359] | 29.334 [28.925, 31.272] |
| dense-four-reference | All 14 optimizer updates + final adapter save | 58.597 [58.285, 58.635] | 58.443 [58.172, 58.840] |
| dense-four-reference | Streaming setup + private-slot installation | 29.255 [28.496, 33.182] | 29.218 [28.806, 31.157] |
| dense-four-reference | Benchmark entry (cached model) | 92.086 [90.894, 96.359] | 91.739 [91.289, 94.835] |
| dense-four-reference | Worker child-process wall (includes Python launch/imports) | 101.001 [99.187, 105.735] | 101.259 [99.685, 104.500] |
| dense-four-reference | Descriptive 14-update training+save time cuts by pair | -0.369%, 0.316%, 0.681%, -0.258% | median 0.029% |

These windows overlap and must not be summed. Setup uses an already prepared local model/cache. The 14-update training window includes six warmup updates, eight measured updates and the final save; it is not a complete dataset/epoch run. Worker stage additionally includes Python launch/imports. Original model download, shard preparation and result transfer are excluded. These descriptive outcomes have no new winner gate and do not replace the preregistered steady-step ≥2% gate.


**BF16 → NF4 diagnostic (non-gating, lossy format change)**

INVALID_DIAGNOSTIC_EVIDENCE

**Budget and cleanup**

Prior leases $0.765158; current lease $1.580692; aggregate conservative estimate $2.345850 / $10.00; remaining $7.654150.

Instance `57401620` and volume `57401619` independently verified absent. Cost is a conservative ledger estimate, not the provider invoice.
