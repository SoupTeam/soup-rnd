# All four held-out timing pairs

A = ordinary SSD; B = RAM hotset.

| Pair | Order | A median s | B median s | Step time reduction | A/B complete entry s |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | AB | 71.737138 | 59.430755 | 17.154829% | 485.551 / 428.487 |
| 1 | BA | 71.684275 | 59.418412 | 17.110953% | 486.099 / 427.904 |
| 2 | BA | 71.762288 | 59.412374 | 17.209476% | 487.411 / 426.761 |
| 3 | AB | 71.726460 | 59.333707 | 17.277799% | 485.654 / 424.725 |

All four measured times in each process (seconds); warmups retained in original metrics:

- Pair 0 A: 71.738502426, 71.740600934, 71.735773458, 71.713433933
- Pair 0 B: 59.335892975, 59.564571499, 59.478763320, 59.382746301
- Pair 1 A: 71.671851121, 71.696698505, 71.747785268, 71.669008454
- Pair 1 B: 59.350218227, 59.404425478, 59.432398656, 59.473790430
- Pair 2 A: 71.744497927, 71.761180667, 71.763394642, 71.772207332
- Pair 2 B: 59.308647844, 59.354056143, 59.470692120, 59.550324259
- Pair 3 A: 71.684044828, 71.690525911, 71.770778939, 71.762394699
- Pair 3 B: 59.329698898, 59.330309737, 59.673300722, 59.337103351

The JSON and evidence archive retain RAM/VRAM, setup, I/O, all six losses and final adapters.
