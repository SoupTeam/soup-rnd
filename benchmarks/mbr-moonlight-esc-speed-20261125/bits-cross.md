# Bits cross E against S (ticket 16)

8 of 8 cells are bit-equal on the loss and all 216 LoRA gradients.

| k | S | loss equal | grads differing | max abs | E==A0 | S==A0 | dataset sha |
|---|---|---|---|---|---|---|---|
| 1 | 64 | yes | 0 / 216 | 0.0e+00 | True | True | agrees |
| 2 | 64 | yes | 0 / 216 | 0.0e+00 | True | True | agrees |
| 4 | 64 | yes | 0 / 216 | 0.0e+00 | True | True | agrees |
| 8 | 64 | yes | 0 / 216 | 0.0e+00 | True | True | agrees |
| 1 | 512 | yes | 0 / 216 | 0.0e+00 | True | True | agrees |
| 2 | 512 | yes | 0 / 216 | 0.0e+00 | True | True | agrees |
| 4 | 512 | yes | 0 / 216 | 0.0e+00 | True | True | agrees |
| 8 | 512 | yes | 0 / 216 | 0.0e+00 | True | True | agrees |
