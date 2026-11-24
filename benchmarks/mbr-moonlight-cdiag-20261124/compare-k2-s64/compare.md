# Boundaries: A0 native (a0-k2-s64) vs C native (c-k2-s64)

First divergent boundary: **{'kind': 'expert_linear', 'layer': 1, 'expert': 1, 'which': 'down', 'phase': 'forward', 'call': 3, 'b_rows': 31, 'a_rows': [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 15, 15, 16, 16], 'matched_rows': 31, 'diff_rows': 31}**

## Module boundaries

Matched 266 against 266 records by (layer, role, input digest); equal: **False**; mismatched: 0; inputs absent from the reference: 250.

## Routed-expert node

Node calls: 52; rows: 6656; matched by input row digest: 256; unmatched: 0; calls whose output rows differ: 1.

## Expert linear calls

Right-hand expert calls: 4928; rows: 79872; matched by row input digest: 2594; unmatched inputs: 0; calls with a differing output row: 62.

### Forward call row counts

- A0 native (a0-k2-s64): M=1 976x, M=2 610x, M=3 414x, M=4 342x, M=5 228x, M=6 174x, M=7 176x, M=8 142x, M=9 104x, M=10 84x, M=11 88x, M=12 70x, M=13 60x, M=14 58x, M=15 52x, M=16 24x, M=17 38x, M=18 32x, M=19 40x, M=20 34x, M=21 26x, M=22 28x, M=23 38x, M=24 28x, M=25 26x, M=26 24x, M=27 30x, M=28 20x, M=29 26x, M=30 26x, M=31 28x, M=32 14x, M=33 24x, M=34 20x, M=35 14x, M=36 10x, M=37 10x, M=38 18x, M=39 18x, M=40 8x, M=41 8x, M=42 2x, M=43 12x, M=44 12x, M=45 2x, M=46 4x, M=47 6x, M=48 8x, M=49 6x, M=50 8x, M=51 18x, M=52 4x, M=53 4x, M=54 10x, M=55 10x, M=56 12x, M=57 8x, M=58 4x, M=59 6x, M=60 4x, M=62 8x, M=63 16x
- C native (c-k2-s64): M=1 318x, M=2 280x, M=3 192x, M=4 164x, M=5 150x, M=6 128x, M=7 92x, M=8 82x, M=9 74x, M=10 68x, M=11 68x, M=12 44x, M=13 40x, M=14 48x, M=15 46x, M=16 44x, M=17 18x, M=18 26x, M=19 22x, M=20 16x, M=21 14x, M=22 28x, M=23 8x, M=24 16x, M=25 14x, M=26 8x, M=27 20x, M=28 10x, M=29 14x, M=30 8x, M=31 12x, M=32 16x, M=33 8x, M=34 6x, M=35 6x, M=36 10x, M=37 8x, M=38 6x, M=39 16x, M=40 12x, M=41 10x, M=42 8x, M=43 8x, M=44 6x, M=45 2x, M=46 8x, M=47 10x, M=48 4x, M=49 10x, M=50 6x, M=51 6x, M=52 4x, M=53 2x, M=54 8x, M=55 8x, M=56 4x, M=57 8x, M=58 16x, M=59 8x, M=60 2x, M=61 12x, M=62 4x, M=63 8x, M=64 8x, M=65 4x, M=68 2x, M=69 6x, M=70 8x, M=71 2x, M=72 2x, M=73 6x, M=74 2x, M=75 4x, M=77 2x, M=78 4x, M=79 2x, M=80 2x, M=81 4x, M=83 6x, M=84 2x, M=85 2x, M=86 6x, M=87 2x, M=89 4x, M=91 4x, M=93 2x, M=94 2x, M=95 2x, M=96 4x, M=101 2x, M=102 2x, M=105 2x, M=106 2x, M=107 4x, M=109 2x, M=110 4x, M=112 2x, M=113 2x, M=116 4x, M=117 12x, M=119 6x, M=121 2x, M=126 2x

### Differing expert calls

| Layer | Expert | Projection | Phase | Call | Left M | Right M | Rows | Differing rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | down | forward | 3 | [15, 16] | 31 | 31 | 31 |
| 1 | 4 | down | forward | 9 | [9, 13] | 22 | 22 | 22 |
| 1 | 6 | gate_up | forward | 12 | [1, 7] | 8 | 8 | 1 |
| 1 | 7 | down | forward | 15 | [1, 4] | 5 | 5 | 1 |
| 1 | 12 | down | forward | 25 | [12, 16] | 28 | 28 | 28 |
| 1 | 13 | gate_up | forward | 26 | [25, 38] | 63 | 63 | 52 |
| 1 | 13 | down | forward | 27 | [25, 38] | 63 | 11 | 6 |
| 1 | 19 | gate_up | forward | 34 | [1, 2] | 3 | 3 | 1 |
| 1 | 23 | gate_up | forward | 42 | [29, 30] | 59 | 59 | 40 |
| 1 | 23 | down | forward | 43 | [29, 30] | 59 | 19 | 19 |
| 1 | 24 | gate_up | forward | 44 | [1] | 2 | 2 | 1 |
| 1 | 24 | down | forward | 45 | [1] | 2 | 1 | 1 |

### First differing row, numbers

{'bitwise_equal': False, 'n_diff': 803, 'numel': 2048, 'max_abs_diff': 0.0001220703125, 'max_abs': 0.0291748046875, 'rel_max_diff': 0.0041841004184100415, 'input_sha': '27327b630bc8fff7b1a85c199300aa65d56e61ab6bf107426067357d292f901c'}
