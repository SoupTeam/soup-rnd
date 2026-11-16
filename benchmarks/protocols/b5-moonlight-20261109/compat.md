# Moonlight-16B-A3B: compatibility check without a GPU, 2026-11-09

Pinned revision `476b36a473d4467f94469414bef6cee75c9c8172`. Run on a Mac CPU before any rental,
from the scripts of ticket 01 (`.scratch/b5-moe-scaling/preflight/`). No weight payload was read.

| Check | Result |
| --- | --- |
| Plan from the 27 shard headers (`compat-plan.json`) against the plan of ticket 01 | identical, headers too |
| NF4 cache bound, trainer rule and writer layout | 18 494 839 040 and 15 384 108 832 bytes, as in ticket 01 |
| Source size | 31 920 879 456 bytes, 27 files |
| LoRA targets resolved on the meta skeleton (`q_lora_rank` null) | `q_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`, the list in `soup.yaml` |
| Tokenizer with `trust_remote_code=False` and `tiktoken` 0.14.0 | loads, vocabulary 163 840, pad id 163 838 |
| The four training texts, full length in tokens | 956, 1013, 1592, 1616. Each is cut to exactly 512, so no padding |
| `soup.yaml` against `SoupConfig` | accepted (`tests/test_b5_real_moe.py`) |

The tokenizer is not checked against the Moonshot code: ticket 01 found one text of nine where
`AutoTokenizer` and the Kimi split pattern of Soup gave different ids. The run's report says so.
