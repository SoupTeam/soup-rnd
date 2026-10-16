# SQuAD 2.0 data attribution

The `train.jsonl`, `validation.jsonl` and `test.jsonl` files are selected,
flattened records from SQuAD 2.0 by Pranav Rajpurkar, Robin Jia and Percy Liang.
Context, question and answer text is retained; fields are reorganized and the
fixed D2 manifest order is restricted to 256/128/64 records. These are local
research splits, not the unreleased official SQuAD test set.

The source dataset is distributed under
[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/),
as stated on the [official SQuAD site](https://rajpurkar.github.io/SQuAD-explorer/).
The adapted dataset records retain that license. This data notice does not
change the Apache-2.0 license of the Soup implementation.

Sources and byte hashes are pinned in `d2-source-split-manifest.json` at
SQuAD-explorer revision `eee5fdbf62f8613a7812b03419e6b29617b74fd1`.
`dataset.json` pins the selected files, model revision and tokenizer hashes.
