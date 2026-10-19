# Original BF16 versus nested NF4: short quality screen

Separate from D1 kernel parity, RAM streaming and synthetic F1. This compares
resident original Mistral-7B-v0.3 against standard bitsandbytes nested NF4/BF16 on
32 fixed, evenly spaced WikiText-2 raw test windows, 512 tokens each, 16352 scored
targets. No LoRA, training, experimental kernel or timing verdict. Read audit.json
for measured NLL, perplexity and the descriptive difference; no equivalence gate.

See benchmarks/d1-quant-quality-rule-20261019.md in the executed source for exact
model/data revisions and sampling. Original weights were SHA-verified. Original
weights and WikiText plaintext are not redistributed. WikiText attribution and
license: https://huggingface.co/datasets/Salesforce/wikitext (CC BY-SA 3.0 / GFDL).
Derived token IDs, NLLs/predictions, offsets, hashes, source, environment and logs
are in raw.tgz. All earlier archive blobs remain unchanged.

With NumPy installed and a Soup checkout containing the executed source commit:
`python replay.py --repo /path/to/soup` verifies archive/source, dataset and model
hashes, input identity, complete finite coverage and recomputes both metrics and
their difference on CPU. It does not perform another GPU run. The exact labelled
rental is released; no personal SSH registration was created. The first rental
stopped before GPU evaluation on the system pip restriction and was released;
its raw logs and receipts are retained separately. The second uses an isolated
venv sharing the image's existing Torch; numerical code and sampling are unchanged. Provider billing
can lag; budget.json distinguishes the visible charge from the cap.
