# Bounded pretrained Mistral7B screen, 2026-10-19

This is separate from the completed synthetic five-block F1. It uses the full
pretrained mistralai/Mistral-7B-v0.3 checkpoint revision
caa1feb0e54d415e2df31207e5f4e273e33509b1, Soup RAM streaming, nested NF4/BF16,
two pageable buffers and unchanged scoped WMMA routing. Read audit.json and raw
records for the actual outcome; preparation is not execution or acceptance.

The rule stops after numerical parity failure. Only after three exact updates,
one short fresh-process ABBA block (5warmup/20timed per arm) is allowed. Even a
clean block is descriptive screening, not formal F1 or long-run quality evidence.
No synthetic40% saving or pretrained quality claim is assumed.

The first preparation stopped because Soup rejects HF cache symlink shards. Its
source/run/logs are preserved under results/attempt-01-preparation in the raw
archive. Before real7B training, an I/O-only commit copied cached bytes locally;
no weights were downloaded again and no numerical kernel, rule or reference was
changed. io-preparation-repair.json records both source identities. The passed
same-GPU native/large-alias gates ran on the original source; their numeric source
is identical to the subsequent real7B source.

booking-plan is the preallocation record; resource_refresh explains selection of
an available replacement VM with a shorter45-minute resource allowance before
CUDA. Exact-ID guards, a guest23GB network stop and clocks/persistence cleanup
bound expenses. API invoices may lag. All prior archive blobs remain unchanged.

raw-parts.json records byte-exact48MiB parts and the server raw SHA. To replay on
CPU with PyTorch installed, use a Soup checkout containing the executed source
commit and run `python replay.py --repo /path/to/soup`. This assembles raw.tgz,
verifies SHA, safely extracts, verifies executed source and saved CUDA witnesses,
and checks parity and any completed timing screen. No GPU rental is needed.
Original model weights, secrets, private SSH keys and account snapshots are not
published. Source/model hashes, natural-text input IDs and derived CUDA tensors
are retained for review.
