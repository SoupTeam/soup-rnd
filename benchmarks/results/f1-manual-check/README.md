# F1 manual check: RTX 3050, real

Four arm records from the manual check in `benchmarks/METHOD.md`, run on
2026-10-09 on the RTX 3050 laptop. Each one makes a real event happen while
`ArmWatch` watches the arm. Every record has the label "RTX 3050, real".

| File | Event during the arm | Arm length | Outcome | Check that caught it |
|---|---|---|---|---|
| `clean.json` | none, on mains, machine left alone | 60 s | ok | none; all three checks ok |
| `sleep.json` | `systemctl suspend`, asleep about 34 s | 120 s | void | `suspend`: boot-time offset grew 33.827 s; suspend count rose from 1 to 2 |
| `battery.json` | charger unplugged, then plugged back in | 60 s | void | `power`: 9 of 32 power samples off mains, about 18 s |
| `foreign-read.json` | `dd` read a 2 GiB file with `iflag=direct` | 60 s | void | `foreign_reads`: 2,147,483,648 bytes on `nvme0n1` |

In every void record the other two checks are ok, so each event voided its
own check only.

The foreign-read record names the reader `zsh`, not `dd`. The `dd` ran in a
subshell and exited inside the arm, and Linux adds a reaped child's read
counter to its parent's. METHOD.md describes this under "Readers and
unattributed bytes".

## Machine and versions

| Item | Value |
|---|---|
| Laptop | ASUS TUF Gaming F15 FX507ZC4 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, driver 595.99.02, power limit 80 W |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic |
| Python | 3.14.4, the system `python3`, no venv |
| systemd | 259 (259.5-0ubuntu3.4) |
| dd | uutils coreutils 0.10.0 |
| Code | `benchmarks/harness/arm_validity.py` at commit `64724a82` |
| Disk under test | `nvme0n1`, holding both `~/.cache/huggingface/hub` and `$HOME` |

## Commands

Setup, from the repository root:

```bash
export PYTHONPATH="$PWD/benchmarks/harness"
MODELS=~/.cache/huggingface/hub
LABEL="RTX 3050, real"
OUT=benchmarks/results/f1-manual-check
mkdir -p "$OUT"
python3 -c "from arm_validity import disk_device; import os, sys; print(disk_device(os.path.expanduser('~')), disk_device(sys.argv[1]))" "$MODELS"
# nvme0n1 nvme0n1
```

`/tmp/watch_arm.py` is the script from the manual check in METHOD.md, unchanged.

Clean, started 16:30:16 +05:

```bash
python3 /tmp/watch_arm.py clean 60 "$LABEL" "$MODELS" > "$OUT/clean.json"
```

Foreign read, started 16:31:32 +05. The test file was written and synced
before the arm started, and deleted after it ended:

```bash
dd if=/dev/urandom of=$HOME/f1-read-test.bin bs=1M count=2048 conv=fsync
( python3 /tmp/watch_arm.py foreign-read 60 "$LABEL" "$MODELS" > "$OUT/foreign-read.json" &
  ARM=$!; sleep 5; dd if=$HOME/f1-read-test.bin of=/dev/null bs=1M iflag=direct; wait $ARM )
rm $HOME/f1-read-test.bin
```

Sleep, started 16:40:14 +05. The script starts the arm, suspends 10 s in, and
waits for the arm. A person woke the laptop:

```bash
python3 /tmp/watch_arm.py sleep 120 "$LABEL" "$MODELS" > "$OUT/sleep.json" &
arm=$!
sleep 10
systemctl suspend
wait "$arm"
```

Battery, started 16:43:56 +05. A person unplugged the charger when the
script printed its prompt, 5 s in:

```bash
python3 /tmp/watch_arm.py battery 60 "$LABEL" "$MODELS" > "$OUT/battery.json" &
arm=$!
sleep 5
echo "UNPLUG THE CHARGER NOW. Count to 10, then plug it back in."
wait "$arm"
```

## Records not kept

A first sleep arm of 300 s ran with no suspend in it and came back ok. It
was not a sleep record, so it is not here.

## Not run

The Windows path was not run on Windows. The test suite drives it with mocks
only.
