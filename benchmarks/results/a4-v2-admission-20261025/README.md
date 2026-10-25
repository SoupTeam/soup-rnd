# A4 protocol v2: three machine admissions, all refused (no v2 session)

`a4_admission.py` at `efdd1a76`, protocol settings (600 s of unbuffered spill-pattern I/O on
the spill volume, 10 s windows, every window's write and read rate >= 50% of that
direction's median), run after a passing preflight on each of three rented vast.ai KVM VMs
on 2026-10-25. Each machine was refused, destroyed and confirmed gone through the API before
the next was rented; the owner's limit was three machines, so A4 v2 stopped there.

| machine | GPU (host / machine) | write median, floor (GB/s) | low write windows (GB/s) | read median, min (GB/s) | cycles | read-back mismatches |
|---|---|---|---|---|---|---|
| 1 | RTX 5090 (702330 / 152098) | 9.82, 4.91 | 51, 52, 53, 58, 59 (2.43-2.77) | 7.39, 5.28 | 4,517 | 0 |
| 2 | RTX 3090 Ti (79466 / 18065) | 2.47, 1.23 | 43, 52, 54-56, 58, 59 (0.31-1.20) | 2.53, 2.33 | 1,278 | 0 |
| 3 | RTX 4090 (59399 / 11740) | 3.52, 1.76 | 35, 37-39, 42, 48, 49, 51, 53-55, 58 (1.70-1.76) | 3.35, 3.23 | 1,663 | 0 |

Per machine: `admission.json` (the record, windows and judgement), `admission-per-second.csv`,
`preflight.json` (the A4 probe's preflight inside a 24 GiB / swap-0 scope),
`cgroup-readback.txt` (the scope shell's and its child's own cgroup limits),
`preflight-facts.txt` (volume and the three required checks on the spill, weights and
same-drive spill paths), `cuda.txt`. Box paths are replaced by `<box>` / `<work>`;
`provenance.json` lists each file's raw and published SHA-256 and the hashes of the raw
archives, which stay outside the repository. `a4_admission.judge` on the published windows
reproduces each recorded judgement.

Not shown: why the writes dipped. All three dipped only after 0.5-0.7 TiB of continuous
writes (seconds 171, 433 and 350), which fits a host-side write cache filling, but nothing
here tells a neighbour, the virtual disk or the host SSD apart.
