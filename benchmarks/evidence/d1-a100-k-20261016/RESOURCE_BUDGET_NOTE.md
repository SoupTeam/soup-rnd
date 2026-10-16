# D1 K operational budget revision, 2026-10-15

This changes resource deadlines only, within the user's existing cumulative
**$10 authorization**. It does not change the numerical source, frozen 98-case
plan, order, warm-up, timed steps, F1 sizing, clock guard, data or accepted rows.
The measured source remains `87b3bc1747ff78ff0fe16b8bdd2fa2eb0105d77c`.

The initial resource calculation reserved $1.80 for prior bills and $0.80 for
network/uncertainty. Refreshed authenticated billing showed prior D1 charges
of $1.279: pilot $0.470, E/F $0.540, G $0.257 and failed loading-only I $0.012.
Replace those reservations with $1.50 prior-cost and $0.30 network reserves.
At the fixed $0.6944444444/hour including 100 GB disk, an 11.75-hour resource
cap costs $8.159722222; conservative cumulative ceiling **$9.959722222 < $10**.
This supersedes the earlier self-imposed 10.5-hour resource cap in J/K; it is
not an increase in the user's budget. No new rental or credential is created.

The actual completed comparisons were retained. The parent controller was
paused while its currently running fresh-process ABBA child finished all
required top-up blocks normally. Only after that child exited was the parent
replaced with the same committed `d1_full_study.py`, same arguments/plan and a
later resource deadline. Completed cases are skipped by the existing resume
logic; no measured arm was killed, overwritten, filtered or silently retried.
The controlled-resume ledger and both phase commands/logs remain in the raw
archive. This is an operational resume, not a new numerical experiment.

Both guards were replaced before stopping their predecessors. The remote guard
inherits only the same rental's existing provider credential, verifies exact
identity 56108892 and addresses only its pinned GPU UUID for cleanup. The
independent local guard now stops at $8.20 estimated/observed new-rental cost.
The new remote guard identity/liveness was checked; no account key was uploaded.

The host wall clock was measured about 97 seconds behind local UTC. Remote
deadlines are shortened by 120 seconds; local UTC deadlines remain explicit.
The revised UTC measurement target is 2026-10-16 00:47 (05:47 Asia/Qyzylorda),
with 1.5 hours reserved for export. Absolute local deletion target is 02:17 UTC
(07:17 Asia/Qyzylorda). These bounds remain below the cumulative dollar cap.
Successful completion still triggers export and release immediately; the cap
does not require keeping a completed machine running.

Full hash-verified export, numerical audits and independent acceptance remain
required. A time/budget/validity stop still means incomplete evidence.
