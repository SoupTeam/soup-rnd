# Moonlight unattended evaluation — how to check and collect results

Everything below runs in **PowerShell** or **cmd** on this PC. Keep the PC on and awake.

## Where is it now? (one line)

```
python -X utf8 C:\Users\sydyk\Downloads\moonlight-autorun\moonlight_watchdog.py progress
```

It prints one line: the state, the step, which scoring piece (of 84) is running, whether the
run is alive, the money, and whether the machine was **deleted** or merely **not answering**.
A line saying the machine "не отвечает по SSH (не удалена)" is a connection problem, not a
deletion; the watchdog never deletes a machine that is merely unreachable before 21:45 UTC.

For the full detail:

```
python -X utf8 C:\Users\sydyk\Downloads\moonlight-autorun\moonlight_watchdog.py status
```

Read `remote.state` and `remote.step`:

| state | step | meaning |
|---|---|---|
| RUNNING | setup / inputs / probe / plan / download / restore-identity / convert / identity-check | preparing the host (about 2–3 h) |
| RUNNING | evaluate | scoring; `shard_index` of `shard_total` shows progress (84 pieces, 2 already done on the first host) |
| RUNNING | pair / analysis | final comparison (about 1 h) |
| DONE | packaged | finished; `detail` shows the gated and statistical verdicts |
| FAILED / STOPPED | packaged | ended early; `detail` says why |

`cost.estimate_usd` is the running total for the whole study (cap $30).
`closed` becomes non-empty once the rented machine and disk are deleted.

The watchdog log, one line every 15 minutes:

```
type C:\Users\sydyk\Downloads\moonlight-autorun\watchdog.log
```

## Getting the results

Nothing to do: when the run ends, the watchdog downloads the archive, checks its SHA256
and then deletes the rented machine and disk. The results land in:

```
C:\Users\sydyk\Downloads\moonlight-quality-results-h2\
```

- `moonlight-resident-20261111t201548z-whole3-h2-final-done.tar.gz` — all evidence
  (`evidence/quality-summary.json` holds the verdicts, `evidence/paired/` the paired rows)
- `remote-status.json` — final state, verdict line and archive SHA256

If the status shows the run ended but nothing was downloaded (for example the PC slept):

```
python C:\Users\sydyk\Downloads\moonlight-autorun\moonlight_watchdog.py download
python C:\Users\sydyk\Downloads\moonlight-autorun\moonlight_watchdog.py close
```

`close` deletes only the instance and volume recorded in `state.json`, and only after both are
verified gone. It refuses while the run is still RUNNING, while the host is unreachable, or
while a finished archive could not be downloaded; `close --force` overrides that. It is safe
to run twice. Run `status` afterwards: `closed.verified_absent` must be `true`.

## Automatic limits

- Cost estimate ≥ $28.5: the watchdog asks the run to stop, waits for its archive,
  downloads it and deletes the machine. ≥ $29.5 or after 2026-11-13 21:45 UTC it makes one
  last download or rescue attempt, then deletes.
- 2026-11-13 20:30 UTC: no new scoring piece starts (pairing and analysis still run);
  21:00 UTC: stop requested.
- A failed scoring piece is never retried; the run stops, packages its evidence and the
  machine is deleted after the download.
- If the download keeps failing, the machine is kept (it only bills about $0.6 per hour)
  until a download works or the hard close.
- If the driver dies it is relaunched twice; after that the watchdog saves what exists and
  deletes the machine.

## If something looks wrong

- `status` shows `remote: null`: the host is unreachable (PC offline, or the provider moved
  the machine). The watchdog keeps the machine and retries every 15 minutes.
- Billing runs only while the machine exists. To stop it immediately and keep nothing:
  `python ...\moonlight_watchdog.py close --force`.
- Quality verdicts are in `evidence/quality-summary.json` inside the archive
  (`gated_quality_decision`, `statistical_quality_decision`).
