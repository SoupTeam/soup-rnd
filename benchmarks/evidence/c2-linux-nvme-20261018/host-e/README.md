# Host E evidence guide

See [`../../../c2-results-20261018.md`](../../../c2-results-20261018.md) for the
decision and scope. All inputs are synthetic; hardware I/O and GPU steps are
real. Production depth remains 1.

- `c2-diag-primitive-e1-campaign.json`: completed five-block canonical F1
  diagnostic, including every arm's sensor record and the parent decision.
  `c2-diag-step-e3-campaign.json` retains the final separate step sequence.
  All child JSON/log files remain byte-for-byte in the archive below.
- `c2-evidence-final.tar.gz`: exact remote journals, environments, source/cache
  metadata, all step sequences (void ones included), test logs and exit files.
  Paths inside the archive retain `workspace/` or `root/`; `manifest.json`
  records original absolute names, sizes and SHA-256 hashes.
- `c2-evidence-final.sha256`: archive digest obtained from the host and checked
  after download. The local `manifest.json` separately hashes this directory's
  artifacts; it excludes itself and is regenerated after final collection.
- `timed-arms.csv`: derived primitive values/counters and, only for an accepted
  complete step series, its three raw times/losses/counters per arm. Primitive
  `pipeline_s` is the submission/completion timer; its exact outer counter
  bracket duration was not stored and `counter_interval_s` is left empty.
  Step `counter_interval_s` is the enclosing sampled physical-counter window.
- `frozen-*.md`: exact Git-blob protocol snapshots for the named source revisions.
  Diagnostic write bounds are separate from strict v1; they cannot authorize
  shipping even if a descriptive effect passes.
- `commands.sh`, `sensor-*-command.txt`, `prequiet-e3-command.txt` and
  `collection-command.txt`: operator commands/scripts, preserved for reproduction.
- `instance-before-cleanup.json`, `cleanup.json`: whitelisted provider records;
  no API key, private SSH key or full provider response is stored here.

Excluded step sequences are preserved **whole**. Step-e1 failed before timing
on its own prefetch during an incorrectly placed idle check. Step-e2 stopped
after child 12 on F1's foreign-read gate; all preceding runs in that sequence
are excluded too. No times or pilot estimates from them enter another series.
