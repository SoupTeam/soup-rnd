# Per-arm logs are in arms.tar.gz

The 48 files below the arm directories of this run (logs, `ArmWatch` records, attempt JSON) are packed in `arms.tar.gz`, so a pull request that adds this run stays readable in the GitHub diff. The files at the top of this directory are unchanged. Contents are byte-identical to what was committed before (each file compared with `cmp`).

Restore the paths that the reports in `benchmarks/` cite (run from this directory):

```bash
tar -xzf arms.tar.gz
```

List without extracting: `tar -tzf arms.tar.gz`.
