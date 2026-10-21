# Core tests on the shared team VPS

`VPS core CI` runs the existing non-ML regression subset on the team's existing
Windows Server 2025 host. It supplements the hosted matrix; it does not replace
full unit/coverage, Python-version, macOS/MLX, GPU, training or release acceptance.

## One-time host preparation

The previous runner belongs only to SoupZero and cannot accept jobs here. An
organization administrator must make the existing host available through one
organization-scoped runner group selected for SoupZero, Soup Wall and Soup R&D.
Use a single active worker, retaining NETWORK SERVICE, `D:\CI` tools/caches and
the `soupzero-ci` label; add `soup-vps`. Keep the new group quarantined until the
service environment is verified, then enable only those repositories. Public
fork workflow execution must require maintainer approval.

The repository workflow does not register services, install Windows prerequisites,
change machine PATH/policy or copy credentials. Its main/ref/repository guards
exclude PRs, forks and arbitrary branches. Permissions are `contents: read`, and
checkout credentials are not persisted.

## Activate and run

After the host is online and this workflow is reviewed and delivered to main:

```sh
gh variable set SOUP_VPS_CI_ENABLED --repo SoupTeam/soup-rnd --body true
gh workflow run vps-ci.yml --repo SoupTeam/soup-rnd --ref main
gh run list --repo SoupTeam/soup-rnd --workflow vps-ci.yml --limit 5
```

Main pushes also start the same core lane. Without the activation variable the
job is skipped before allocation; skipped does not mean tested. Remove the
variable to stop future allocation. Existing work must be cancelled separately
if necessary.

The single organization worker serializes work across the three repositories.
The workflow's per-run concurrency key only preserves requests within this
repository; it is not a global queue or a guarantee of start order. The
30-minute job limit excludes waiting for a worker.

## Executed selection and evidence

The job verifies the fixed main event SHA and a clean checkout, reuses managed
Python 3.12 and uv 0.12.6, and creates a fresh venv outside the checkout. It
installs declared core plus pytest/ruff only, then refuses an unexpected training
stack (`torch`, `transformers`, `peft`, `trl`, `mlx`). Hub/dataset/Transformers
offline settings apply to subprocesses. It never installs `[dev]` or downloads
a model in this lane.

Full-source Ruff runs, followed by the existing config, format, registry,
requires-python-bound and fork-badge test files. The named subset overrides
pytest's global coverage options; it does not lower the normal matrix's coverage
gate. JUnit is checked for actual passing cases and failures; zero/all-skipped
execution refuses success.

Artifacts retain source/tree/runner identity, explicit core-only scope, JUnit,
passing/skipped/failed counts and logs for 14 days. The venv/cache is excluded.
Use each run's actual SHA; earlier SoupZero-hosted probe evidence does not mean
this repo-owned workflow has already run. Final activation must be demonstrated
by a run in this repository after organization-runner migration.

The existing `ci:full`, merge-freshness, matrix, smoke and release policies remain
unchanged. A successful VPS subset alone is not permission to merge a PR or
publish a release.
