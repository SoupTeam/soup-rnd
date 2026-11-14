"""Review all four completed pilot arms; projections never authorize full training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from policy_pilot_support import bound, contained, load_contract
from run import write
from summarize import summarize


def review(spec: dict, roots: list[Path], science_sha: str, out: Path) -> dict:
    configurations = set()
    for root in roots:
        meta = json.loads((root / 'prospective_run.json').read_text())
        configurations.add(meta['launch_config_sha256'])
        if meta.get('new_series_not_historical_redeployment') is not True:
            raise ValueError('Historical pilot cannot substitute for a fresh matched arm')
        external = json.loads(Path(str(root) + '.launcher.json').read_text())
        if (external['exitcode'] != 0 or external['hard_timeout'] is not False
                or external['contract_sha256'] != science_sha
                or external['bound_s'] != bound(spec, 'pilot')):
            raise ValueError('External completion/deadline receipt missing or failed')
    if len(configurations) != 1:
        raise ValueError('Private execution configuration changed between arms')
    comparison = summarize(spec, roots, 'pilot', science_sha, out)
    if not comparison['F1_all_accepted']:
        raise ValueError('One or more pilot timings are invalid')
    plans = json.loads(contained(Path(__file__).parents[2], spec['input_plan']['path']).read_text())
    for arm, row in comparison['rows'].items():
        full = plans['arms'][arm]['full']
        pilot = plans['arms'][arm]['pilot']
        token_ratio = full['tokens']['useful_loss_tokens'] / pilot['tokens']['useful_loss_tokens']
        slots_ratio = (full['tokens']['processed_input_slots']
                       / pilot['tokens']['processed_input_slots'])
        attention_ratio = (full['attention_length_squared_proxy']
                           / pilot['attention_length_squared_proxy'])
        row['train_only_heuristic_projection_s'] = {
            'useful_token_linear': row['actual_train_s'] * token_ratio,
            'processed_slot_linear': row['actual_train_s'] * slots_ratio,
            'attention_proxy_linear': row['actual_train_s'] * attention_ratio,
        }
        row['projection_is_not_a_bound_or_end_to_end_forecast'] = True
    comparison.update(
        status='FOUR_PILOT_RESOURCE_REVIEW_PENDING', quality_verdict='NO_VERDICT',
        quality_ranking_authorized=False, full_training_authorized=False,
        pilot_prefixes_have_different_examples=True,
        full_run_needs_new_committed_bounds_and_fresh_private_budget_admission=True,
        projection_limitations='Prefix lengths/composition differ; linear projections exclude '
                               'setup, eval, reload, traffic, host changes and nonlinear runtime.',
    )
    return comparison


def main() -> None:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha', required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs=4, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    spec = load_contract(args.spec, args.spec_sha, args.artifact_root)
    args.out.mkdir(parents=True, exist_ok=False)
    result = review(spec, args.runs, args.spec_sha, args.out)
    write(args.out / 'comparison.json', result)
    Console().print('FOUR_PILOT_RESOURCE_REVIEW_PENDING; no full-run authorization')


if __name__ == '__main__':
    main()
