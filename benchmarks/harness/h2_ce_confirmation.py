"""Four fresh CE correctness processes and tensor review; never a speed benchmark.

Launch only within the separately verified exclusive GPU/F1/clocks envelope.
This runner does not rent hardware, restore a stopped instance or authorize spending.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

from h2_ce_training import Recorder, compare_tensors, recording_trainer, review_training, write_json
from h2_gpu_correctness import (
    SOURCES,
    preflight,
    sha_file,
    validate_authority,
    validate_live_identity,
    validate_loaded,
)
from h2_numerics import collect_probe, compare_probes, read_probe
from h2_operator_capture import collect
from h2_operator_patch import kernel_audit, validate_counts
from h2_profile_window import ProfileWindow
from h2_update_oracle import snapshot
from h2_watchdog import bounded_process
from rich.console import Console

REPO = Path(__file__).parents[2]
SPEC = REPO / 'benchmarks/h2'
HARNESS = REPO / 'benchmarks/harness'
PHASES = ('baseline_probes', 'ce_only_probes', 'baseline_training', 'ce_only_training')
ARTIFACTS = tuple('benchmarks/harness/' + n for n in sorted(set(SOURCES) | {
    'h2_ce_confirmation.py', 'h2_ce_training.py', 'h2_ce_workset.py',
    'h2_operator_patch.py', 'h2_operator_capture.py',
})) + tuple('benchmarks/h2/' + n for n in (
    'operator_study.json', 'numerical_policy.json', 'candidate_configs/baseline.json',
    'runtime_candidate.lock', 'synthetic_fixture.jsonl', 'prepared_fixture_candidate.jsonl',
    'CE_CONFIRMATION_PROTOCOL.txt', 'ce_confirmation_plan.json', 'ce_confirmation_workset.json',
))
_USED = False


def validate_contract(contract: dict) -> dict:
    validate_authority(contract, 'baseline')
    study = json.loads((SPEC / 'operator_study.json').read_text())
    plan = json.loads((SPEC / 'ce_confirmation_plan.json').read_text())
    if (contract.get('schema') != 'h2-ce-authority-v1' or contract['max_seconds'] > 600
            or contract['budget_cap_usd'] > 5):
        raise ValueError('CE-specific bounded authority required')
    start = contract.get('series_started_unix_s')
    if (type(start) not in (int, float) or not math.isfinite(start) or start > time.time()
            or not 0 < contract['deadline_unix_s'] - start <= 2400):
        raise ValueError('Single shared 2400s absolute deadline required')
    worst = (contract['prior_spend_usd'] + contract['hourly_rate_usd'] * 2400 / 3600
             + contract['cleanup_traffic_reserve_usd'])
    if worst > contract['budget_cap_usd']:
        raise ValueError('Whole CE series exceeds cumulative H2 budget')
    for key in ('model_revision', 'model_files_sha256', 'versions', 'numerical_policy_sha256'):
        if contract.get(key) != study[key]:
            raise ValueError('Frozen study pin differs: ' + key)
    pins = contract.get('artifacts_sha256', {})
    if set(pins) != set(ARTIFACTS) or any(sha_file(REPO / n) != pins[n] for n in ARTIFACTS):
        raise ValueError('Missing or changed CE source artifact')
    for name, digest in plan['files_sha256'].items():
        if sha_file(SPEC / name) != digest:
            raise ValueError('Prospective plan artifact differs: ' + name)
    if set(contract.get('runner_sha256', {})) != set(SOURCES):
        raise ValueError('Original runner pins required')
    return study


def committed_preflight(contract: dict) -> dict:
    study = validate_contract(contract)
    preflight(contract, 'baseline')
    for name in ARTIFACTS:
        raw = subprocess.run(['git', 'show', contract['source_commit'] + ':' + name],
                             cwd=REPO, check=True, capture_output=True, timeout=10).stdout
        if hashlib.sha256(raw).hexdigest() != contract['artifacts_sha256'][name]:
            raise ValueError('Unpublished source artifact: ' + name)
    return study


def tree_seal(root: Path) -> None:
    write_json(root / 'files_sha256.json', {
        p.relative_to(root).as_posix(): sha_file(p) for p in root.rglob('*') if p.is_file()})


def read_child(root: Path, phase: str) -> dict:
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Symlink evidence rejected')
    pins = json.loads((root / 'files_sha256.json').read_text())
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(pins) | {'files_sha256.json'}:
        raise ValueError('Child file inventory mismatch')
    for name, digest in pins.items():
        path = os.path.realpath(root / name)
        if (os.path.commonpath([os.path.realpath(root), path]) != os.path.realpath(root)
                or sha_file(Path(path)) != digest):
            raise ValueError('Child SHA/containment mismatch')
    data = json.loads((root / 'evidence.json').read_text())
    arm = phase.removesuffix('_probes').removesuffix('_training')
    if data['status'] != 'COMPLETE_CE_CAPTURE' or data['phase'] != phase:
        raise ValueError('Incomplete/wrong child capture')
    validate_counts(arm, data['patch_after']['patch_calls'])
    if (data['patch_after']['status'] != 'PASS_EXPLICIT_PATCH_BINDINGS'
            or data['patch_after']['arm'] != arm):
        raise ValueError('Invalid patch binding evidence')
    trace = root / ('profile/kernel_trace.json' if phase.endswith('_training')
                    else 'probe_00/kernel_trace.json')
    kernel_audit(json.loads(trace.read_text())['traceEvents'], arm)
    return data


def review_pair(root: Path, *, training: bool) -> dict:
    work = json.loads((SPEC / 'ce_confirmation_workset.json').read_text())
    policy = json.loads((SPEC / 'numerical_policy.json').read_text())
    suffix = '_training' if training else '_probes'
    paths = [root / (arm + suffix) / 'runner' for arm in ('baseline', 'ce_only')]
    evidence = [read_child(path, arm + suffix)
                for path, arm in zip(paths, ('baseline', 'ce_only'))]
    contract = json.loads((root / 'contract.json').read_text())
    if set(contract['artifacts_sha256']) != set(ARTIFACTS) or any(
            sha_file(REPO / name) != contract['artifacts_sha256'][name] for name in ARTIFACTS):
        raise ValueError('Offline reviewer/source differs from captured authority')
    for path, data in zip(paths, evidence):
        if json.loads((path / 'contract.json').read_text()) != contract:
            raise ValueError('Child contract differs from whole-series authority')
        if any(data[key] != contract[key] for key in ('source_commit', 'gpu_uuid', 'boot_id')):
            raise ValueError('Child identity differs from authority')
    expected_initial = json.loads((SPEC / 'operator_study.json').read_text())[
        'reference_identity']['before_sha256']
    for key in ('source_commit', 'gpu_uuid', 'boot_id', 'initial_identity'):
        if evidence[0][key] != evidence[1][key]:
            raise ValueError('Pair identity differs: ' + key)
    if evidence[0]['initial_identity'] != expected_initial:
        raise ValueError('Pair differs from historical initial state')
    if training:
        probe_initial = json.loads((root / 'baseline_probes/runner/evidence.json').read_text())
        if probe_initial['initial_identity'] != evidence[0]['initial_identity']:
            raise ValueError('Training initial state differs from accepted probe model')
        for path in paths:
            data = json.loads((path / 'training/training.json').read_text())
            if data['compute_policy'] != 'bf16-autocast-cuda':
                raise ValueError('GPU training review requires BF16 CUDA evidence')
            if hashlib.sha256(json.dumps(data['before'], sort_keys=True,
                                        separators=(',', ':')).encode()).hexdigest() != (
                                            expected_initial):
                raise ValueError('Training tensors start from a different state')
        return review_training(*(p / 'training' for p in paths), policy, work['training_batches'])
    results = []
    for index, expected in enumerate(work['probes']):
        captures = [p / f'probe_{index:02d}' / 'model_probe' for p in paths]
        tensors = []
        for path in captures:
            data, values = read_probe(path)
            if data['compute_policy'] != 'bf16-autocast-cuda':
                raise ValueError('GPU probe review requires BF16 CUDA evidence')
            if hashlib.sha256(json.dumps(data['before'], sort_keys=True,
                                        separators=(',', ':')).encode()).hexdigest() != (
                                            expected_initial):
                raise ValueError('Probe tensors start from a different state')
            if {k.removeprefix('input.'): v.tolist() for k, v in values.items()
                    if k.startswith('input.')} != expected:
                raise ValueError('Probe actual input differs from frozen workset')
            tensors.append({k: v for k, v in values.items() if k.startswith('gradient.')})
        result = compare_probes(*captures, policy)
        if result['status'] != 'PASS_MODEL_PROBE_TOLERANCES':
            raise ValueError('Probe comparison failed: ' + str(result['reasons']))
        compare_tensors(*tensors, policy)
        results.append(result)
    return {'status': 'PASS_CE_PROBES', 'probes': results, 'performance_verdict': 'NO_VERDICT'}


def run_child(contract: dict, phase: str, out: Path) -> dict:
    global _USED
    if _USED or phase not in PHASES:
        raise ValueError('Fresh process and declared phase required')
    _USED = True
    expected = Path(contract['evidence_root']) / phase / 'runner'
    if os.path.realpath(out) != os.path.realpath(expected):
        raise ValueError('Unauthorized child evidence path')
    out.mkdir(parents=True, exist_ok=False)
    result = {'status': 'FAILED', 'phase': phase, 'performance_verdict': 'NO_VERDICT'}
    try:
        if platform.system() != 'Linux':
            raise ValueError('Linux CUDA required')

        def expired(*unused) -> None:
            raise TimeoutError('CE child deadline reached')
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(contract['max_seconds'])
        study = committed_preflight(contract)
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          TOKENIZERS_PARALLELISM='false')
        import torch
        import trl
        from datasets import Dataset
        from h2_operator_patch import PatchController

        import soup_cli
        from soup_cli.config.schema import SoupConfig
        from soup_cli.data.loader import load_dataset
        from soup_cli.trainer.sft import SFTTrainerWrapper

        source = os.path.realpath(REPO / 'src/soup_cli')
        if os.path.commonpath([source, os.path.realpath(soup_cli.__file__)]) != source:
            raise ValueError('Imported Soup is outside pinned source')
        if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
                or not torch.cuda.is_bf16_supported()
                or torch.cuda.get_device_capability()[0] < 8
                or torch.cuda.get_device_properties(0).total_memory < 16 * 1024**3):
            raise ValueError('One Ampere+ BF16 CUDA GPU with >=16GiB required')
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        validate_live_identity(contract, boot,
                               str(torch.cuda.get_device_properties(0).uuid))
        training = phase.endswith('_training')
        arm = phase.removesuffix('_training').removesuffix('_probes')
        control = PatchController(arm)
        holder: dict = {}
        window = None
        if training:
            (out / 'profile').mkdir()
            window = ProfileWindow(out / 'profile', cuda=True, microbatch=5)
            trl.SFTTrainer = recording_trainer(trl.SFTTrainer, holder, window)
        recipe = json.loads((SPEC / 'candidate_configs/baseline.json').read_text())
        recipe['base'] = contract['model_path']
        recipe['data']['train'] = str(SPEC / 'synthetic_fixture.jsonl')
        recipe['output'] = str(out / 'trained_output')
        recipe['training']['use_liger'] = arm == 'ce_only'
        config = SoupConfig.model_validate(recipe)
        wrapper = SFTTrainerWrapper(config, device='cuda', report_to='none')
        wrapper.setup(load_dataset(config.data))
        loaded = validate_loaded(wrapper, 'operator_diagnostic')
        result['loaded'] = loaded
        if (wrapper.trainer.optimizer is not None or bool(wrapper.trainer.args.use_liger_kernel)
                != (arm == 'ce_only') or loaded['attention_implementation'] != 'sdpa'):
            raise ValueError('Unexpected Trainer/attention state')
        prepared = [json.loads(line) for line in
                    (SPEC / 'prepared_fixture_candidate.jsonl').read_text().splitlines()]
        actual = [{k: row[k] for k in ('input_ids', 'attention_mask', 'labels')}
                  for row in wrapper.trainer.train_dataset]
        if actual != prepared:
            raise ValueError('Actual prepared fixture differs from frozen rows')
        names = list(loaded['trainable_shapes'])
        before = snapshot(wrapper.model, names)
        identity = hashlib.sha256(json.dumps(before, sort_keys=True,
                                            separators=(',', ':')).encode()).hexdigest()
        if identity != study['reference_identity']['before_sha256']:
            raise ValueError('Initial model differs from historical pinned state')
        result['initial_identity'] = identity
        work = json.loads((SPEC / 'ce_confirmation_workset.json').read_text())
        if training:
            rows = [{k: v[0] for k, v in batch.items()} for batch in work['training_batches']]
            wrapper.trainer.train_dataset = Dataset.from_list(rows)

            def collator(items: list[dict]) -> dict:
                if len(items) != 1:
                    raise ValueError('Fixed single-row microbatches required')
                return {k: torch.tensor([items[0][k]], dtype=torch.long) for k in items[0]}
            wrapper.trainer.data_collator = collator
            holder['recorder'] = Recorder(out / 'training', wrapper.model, names,
                                          work['training_batches'], accumulation=4,
                                          compute_policy='bf16-autocast-cuda')
            wrapper.train()  # Trainer alone invokes the instance patch in this process.
            holder['recorder'].finish(holder['optimizer'],
                                      global_step=wrapper.trainer.state.global_step)
            window.require_complete()
            trace = out / 'profile/kernel_trace.json'
        else:
            result['patch_lifecycle'] = control.prepare_capture(wrapper.trainer, wrapper.model)
            for index, batch in enumerate(work['probes']):
                tensors = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
                path = out / f'probe_{index:02d}'
                if index == 0:
                    collect(wrapper.model, tensors, names, path, cuda=True,
                            expected_identity=study['reference_identity'])
                else:
                    collect_probe(wrapper.model, tensors, names, path / 'model_probe', bf16=True)
            trace = out / 'probe_00/kernel_trace.json'
        result['patch_after'] = control.verify(wrapper.model)
        result['kernels'] = kernel_audit(json.loads(trace.read_text())['traceEvents'], arm)
        result.update(status='COMPLETE_CE_CAPTURE', source_commit=contract['source_commit'],
                      gpu_uuid=contract['gpu_uuid'], boot_id=contract['boot_id'])
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        write_json(out / 'evidence.json', result)
        write_json(out / 'contract.json', contract)
        tree_seal(out)
    return result


def run_series(contract: dict, out: Path) -> dict:
    validate_contract(contract)
    if os.path.realpath(out) != contract.get('evidence_root'):
        raise ValueError('Exact new evidence root required')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'contract.json', contract)
    result = {'status': 'BLOCKED', 'phases': {}, 'performance_verdict': 'NO_VERDICT',
              'external_controls_required': True, 'provider_stop_performed': False}
    try:
        for index, phase in enumerate(PHASES):
            remaining = contract['deadline_unix_s'] - time.time() - 40
            if remaining <= 2:
                raise TimeoutError('Shared deadline cleanup reserve reached')
            report = bounded_process(
                [sys.executable, str(Path(__file__)), '--contract', str(out / 'contract.json'),
                 '--child', '--phase', phase, '--out', str(out / phase / 'runner')],
                out / phase / 'supervisor', seconds=min(contract['max_seconds'], remaining))
            result['phases'][phase] = report
            if (report['status'] != 'CHILD_EXITED' or not report['group_cleanup_confirmed']
                    or report.get('child_status', {}).get('returncode') != 0):
                raise ValueError('Child failed; no subsequent phases or retries')
            evidence = json.loads((out / phase / 'runner/evidence.json').read_text())
            if evidence['status'] != 'COMPLETE_CE_CAPTURE':
                raise ValueError('Incomplete child evidence')
            if index in (1, 3):
                review = review_pair(out, training=index == 3)
                write_json(out / ('training_review.json' if index == 3 else 'probe_review.json'),
                           review)
        result['status'] = 'COMPLETE_CE_NUMERICAL_SERIES'
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        write_json(out / 'series.json', result)
        tree_seal(out)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--review-only', action='store_true')
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--phase', choices=PHASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.review_only:
        if args.child or args.phase or args.contract:
            parser.error('Offline review accepts only --out')
        review_pair(args.out, training=False)
        review_pair(args.out, training=True)
        Console().print('PASS_CE_NUMERICAL_REVIEW; external controls pending; NO_VERDICT')
        return
    if args.contract is None or args.child != (args.phase is not None):
        parser.error('Require contract and a valid supervised phase')
    contract = json.loads(args.contract.read_text())
    result = run_child(contract, args.phase, args.out) if args.child else run_series(contract,
                                                                                   args.out)
    Console().print(result['status'] + '; performance=NO_VERDICT')
    if result['status'] not in {'COMPLETE_CE_CAPTURE', 'COMPLETE_CE_NUMERICAL_SERIES'}:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
