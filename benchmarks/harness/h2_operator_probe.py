"""Seven fresh-process, one-backward H2 diagnostics. No optimizer or speedup verdict."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

from h2_gpu_correctness import (
    SOURCES,
    preflight,
    sha_file,
    validate_authority,
    validate_live_identity,
    validate_loaded,
)
from h2_operator_patch import ARMS, patch_flags
from h2_watchdog import bounded_process
from rich.console import Console

REPO = Path(__file__).parents[2]
ARTIFACTS = (
    'benchmarks/harness/h2_operator_probe.py', 'benchmarks/harness/h2_operator_patch.py',
    'benchmarks/harness/h2_operator_capture.py', 'benchmarks/h2/operator_study.json',
    'benchmarks/h2/OPERATOR_DIAGNOSTIC_PROTOCOL.txt',
    'benchmarks/h2/OPERATOR_LIFECYCLE_RECOVERY_PROTOCOL.txt',
    'benchmarks/h2/candidate_configs/baseline.json', 'benchmarks/h2/synthetic_fixture.jsonl',
    'benchmarks/h2/runtime_candidate.lock',
)
_USED = False


def validate_contract(contract: dict) -> dict:
    # Reuse existing license, ownership, deadline and cost checks without modifying them.
    validate_authority(contract, 'baseline')
    study = json.loads((REPO / 'benchmarks/h2/operator_study.json').read_text())
    if contract.get('schema') != 'h2-operator-authority-v1':
        raise ValueError('Separate operator diagnostic authority required')
    if (contract.get('arms') != list(ARMS) or study['arm_order'] != list(ARMS)
            or contract['max_seconds'] > study['max_seconds_per_arm']
            or contract['budget_cap_usd'] > 5):
        raise ValueError('Unexpected arms, duration or H2 budget cap')
    for key in ('model_revision', 'versions', 'model_files_sha256',
                'numerical_policy_sha256', 'recipe_sha256', 'fixture_sha256'):
        if contract.get(key) != study[key]:
            raise ValueError('Frozen study identity differs: ' + key)
    started = contract.get('series_started_unix_s')
    if (type(started) not in (int, float) or not 0 <= time.time() - started <= 2140
            or not 0 < contract['deadline_unix_s'] - started <= 2140):
        raise ValueError('Whole-series absolute bound required')
    total_seconds = len(ARMS) * contract['max_seconds'] + 40
    total = (contract['prior_spend_usd'] + contract['hourly_rate_usd'] * total_seconds / 3600
             + contract['cleanup_traffic_reserve_usd'])
    if total > contract['budget_cap_usd']:
        raise ValueError('Whole diagnostic series exceeds H2 budget')
    pins = contract.get('source_files_sha256', {})
    if set(pins) != set(ARTIFACTS):
        raise ValueError('Incomplete source artifact manifest')
    for name in ARTIFACTS:
        if pins[name] != sha_file(REPO / name):
            raise ValueError('Source artifact SHA differs: ' + name)
    for name in SOURCES:
        if contract.get('runner_sha256', {}).get(name) != sha_file(
                REPO / 'benchmarks/harness' / name):
            raise ValueError('Inherited runner source differs: ' + name)
    if sha_file(REPO / 'benchmarks/h2/numerical_policy.json') != (
            study['numerical_policy_sha256']):
        raise ValueError('Frozen numerical policy differs')
    return study


def committed_preflight(contract: dict) -> dict:
    study = validate_contract(contract)
    preflight(contract, 'baseline')
    for name in ARTIFACTS:
        committed = subprocess.run(
            ['git', 'show', f"{contract['source_commit']}:{name}"], cwd=REPO,
            check=True, capture_output=True, timeout=10,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != contract['source_files_sha256'][name]:
            raise ValueError('Diagnostic artifact is not committed: ' + name)
    return study


def run_child(contract: dict, arm: str, out: Path) -> dict:
    global _USED
    if _USED:
        raise ValueError('One arm per fresh process; no retry')
    _USED = True
    patch_flags(arm)
    out.mkdir(parents=True, exist_ok=False)
    result = {'status': 'FAILED', 'arm': arm, 'performance_verdict': 'NO_VERDICT'}
    try:
        study = committed_preflight(contract)
        expected_output = Path(contract['evidence_root']) / arm / 'runner'
        if os.path.realpath(out) != os.path.realpath(expected_output):
            raise ValueError('Child output differs from authorized evidence root')
        if platform.system() != 'Linux' or not hasattr(signal, 'SIGALRM'):
            raise ValueError('Linux CUDA required')

        def expired(*unused) -> None:
            raise TimeoutError('Probe deadline reached')

        signal.signal(signal.SIGALRM, expired)
        signal.alarm(contract['max_seconds'])
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          TOKENIZERS_PARALLELISM='false')
        import torch
        from h2_operator_capture import collect
        from h2_operator_patch import PatchController, kernel_audit

        import soup_cli
        from soup_cli.config.schema import SoupConfig
        from soup_cli.data.loader import load_dataset
        from soup_cli.trainer.sft import SFTTrainerWrapper

        source_root = os.path.realpath(REPO / 'src/soup_cli')
        imported_source = os.path.realpath(soup_cli.__file__)
        if os.path.commonpath([source_root, imported_source]) != source_root:
            raise ValueError('Imported Soup does not belong to committed checkout')
        if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
                or not torch.cuda.is_bf16_supported()
                or torch.cuda.get_device_capability()[0] < 8
                or torch.cuda.get_device_properties(0).total_memory < 16 * 1024**3):
            raise ValueError('One Ampere+ bf16 CUDA GPU with >=16GiB required')
        boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        validate_live_identity(contract, boot_id, str(torch.cuda.get_device_properties(0).uuid))
        control = PatchController(arm)
        recipe = json.loads((REPO / 'benchmarks/h2/candidate_configs/baseline.json').read_text())
        recipe['base'] = contract['model_path']
        recipe['data']['train'] = str(REPO / 'benchmarks/h2/synthetic_fixture.jsonl')
        recipe['output'] = str(out / 'unused_training_output')
        recipe['training']['use_liger'] = arm != 'baseline'
        config = SoupConfig.model_validate(recipe)
        data = load_dataset(config.data)
        if len(data['train']) != 128:
            raise ValueError('Fixture row count differs')
        wrapper = SFTTrainerWrapper(config, device='cuda', report_to='none')
        wrapper.setup(data)
        result['patch_lifecycle'] = control.prepare_capture(wrapper.trainer, wrapper.model)
        # CE-only / RoPE-only deliberately need not replace any module classes.
        result['loaded'] = validate_loaded(wrapper, 'operator_diagnostic')
        result['patch_before'] = control.verify(wrapper.model)
        if (bool(wrapper.trainer.args.use_liger_kernel) != (arm != 'baseline')
                or wrapper.trainer.optimizer is not None):
            raise ValueError('Unexpected Trainer treatment/optimizer state')
        batch = wrapper.trainer.data_collator([wrapper.trainer.train_dataset[0]])
        names = list(result['loaded']['trainable_shapes'])
        result['capture'] = collect(wrapper.model, batch, names, out / 'capture', cuda=True,
                                    expected_identity=study['reference_identity'])
        result['patch_after'] = control.verify(wrapper.model)
        trace = json.loads((out / 'capture/kernel_trace.json').read_text())
        result['kernels'] = kernel_audit(trace['traceEvents'], arm)
        result.update(status='COMPLETE_OPERATOR_CAPTURE', source_commit=contract['source_commit'],
                      gpu_uuid=contract['gpu_uuid'], boot_id=contract['boot_id'])
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        (out / 'evidence.json').write_text(json.dumps(result, indent=2) + '\n')
        (out / 'contract.json').write_text(json.dumps(contract, indent=2) + '\n')
        pins = {p.relative_to(out).as_posix(): sha_file(p) for p in out.rglob('*') if p.is_file()}
        (out / 'files_sha256.json').write_text(json.dumps(pins, indent=2) + '\n')
    return result


def run_series(contract: dict, out: Path) -> dict:
    study = validate_contract(contract)
    if os.path.realpath(out) != contract.get('evidence_root'):
        raise ValueError('Exact new evidence root must be authorized')
    out.mkdir(parents=True, exist_ok=False)
    result = {'status': 'BLOCKED', 'arms': {}, 'performance_verdict': 'NO_VERDICT',
              'provider_stop_performed': False}
    try:
        saved_contract = out / 'contract.json'
        saved_contract.write_text(json.dumps(contract, indent=2) + '\n')
        # CPU-only readback before any GPU child; exact historical reference pinned in study.
        from h2_numerics import compare_probes, read_probe

        historical = Path(contract['historical_reference_probe'])
        if sha_file(historical / 'files_sha256.json') != (
                study['historical_reference_manifest_sha256']):
            raise ValueError('Historical reference pin differs')
        read_probe(historical)
        policy = json.loads((REPO / 'benchmarks/h2/numerical_policy.json').read_text())
        for arm in ARMS:
            seconds = min(contract['max_seconds'], contract['deadline_unix_s'] - time.time() - 2)
            if seconds <= 0:
                raise TimeoutError('No time remaining; no next child')
            argv = [sys.executable, str(Path(__file__).absolute()), '--contract',
                    str(saved_contract.absolute()), '--arm', arm, '--child',
                    '--out', str((out / arm / 'runner').absolute())]
            watchdog = bounded_process(argv, out / arm, seconds=seconds)
            if (watchdog['status'] != 'CHILD_EXITED' or not watchdog['group_cleanup_confirmed']
                    or watchdog['child_status']['returncode'] != 0):
                raise RuntimeError('Failed supervised capture: ' + arm)
            capture = out / arm / 'runner/capture/model_probe'
            reference = historical if arm == 'baseline' else (
                out / 'baseline/runner/capture/model_probe')
            pair = compare_probes(reference, capture, policy)
            (out / arm / 'comparison.json').write_text(json.dumps(pair, indent=2) + '\n')
            result['arms'][arm] = pair['status']
            if arm == 'baseline' and pair['status'] != 'PASS_MODEL_PROBE_TOLERANCES':
                raise ValueError('Fresh baseline failed historical continuity; stop series')
            # A valid treatment mismatch is a diagnostic observation, not an infrastructure retry.
        result['status'] = 'COMPLETE_DIAGNOSTIC_MATRIX'
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        (out / 'series.json').write_text(json.dumps(result, indent=2) + '\n')
        (out / 'files_sha256.json').write_text(json.dumps(
            {p.relative_to(out).as_posix(): sha_file(p) for p in out.rglob('*') if p.is_file()},
            indent=2) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--arm', choices=list(ARMS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    if args.child:
        result = run_child(contract, args.arm, args.out)
        ok = result['status'] == 'COMPLETE_OPERATOR_CAPTURE'
    else:
        if args.arm is not None:
            parser.error('--arm is reserved for a supervised child')
        result = run_series(contract, args.out)
        ok = result['status'] == 'COMPLETE_DIAGNOSTIC_MATRIX'
    Console().print(result['status'] + '; performance=NO_VERDICT')
    if not ok:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
