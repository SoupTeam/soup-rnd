"""Per-process CE ownership and existing F1/clock sensors; no acceptance fallback."""
from __future__ import annotations

import math
from dataclasses import asdict

from arm_validity import CHECKS, ArmWatch, Sampler
from l2l_box import confirm_gpu_owner, gpu_compute_pids


def validate_controls(data: dict, phase: str) -> None:
    """Recompute acceptance from retained evidence, including all sampled GPU PIDs."""
    try:
        owner = data['ownership']
        pids = owner['after']
        if (data['phase'] != phase or owner['method'] != 'exclusive-local-CUDA-initialization'
                or owner['before'] != [] or owner['context_active'] is not True
                or owner['single_device'] is not True or len(pids) != 1
                or type(pids[0]) is not int or pids[0] <= 0):
            raise ValueError('Exclusive CUDA ownership not established')
        samples = data['gpu_pid_samples']
        if len(samples) < 2 or any(s['problem'] or s['reading'] != pids for s in samples):
            raise ValueError('GPU PID samples missing, unknown or contaminated')
        times = [s['at_s'] for s in samples]
        if (any(not math.isfinite(t) or t < 0 for t in times)
                or times != sorted(times) or times[-1] <= times[0]):
            raise ValueError('Invalid GPU sample chronology')
        f1 = data['f1']
        if (f1['outcome'] != 'ok' or set(f1['required']) != set(CHECKS)
                or set(f1['checks']) != set(CHECKS)
                or any(f1['checks'][key]['outcome'] != 'ok' for key in CHECKS)):
            raise ValueError('All unchanged F1 checks must be ok')
        clocks = f1['gpu_clock']
        values = [clocks['sm_clock_mhz'][key] for key in ('min', 'median', 'max')]
        if (clocks['outcome'] != 'ok' or clocks['samples'] < 2
                or clocks['unknown_samples'] != 0
                or any(not math.isfinite(v) or v <= 0 for v in values)
                or values != sorted(values)):
            raise ValueError('Complete finite positive clock evidence required')
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('Missing or malformed CE controls') from error


class CaptureControls:
    """One current-process CUDA context; stop samplers before sealing evidence."""

    def __init__(self, phase: str, model_path: str) -> None:
        self.phase = phase
        self.model_path = model_path
        self.owner: dict = {}
        self.watch = None
        self.pids = None
        self.record: dict = {'phase': phase, 'status': 'NOT_STARTED'}

    def start(self) -> None:
        """Require observably idle GPU before creating our first CUDA allocation."""
        before = gpu_compute_pids()
        if before != []:
            raise ValueError('GPU not observably idle before CE CUDA initialization')
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('One visible CUDA GPU required')
        probe = torch.empty(1, device='cuda')
        torch.cuda.synchronize()
        self.owner = confirm_gpu_owner(before, context_active=True, single_device=True)
        del probe
        if self.owner['method'] != 'exclusive-local-CUDA-initialization':
            raise ValueError('Unresolved host PID mapping')
        self.watch = ArmWatch(arm=self.phase, round=0, run=0, model_path=self.model_path)
        self.watch.start()
        self.pids = Sampler('ce_gpu_owner', gpu_compute_pids, 2.0)
        self.pids.start()
        self.record.update(status='RUNNING', ownership=self.owner)

    def stop(self) -> dict:
        """Keep failed sensor evidence and raise on any unknown or contaminated check."""
        self.record.update(status='FAILED', ownership=self.owner)
        try:
            if self.pids is not None:
                self.pids.stop()
                self.record['gpu_pid_samples'] = [asdict(s) for s in self.pids.samples]
        finally:
            if self.watch is not None:
                self.record['f1'] = self.watch.stop()
        validate_controls(self.record, self.phase)
        self.record['status'] = 'PASS_CE_CONTROLS'
        return self.record
