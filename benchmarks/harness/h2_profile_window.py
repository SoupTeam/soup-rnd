"""Single-microbatch attribution window, never a performance measurement."""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class ProfileWindow:
    """Keep one forward/backward trace, not a full-training event history."""

    def __init__(self, out: Path, *, cuda: bool, microbatch: int = 5) -> None:
        if type(microbatch) is not int or microbatch < 1:
            raise ValueError('Positive one-based microbatch required')
        self.out = out
        self.cuda = cuda
        self.microbatch = microbatch
        self.calls = 0
        self.evidence: dict[str, Any] | None = None
        self.operations: list[dict[str, Any]] = []

    @contextmanager
    def step(self) -> Iterator[None]:
        self.calls += 1
        if self.calls != self.microbatch:
            yield
            return
        import torch

        trace = self.out / 'kernel_trace.json'
        receipt = self.out / 'profile_window.json'
        if trace.exists() or receipt.exists():
            raise FileExistsError('Profile evidence must be new')
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            torch.cuda.synchronize()
        # No schedule/repeats: this context covers exactly one base training_step.
        with torch.profiler.profile(activities=activities, record_shapes=False,
                                    profile_memory=False, with_stack=False,
                                    with_flops=False, with_modules=False) as prof:
            with torch.profiler.record_function('h2_selected_forward_backward'):
                yield
            if self.cuda:
                torch.cuda.synchronize()
        # Export before Python event aggregation; release profiler before next batch.
        prof.export_chrome_trace(str(trace))
        self.operations = [{'name': event.key, 'count': event.count}
                           for event in prof.key_averages()]
        self.evidence = {
            'status': 'SINGLE_MICROBATCH_TRACE_EXPORTED',
            'microbatch_1based': self.microbatch,
            'recorded_microbatches': 1,
            'activities': ['CPU', 'CUDA'] if self.cuda else ['CPU'],
            'scope': 'One base training_step forward/backward; no optimizer or full-run coverage',
            'trace_sha256': self._sha(trace), 'trace_bytes': trace.stat().st_size,
            'executed_operations': self.operations, 'performance_verdict': 'NO_VERDICT',
        }
        receipt.write_text(json.dumps(self.evidence, indent=2) + '\n')
        del prof

    @staticmethod
    def _sha(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def require_complete(self) -> dict:
        if self.evidence is None:
            raise ValueError('Required selected-microbatch trace was not completed')
        return self.evidence
