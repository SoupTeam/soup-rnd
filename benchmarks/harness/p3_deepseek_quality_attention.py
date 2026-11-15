"""Strict native-BF16 attention policy and bounded executed CUDA kernel receipts.

Only the approved efficient backend and the independent MATH floor are supported.
Profiling covers sampled SDPA forwards and their attention-only autograd nodes,
not model windows. Receipts contain metadata and correlated kernel names only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

MAIN_BACKEND = "EFFICIENT_ATTENTION"
FLOOR_BACKEND = "MATH"
_POLICY_PATH = "benchmarks/results/p3-deepseek-quality/attention-protocol-v1.json"
_POLICY_SHA256 = "fc887c5a996bd58536ef0325b61405cd73a2a83cf363fcc4156204ab180e1ed6"
_MAX_SIGNATURES = 16
_MAX_PROFILED_OPERATIONS = 5 * _MAX_SIGNATURES
_FLAGS = {
    "algorithms_enabled": True, "warn_only": False, "allow_tf32": False,
    "cudnn_allow_tf32": False, "cudnn_benchmark": False, "cudnn_deterministic": True,
}
_CALL_FIELDS = (
    "dtype", "device", "device_index", "key_dtype", "value_dtype", "key_device", "value_device",
    "query_shape", "key_shape", "value_shape",
)


def policy_identity(profile_id: str = "deepseek-v3") -> dict[str, Any]:
    """Admit the profile's frozen bytes, never a caller-selected replacement contract."""
    path, expected = _POLICY_PATH, _POLICY_SHA256
    if profile_id != "deepseek-v3":
        from benchmarks.harness.p3_deepseek_quality_profiles import get_profile

        profile = get_profile(profile_id)
        path = "benchmarks/results/p3-moonlight-quality/attention-protocol-v1.json"
        expected = profile.attention_protocol_sha256
    payload = (Path(__file__).parents[2] / path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("attention protocol differs from the approved immutable SHA256")
    contract = json.loads(payload)
    if (not isinstance(contract, dict)
            or contract.get("main_backend") != MAIN_BACKEND
            or contract.get("alternative_noise_floor_backend") != FLOOR_BACKEND
            or contract.get("fallback_allowed") is not False
            or contract.get("determinism") != {
                "algorithms_enabled": True, "allow_tf32": False,
                "autocast": False, "warn_only": False,
            }):
        raise ValueError("attention protocol does not describe the approved backend contract")
    return {"path": path, "sha256": expected, "contract": contract}


def require_ampere_cuda(device: str) -> dict[str, Any]:
    """Reject CPU, pre-Ampere CUDA and BF16 emulation before model construction."""
    import torch

    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("attention qualification requires Ampere-or-newer CUDA, never CPU")
    if not torch.cuda.is_available():
        raise ValueError("attention qualification requires available CUDA")
    index = torch.cuda.current_device() if target.index is None else target.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError("attention CUDA device index is unavailable")
    target = torch.device("cuda", index)
    capability = list(torch.cuda.get_device_capability(target))
    if tuple(capability) < (8, 0):
        raise ValueError("attention requires Ampere CUDA capability >= 8.0 and native BF16")
    with torch.cuda.device(target):
        try:
            native_bf16 = bool(torch.cuda.is_bf16_supported(including_emulation=False))
        except TypeError as error:
            raise ValueError(
                "CUDA build cannot establish non-emulated native BF16 support"
            ) from error
    if not native_bf16:
        raise ValueError("attention requires native CUDA BF16, not emulated BF16")
    return {
        "device": str(target), "name": torch.cuda.get_device_name(target),
        "capability": capability, "native_bf16": native_bf16, "bf16_supported": native_bf16,
        "native_bf16_check": "torch.cuda.is_bf16_supported(including_emulation=False)",
        "cuda_version": torch.version.cuda,
    }


def _flags() -> dict[str, bool]:
    import torch

    return {
        "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def _backend_flags() -> dict[str, bool]:
    import torch

    return {
        "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
    }


def _expected_backend_flags(backend: str) -> dict[str, bool]:
    return {"efficient": backend == MAIN_BACKEND, "math": backend == FLOOR_BACKEND,
            "flash": False, "cudnn": False}


def _signature(call: dict[str, Any]) -> str:
    return json.dumps({name: call[name] for name in _CALL_FIELDS}, sort_keys=True)


def _validate_call(call: Any, *, dtype: str, hardware: dict) -> str:
    if not isinstance(call, dict) or any(name not in call for name in _CALL_FIELDS):
        raise ValueError("attention Q/K/V metadata is incomplete")
    for name in ("dtype", "key_dtype", "value_dtype"):
        if call[name] != "torch." + dtype:
            raise ValueError("attention Q/K/V compute dtype differs")
    index = call["device_index"]
    if (type(index) is not int or index < 0 or hardware.get("device") != f"cuda:{index}"
            or any(call[name] != "cuda" for name in ("device", "key_device", "value_device"))
            or any(call.get(name, index) != index
                   for name in ("key_device_index", "value_device_index"))):
        raise ValueError("attention Q/K/V must execute on the admitted CUDA device")
    shapes = [call[name] for name in ("query_shape", "key_shape", "value_shape")]
    if any(not isinstance(shape, list) or len(shape) != 4
           or any(type(size) is not int or size < 1 for size in shape) for shape in shapes):
        raise ValueError("attention Q/K/V shapes must be nonempty four-dimensional tensors")
    query, key, value = shapes
    if query[:2] != key[:2] or key[:3] != value[:3] or query[-1] != key[-1]:
        raise ValueError("attention Q/K/V head, batch or sequence dimensions differ")
    return _signature(call)


def require_execution_evidence(
    report: dict[str, Any], *, backend: str, dtype: str, backward: bool = True, min_calls: int = 1,
    profile_id: str = "deepseek-v3",
) -> None:
    """Validate live or serialized execution evidence without trusting requested flags."""
    if (profile_id not in ("deepseek-v3", "moonlight-16b-a3b")
            or not isinstance(report, dict)
            or report.get("profile_id", "deepseek-v3") != profile_id):
        raise ValueError("attention evidence belongs to a different model profile")
    if backend not in (MAIN_BACKEND, FLOOR_BACKEND) or dtype not in ("bfloat16", "float32"):
        raise ValueError("unsupported attention backend or dtype")
    if type(min_calls) is not int or min_calls < 1 or type(backward) is not bool:
        raise ValueError("attention admission requires a positive minimum call count")
    if not isinstance(report, dict) or report.get("backend_requested") != backend:
        raise ValueError("attention requested backend differs")
    if (type(report.get("executed_sdpa_calls")) is not int
            or report["executed_sdpa_calls"] < min_calls):
        raise ValueError("attention executed SDPA call evidence is incomplete")
    if (report.get("deterministic_flags") != _FLAGS
            or report.get("backend_flags") != _expected_backend_flags(backend)):
        raise ValueError("attention deterministic/backend flags differ from the strict policy")
    hardware = report.get("hardware", {})
    capability = hardware.get("capability")
    if (not isinstance(capability, list) or len(capability) != 2
            or any(type(part) is not int for part in capability) or tuple(capability) < (8, 0)
            or hardware.get("native_bf16") is not True
            or hardware.get("bf16_supported") is not True
            or hardware.get("native_bf16_check") != (
                "torch.cuda.is_bf16_supported(including_emulation=False)")
            or not isinstance(hardware.get("name"), str) or not hardware["name"]
            or not isinstance(hardware.get("cuda_version"), str) or not hardware["cuda_version"]):
        raise ValueError("attention hardware lacks native Ampere CUDA BF16 evidence")
    scope = report.get("instrumentation", {})
    if (scope.get("scope") != "bounded_attention_operators"
            or scope.get("retains_tensors") is not False
            or scope.get("full_window_trace") is not False
            or scope.get("signature_limit_exceeded") is not False
            or scope.get("profiling_errors") != []
            or scope.get("max_signatures") != _MAX_SIGNATURES
            or scope.get("max_profiled_operations") != _MAX_PROFILED_OPERATIONS
            or type(scope.get("profiled_operations")) is not int
            or not 1 <= scope["profiled_operations"] <= _MAX_PROFILED_OPERATIONS):
        raise ValueError("attention bounded profiling evidence is incomplete or inconsistent")
    calls, samples = report.get("unique_calls"), report.get("samples")
    if (not isinstance(calls, list) or not 1 <= len(calls) <= _MAX_SIGNATURES
            or not isinstance(samples, list) or len(samples) != len(calls)):
        raise ValueError("attention has absent or unprofiled Q/K/V signatures")
    signatures = [_validate_call(call, dtype=dtype, hardware=hardware) for call in calls]
    sample_signatures = [
        _validate_call(sample, dtype=dtype, hardware=hardware) for sample in samples
    ]
    if len(set(signatures)) != len(signatures) or sorted(signatures) != sorted(sample_signatures):
        raise ValueError("attention sampled Q/K/V signatures differ from executed calls")
    for sample in samples:
        for direction in (("forward", "backward") if backward else ("forward",)):
            evidence = sample.get(direction, {})
            operators, kernels = evidence.get("operators"), evidence.get("cuda_kernel_names")
            if (not isinstance(operators, list) or not operators
                    or any(not isinstance(name, str) or not name.startswith("aten::")
                           for name in operators)
                    or not isinstance(kernels, list) or not kernels
                    or any(not isinstance(name, str) or not name.strip() for name in kernels)):
                raise ValueError(
                    f"attention {direction} lacks executed operators/CUDA kernel names"
                )
            if backend == MAIN_BACKEND:
                expected = "aten::_scaled_dot_product_efficient_attention" + (
                    "_backward" if direction == "backward" else "")
                if expected not in operators:
                    raise ValueError(
                        f"attention {direction} did not execute the efficient operator"
                    )
                forbidden = ("_attention_math", "_flash_attention", "_cudnn_attention")
                if any(any(part in name for part in forbidden) for name in operators):
                    raise ValueError("attention efficient execution contains another backend")
                prefix = "fmha_cutlassF" if direction == "forward" else "fmha_cutlassB"
                if not any(prefix in name for name in kernels):
                    raise ValueError(
                        f"attention {direction} CUDA kernel is not native efficient SDPA"
                    )
                if any("flash" in name.lower() or "cudnn" in name.lower()
                       or ("fmha_cutlassB" if direction == "forward" else "fmha_cutlassF") in name
                       for name in kernels):
                    raise ValueError("attention efficient CUDA kernel backend/direction differs")
            elif direction == "forward":
                if "aten::_scaled_dot_product_attention_math" not in operators:
                    raise ValueError("attention floor did not execute the MATH forward operator")
                if any("efficient_attention" in name or "flash_attention" in name
                       or "cudnn_attention" in name for name in operators):
                    raise ValueError("attention MATH floor contains a fused backend")
            elif not {"aten::bmm", "aten::_softmax_backward_data"} <= set(operators):
                raise ValueError("attention MATH floor lacks matrix/softmax backward operators")
            if backend == FLOOR_BACKEND and any(
                "fmha" in name.lower() or "flash" in name.lower()
                or "efficient_attention" in name.lower() or "cudnn" in name.lower()
                for name in kernels
            ):
                raise ValueError("attention MATH floor contains fused attention CUDA kernels")


class _AttentionObservation:
    def __init__(
        self, backend: str, hardware: dict[str, Any], *, profile_id: str = "deepseek-v3",
    ) -> None:
        self.backend = backend
        self.hardware = hardware
        self.profile_id = profile_id
        self.count = 0
        self.calls: dict[str, dict[str, Any]] = {}
        self.samples: dict[str, dict[str, Any]] = {}
        self.forward_modes: set[tuple[str, bool]] = set()
        self.backward_attached: set[str] = set()
        self.handles: list[Any] = []
        self.profiling_errors: list[str] = []
        self.profiled_operations = 0
        self.signature_limit_exceeded = False
        self.deterministic_flags = _flags()
        self.backend_flags = _backend_flags()
        self.active_profile: Any = None

    def _check_permissions(self) -> None:
        flags, backends = _flags(), _backend_flags()
        if flags != _FLAGS or backends != _expected_backend_flags(self.backend):
            self.deterministic_flags, self.backend_flags = flags, backends
            raise ValueError("attention execution permissions changed inside the strict context")

    def _start_profile(self) -> Any:
        import torch
        from torch.profiler import ProfilerActivity, profile, supported_activities

        self._check_permissions()
        if self.active_profile is not None:
            raise RuntimeError("attention operator profiling cannot overlap")
        if self.profiled_operations >= _MAX_PROFILED_OPERATIONS:
            raise RuntimeError("attention bounded operator profiling limit exceeded")
        if ProfilerActivity.CUDA not in supported_activities():
            raise RuntimeError("attention CUDA profiler activity is unavailable")
        torch.cuda.synchronize(self.hardware["device"])
        profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                           record_shapes=False, profile_memory=False, with_stack=False)
        profiler.start()
        self.profiled_operations += 1
        self.active_profile = profiler
        return profiler

    def _stop_profile(self, profiler: Any, signature: str, direction: str) -> None:
        import torch

        try:
            torch.cuda.synchronize(self.hardware["device"])
        finally:
            try:
                profiler.stop()
            finally:
                self.active_profile = None
        operators, kernels = set(), set()
        for event in profiler.events():
            if event.name.startswith("aten::"):
                operators.add(event.name)
                pending = [event]
                while pending:
                    child = pending.pop()
                    kernels.update(kernel.name for kernel in child.kernels)
                    pending.extend(child.cpu_children)
        target = self.samples[signature][direction]
        target["operators"] = sorted(set(target["operators"]) | operators)
        target["cuda_kernel_names"] = sorted(set(target["cuda_kernel_names"]) | kernels)
        self._check_permissions()

    def _attach_backward(self, result: Any, inputs: tuple[Any, Any, Any], signature: str) -> None:
        # Node names select a small profiling range; they are never evidence of an executed kernel.
        boundaries = {value.grad_fn for value in inputs if value.grad_fn is not None}
        pending, visited, nodes = [result.grad_fn], set(), []
        while pending:
            node = pending.pop()
            if node is None or node in visited or node in boundaries:
                continue
            visited.add(node)
            if len(visited) > 256:
                raise RuntimeError("attention backward graph exceeded its bounded traversal")
            name = node.name()
            selected = ("EfficientAttentionBackward" in name if self.backend == MAIN_BACKEND
                        else "BmmBackward" in name or "SoftmaxBackward" in name)
            if selected:
                nodes.append(node)
                if self.backend == MAIN_BACKEND:
                    continue
            pending.extend(next_node for next_node, _index in node.next_functions)
        expected = 1 if self.backend == MAIN_BACKEND else 3
        if len(nodes) != expected:
            raise RuntimeError("attention could not isolate its bounded native backward nodes")
        for node in nodes:
            state: dict[str, Any] = {}

            def before(_gradients: Any, state: dict = state) -> None:
                try:
                    state["profiler"] = self._start_profile()
                except BaseException as error:
                    self.profiling_errors.append(f"backward start: {type(error).__name__}: {error}")
                    raise

            def after(_inputs: Any, _outputs: Any, state: dict = state) -> None:
                try:
                    self._stop_profile(state.pop("profiler"), signature, "backward")
                except BaseException as error:
                    self.profiling_errors.append(f"backward stop: {type(error).__name__}: {error}")
                    raise

            self.handles.extend((node.register_prehook(before), node.register_hook(after)))
        self.backward_attached.add(signature)

    def observe(self, original: Any, query: Any, key: Any, value: Any,
                *args: Any, **kwargs: Any) -> Any:
        import torch

        self._check_permissions()
        call = {
            "dtype": str(query.dtype), "device": query.device.type,
            "device_index": query.device.index, "key_dtype": str(key.dtype),
            "value_dtype": str(value.dtype), "key_device": key.device.type,
            "value_device": value.device.type, "key_device_index": key.device.index,
            "value_device_index": value.device.index,
            "query_shape": list(query.shape), "key_shape": list(key.shape),
            "value_shape": list(value.shape),
        }
        dtype = str(query.dtype).removeprefix("torch.")
        if dtype not in ("bfloat16", "float32"):
            raise ValueError("attention only admits BF16 or explicit FP32 qualification")
        signature = _validate_call(call, dtype=dtype, hardware=self.hardware)
        if signature not in self.calls:
            if len(self.calls) == _MAX_SIGNATURES:
                self.signature_limit_exceeded = True
                raise RuntimeError("attention Q/K/V signature limit exceeded")
            self.calls[signature] = call
            self.samples[signature] = {
                **call, "forward": {"operators": [], "cuda_kernel_names": []},
                "backward": {"operators": [], "cuda_kernel_names": []},
            }
        grad = torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (query, key, value)
        )
        mode = (signature, grad)
        if mode not in self.forward_modes:
            profiler = None
            try:
                profiler = self._start_profile()
                result = original(query, key, value, *args, **kwargs)
            except BaseException as error:
                self.profiling_errors.append(f"forward: {type(error).__name__}: {error}")
                raise
            finally:
                if profiler is not None:
                    self._stop_profile(profiler, signature, "forward")
            self.forward_modes.add(mode)
        else:
            result = original(query, key, value, *args, **kwargs)
        self.count += 1
        if grad and signature not in self.backward_attached:
            self._attach_backward(result, (query, key, value), signature)
        return result

    def close(self) -> None:
        try:
            if self.active_profile is not None:
                self.active_profile.stop()
                self.profiling_errors.append("backward operator did not complete")
                self.active_profile = None
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()

    def report(self) -> dict[str, Any]:
        return copy.deepcopy({
            "profile_id": self.profile_id,
            "backend_requested": self.backend, "hardware": self.hardware,
            "executed_sdpa_calls": self.count,
            "unique_calls": [self.calls[key] for key in sorted(self.calls)],
            "samples": [self.samples[key] for key in sorted(self.samples)],
            "deterministic_flags": self.deterministic_flags, "backend_flags": self.backend_flags,
            "instrumentation": {
                "scope": "bounded_attention_operators", "max_signatures": _MAX_SIGNATURES,
                "max_profiled_operations": _MAX_PROFILED_OPERATIONS,
                "profiled_operations": self.profiled_operations,
                "retains_tensors": False, "full_window_trace": False,
                "signature_limit_exceeded": self.signature_limit_exceeded,
                "profiling_errors": self.profiling_errors,
            },
        })

    def require_complete(self, *, dtype: str, backward: bool = True, min_calls: int = 1) -> None:
        self._check_permissions()
        require_execution_evidence(
            self.report(), backend=self.backend, dtype=dtype, backward=backward,
            min_calls=min_calls, profile_id=self.profile_id,
        )


@contextmanager
def attention_execution(
    backend: str, *, device: str, profile_id: str = "deepseek-v3",
) -> Iterator[_AttentionObservation]:
    """Select one backend, enforce strict determinism, restore flags without reseeding."""
    if backend not in (MAIN_BACKEND, FLOOR_BACKEND):
        raise ValueError("unsupported attention backend; only EFFICIENT_ATTENTION or MATH")
    policy_identity(profile_id)
    import torch
    import torch.nn.functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if torch.device(device).type != "cuda":
        raise ValueError("attention execution requires native Ampere CUDA, never CPU")
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (None, ":4096:8"):
        raise ValueError("attention requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if workspace is None:
        if torch.cuda.is_initialized():
            raise ValueError(
                "attention requires CUBLAS_WORKSPACE_CONFIG before CUDA initialization"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    previous = _flags()
    original = functional.scaled_dot_product_attention
    observation = None
    try:
        hardware = require_ampere_cuda(device)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with sdpa_kernel([getattr(SDPBackend, backend)]):
            observation = _AttentionObservation(backend, hardware, profile_id=profile_id)

            def observed(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
                return observation.observe(original, query, key, value, *args, **kwargs)

            functional.scaled_dot_product_attention = observed
            try:
                yield observation
            finally:
                functional.scaled_dot_product_attention = original
                observation.close()
    finally:
        torch.use_deterministic_algorithms(previous["algorithms_enabled"],
                                           warn_only=previous["warn_only"])
        torch.backends.cuda.matmul.allow_tf32 = previous["allow_tf32"]
        torch.backends.cudnn.allow_tf32 = previous["cudnn_allow_tf32"]
        torch.backends.cudnn.benchmark = previous["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]
        if workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
