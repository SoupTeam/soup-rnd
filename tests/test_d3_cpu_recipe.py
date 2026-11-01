"""CPU qualification refusals precede storage reads and expert arena allocation."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("system,machine", [("linux", "x86_64"), ("win32", "amd64")])
def test_qualified_one_thread_float32_cpu_recipe(system, machine, monkeypatch):
    import torch

    from soup_cli.utils import expert_training

    monkeypatch.setattr(expert_training, "_cpu_training_platform", lambda: (system, machine))
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    expert_training.validate_cpu_expert_recipe(dtype="float32")


@pytest.mark.parametrize("threads,dtype", [(2, "float32"), (1, "bfloat16")])
def test_unqualified_thread_or_dtype_is_explicitly_refused(threads, dtype, monkeypatch):
    import torch

    from soup_cli.utils import expert_training

    monkeypatch.setattr(expert_training, "_cpu_training_platform", lambda: ("linux", "x86_64"))
    monkeypatch.setattr(torch, "get_num_threads", lambda: threads)
    with pytest.raises(ValueError, match="CPU stream_experts exactness requires"):
        expert_training.validate_cpu_expert_recipe(dtype=dtype)


@pytest.mark.parametrize("entry", ["builder", "installer"])
@pytest.mark.parametrize("system,machine", [
    ("darwin", "arm64"), ("darwin", "x86_64"), ("linux", "aarch64"), ("win32", "arm64"),
])
def test_unqualified_platform_is_refused_before_any_cache_access(
    entry, system, machine, monkeypatch,
):
    from soup_cli.utils import expert_training, layer_stream_runtime

    config = SimpleNamespace(model_type="olmoe", output_router_logits=False, hidden_act="silu")
    model = SimpleNamespace(config=config)
    options = SimpleNamespace(backend="grouped")
    monkeypatch.setattr(expert_training, "_cpu_training_platform", lambda: (system, machine))
    monkeypatch.setattr(layer_stream_runtime, "build_meta_skeleton", lambda *a, **kw: model)

    class UnreadableIndex:
        def __getattribute__(self, name):
            raise AssertionError(f"read cache metadata before CPU qualification: {name}")

    def forbidden(*args, **kwargs):
        raise AssertionError("materialized weights before CPU qualification")

    monkeypatch.setattr(layer_stream_runtime, "materialize_extras", forbidden)
    with pytest.raises(ValueError, match="Apple Silicon/other CPU platforms are unqualified"):
        if entry == "builder":
            layer_stream_runtime.build_streamed_model(
                model_id="unused", shard_dir="unused", index=UnreadableIndex(),
                lora_config=None, expert_options=options, device="cpu", dtype="float32",
            )
        else:
            expert_training.install_expert_training(
                model, "unused", UnreadableIndex(), options, device="cpu",
                dtype="float32", codes={},
            )


def test_cpu_qualification_does_not_change_cuda_sequential_validation(monkeypatch):
    from soup_cli.utils import expert_training

    def forbidden(**kwargs):
        raise AssertionError("CUDA incorrectly entered CPU qualification")

    monkeypatch.setattr(expert_training, "validate_cpu_expert_recipe", forbidden)
    expert_training.validate_expert_training(
        SimpleNamespace(model_type="olmoe"), SimpleNamespace(backend="sequential"),
        device="cuda", dtype="bfloat16",
    )
