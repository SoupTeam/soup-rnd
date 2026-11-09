"""Tiny CUDA probe only: temporarily bypasses the unqualified production refusal.

The bypass is limited to this test and must not be copied to the trainer.
"""

import os

import pytest
import torch

from benchmarks.harness import b13_moe_fixture as fixture
from benchmarks.harness.b14_group_diag import find_cublas
from soup_cli.config.schema import ExpertStreamConfig
from soup_cli.utils.layer_shard import shard_checkpoint
from soup_cli.utils.layer_stream_runtime import build_streamed_model


def _apply_nf4(base, shards, index):
    from bitsandbytes.functional import dequantize_4bit
    from safetensors.torch import load_file

    from soup_cli.utils.expert_shard import EXPERT_PARTS, expert_key
    from soup_cli.utils.layer_shard import (
        NF4_CODE_KEY,
        NF4_NESTED_CODE_KEY,
        expert_paths,
        extras_shard_path,
        layer_paths,
    )
    from soup_cli.utils.layer_stream_runtime import rebuild_quant_state

    extras = load_file(extras_shard_path(str(shards)))
    codes = {key: extras[key] for key in (NF4_CODE_KEY, NF4_NESTED_CODE_KEY) if key in extras}
    layers = base.model.layers
    with torch.no_grad():
        for idx, path in enumerate(layer_paths(str(shards), index)):
            stored = load_file(path)
            for short, spec in index.quant_specs.items():
                if short in stored:
                    state = rebuild_quant_state(short, stored, spec, codes)
                    layers[idx].get_parameter(short).copy_(
                        dequantize_4bit(stored[short], quant_state=state))
        for idx, path in expert_paths(str(shards), index).items():
            stored = load_file(path)
            experts = layers[idx].mlp.experts
            for expert in range(index.expert_count):
                for part, target in zip(EXPERT_PARTS, (experts.gate_up_proj, experts.down_proj)):
                    key = expert_key(expert, part)
                    state = rebuild_quant_state(key, stored, index.expert_quant_specs[part], codes)
                    target[expert].copy_(dequantize_4bit(stored[key], quant_state=state))


def _compare_dequant_cpu_cuda(shards, index):
    from bitsandbytes.functional import dequantize_4bit
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import (
        NF4_CODE_KEY,
        NF4_NESTED_CODE_KEY,
        extras_shard_path,
        layer_paths,
    )
    from soup_cli.utils.layer_stream_runtime import rebuild_quant_state

    extras = load_file(extras_shard_path(str(shards)))
    codes = {key: extras[key] for key in (NF4_CODE_KEY, NF4_NESTED_CODE_KEY) if key in extras}
    stored = load_file(layer_paths(str(shards), index)[0])
    mismatches = {}
    for short, spec in index.quant_specs.items():
        if short not in stored:
            continue
        cpu = dequantize_4bit(stored[short], rebuild_quant_state(short, stored, spec, codes))
        gpu_stored = {key: value.cuda() for key, value in stored.items()}
        gpu_codes = {key: value.cuda() for key, value in codes.items()}
        gpu = dequantize_4bit(gpu_stored[short],
                              rebuild_quant_state(short, gpu_stored, spec, gpu_codes)).cpu()
        if not torch.equal(cpu, gpu):
            mismatches[short] = float((cpu.float() - gpu.float()).abs().max())
    return mismatches


def _match_quantized_lora_accumulation(model):
    """Reference PEFT 4-bit's BF16-rounded LoRA addition, retaining dense weights."""
    import types

    from peft.tuners.lora.layer import Linear

    def forward(module, x, *args, **kwargs):
        result = module.base_layer(x, *args, **kwargs)
        for name in module.active_adapters:
            if name in module.lora_A:
                adapted = x.to(module.lora_A[name].weight.dtype)
                contribution = module.lora_B[name](module.lora_A[name](
                    module.lora_dropout[name](adapted))) * module.scaling[name]
                result = result + contribution.to(result.dtype)
        return result

    for module in model.modules():
        if isinstance(module, Linear):
            module.forward = types.MethodType(forward, module)


def _match_nf4_projection_kernel(base, shards, index):
    """Keep resident topology, but match the actual frozen bitsandbytes projection op."""
    import types

    import bitsandbytes as bnb
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import (
        NF4_CODE_KEY,
        NF4_NESTED_CODE_KEY,
        extras_shard_path,
        layer_paths,
    )
    from soup_cli.utils.layer_stream_runtime import rebuild_quant_state

    extras = load_file(extras_shard_path(str(shards)))
    codes = {key: extras[key].cuda() for key in (NF4_CODE_KEY, NF4_NESTED_CODE_KEY)
             if key in extras}
    for idx, path in enumerate(layer_paths(str(shards), index)):
        stored = {key: value.cuda() for key, value in load_file(path).items()}
        for short, spec in index.quant_specs.items():
            if short not in stored:
                continue
            module = base.model.layers[idx].get_submodule(short.removesuffix(".weight"))
            packed = stored[short]
            state = rebuild_quant_state(short, stored, spec, codes)

            def forward(module, x, *, packed=packed, state=state):
                return bnb.matmul_4bit(x, packed.t(), bias=module.bias, quant_state=state)

            module.forward = types.MethodType(forward, module)


@pytest.mark.gpu
@pytest.mark.parametrize("storage", ["dense", "nf4", "nf4-nested"])
@pytest.mark.parametrize("seq", [6, 64])
def test_tiny_deepseek_cuda_grouped_correctness(tmp_path, monkeypatch, storage, seq):
    from soup_cli.utils import expert_training
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, quantised_layer_suffixes

    library = find_cublas()
    assert library is not None
    backend = os.environ.get("D3_TEST_BACKEND", "grouped")
    assert backend in ("grouped", "sequential")

    def experimental_only(config, options, *, device, dtype):
        assert config.model_type == "deepseek_v3"
        assert (config.hidden_size, config.moe_intermediate_size) == (32, 16)
        assert str(device) == "cuda" and dtype == "bfloat16"
        assert options.backend == backend and options.cublas_library == str(library)

    monkeypatch.setattr(expert_training, "validate_expert_training", experimental_only)
    config = fixture.moe_config()
    weights = fixture.moe_weights_dir(tmp_path, config)
    shards = tmp_path / "shards"
    quant = "none" if storage == "dense" else "nf4"
    nested = storage == "nf4-nested"
    extra = {}
    if quant == "nf4":
        shell = build_meta_skeleton(str(weights), dtype="bfloat16", quant="nf4",
                                    double_quant=nested)
        extra = {"quant_suffixes": quantised_layer_suffixes(shell), "quant_device": "cuda",
                 "double_quant": nested, "quant": quant}
    index = shard_checkpoint(str(weights), str(shards), dtype="bfloat16", arch="deepseek_v3",
                             expert_layout="per_expert", **extra)
    lora = fixture.moe_lora_config()
    options = ExpertStreamConfig(backend=backend, weight_budget_bytes=2**22,
                                 cublas_library=str(library))
    streamed, runtime = build_streamed_model(
        model_id=str(weights), shard_dir=str(shards), index=index, lora_config=lora,
        device="cuda", dtype="bfloat16", buffers=2, pin=False, seed=71, quant=quant,
        double_quant=nested,
        expert_options=options,
    )
    try:
        fixture.randomize_adapters(streamed)
        base = fixture.resident_base(weights, dtype="bfloat16")
        if quant == "nf4":
            _apply_nf4(base, shards, index)
            print("CPU_GPU_DEQUANT_DIFF", _compare_dequant_cpu_cuda(shards, index), flush=True)
            _match_nf4_projection_kernel(base, shards, index)
        base = base.to("cuda")
        resident = fixture.resident_adapter(base, lora, source=streamed)
        if quant == "nf4":
            _match_quantized_lora_accumulation(resident)
        if os.environ.get("D3_NEGATIVE_CONTROL") == "mutate_experts":
            with torch.no_grad():
                resident.get_base_model().model.layers[1].mlp.experts.down_proj.add_(0.05)
        from soup_cli.utils.layer_stream_runtime import decoder_owner

        traces = [{}, {}]
        hooks = []
        for model, trace in zip((resident, streamed), traces):
            for layer_id, layer in enumerate(decoder_owner(model).layers):
                def capture(_module, _args, output, *, layer_id=layer_id, trace=trace):
                    value = output[0] if isinstance(output, tuple) else output
                    trace[layer_id] = value.detach().cpu().clone()
                hooks.append(layer.register_forward_hook(capture))
                if layer_id == 0:
                    inner = getattr(layer, "inner", layer)
                    for part in ("input_layernorm", "self_attn.q_a_proj.base_layer",
                                 "self_attn.q_a_proj.lora_A.default",
                                 "self_attn.q_a_proj.lora_B.default",
                                 "self_attn.q_a_proj", "self_attn.q_b_proj",
                                 "self_attn.kv_a_proj_with_mqa", "self_attn.kv_b_proj",
                                 "self_attn.o_proj", "self_attn", "post_attention_layernorm",
                                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "mlp"):
                        module = inner.get_submodule(part)
                        def detail(_module, args, output, *, part=part, trace=trace):
                            value = output[0] if isinstance(output, tuple) else output
                            if args:
                                trace[part + ":input"] = args[0].detach().cpu().clone()
                            trace[part + ":output"] = value.detach().cpu().clone()
                            if part in ("self_attn.q_a_proj", "mlp.down_proj"):
                                from bitsandbytes.functional import dequantize_4bit

                                base_weight = getattr(_module, "base_layer", _module).weight
                                state = getattr(base_weight, "quant_state", None)
                                decoded = (dequantize_4bit(base_weight, quant_state=state)
                                           if state is not None else base_weight)
                                trace[part + ":base_weight"] = decoded.detach().cpu().clone()
                                if part == "mlp.down_proj":
                                    print("DOWN_META", type(_module).__name__, str(args[0].dtype),
                                          str(value.dtype), str(decoded.dtype), decoded.stride(),
                                          tuple(args[0].stride()), flush=True)
                        hooks.append(module.register_forward_hook(detail))
        batch = {key: value.to("cuda") for key, value in
                 fixture.moe_batch(config, rows=1, seq=seq).items()}
        resident.eval()
        with torch.no_grad():
            a = resident(**batch).logits
            b = resident(**batch).logits
        print("RESIDENT_AA_EXACT", torch.equal(a, b), flush=True)
        models = (resident, streamed)
        optimizers = [torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.1)
                      for model in models]
        for _ in range(2):
            outputs = []
            for model, optimizer in zip(models, optimizers):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                out = model(**batch)
                out.loss.backward()
                outputs.append(out)
            differing = {layer: float((traces[0][layer].float() - traces[1][layer].float())
                                      .abs().max()) for layer in traces[0]
                         if not torch.equal(traces[0][layer], traces[1][layer])}
            assert torch.equal(outputs[0].logits, outputs[1].logits), f"logits differ; {differing=}"
            assert torch.equal(outputs[0].loss, outputs[1].loss), "loss differs"
            reference = fixture.snapshot(resident, gradients=True)
            actual = fixture.snapshot(streamed, gradients=True)
            assert reference and reference.keys() == actual.keys()
            for name in reference:
                assert reference[name] is not None, name
                assert torch.equal(reference[name], actual[name]), name
            for optimizer in optimizers:
                optimizer.step()
            reference = fixture.snapshot(resident)
            actual = fixture.snapshot(streamed)
            assert all(torch.equal(reference[name], actual[name]) for name in reference)
        if seq == 64 and backend == "grouped":
            assert runtime.expert_resources.gemm.stats()["grouped_matrices"] > 0
    finally:
        for hook in locals().get("hooks", []):
            hook.remove()
        runtime.close()
