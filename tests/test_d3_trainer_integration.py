"""D3 public builder / PEFT / Trainer integration on synthetic CPU OLMoE weights."""

from __future__ import annotations

import copy

import pytest


def _same_bits(left, right):
    """Exact dtype/shape/bytes, including signed zero; no numerical tolerance."""
    import torch

    return (left.dtype == right.dtype and left.shape == right.shape and torch.equal(
        left.detach().contiguous().reshape(-1).view(torch.uint8),
        right.detach().contiguous().reshape(-1).view(torch.uint8),
    ))


@pytest.fixture
def cpu_reference_threads():
    """Freeze the CPU arithmetic recipe; parallel BLAS can change rounding.

    This fixture qualifies one-thread CPU parity, not arbitrary CPU thread
    counts. CUDA parity has its own frozen math policy and qualification.
    """
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def checkpoint(tmp_path, cpu_reference_threads):
    import torch
    from transformers import OlmoeConfig, OlmoeForCausalLM

    torch.manual_seed(71)
    cfg = OlmoeConfig(
        hidden_size=32, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, num_experts=4,
        num_experts_per_tok=2, vocab_size=64, max_position_embeddings=128,
        output_router_logits=False, attention_dropout=0.0, tie_word_embeddings=True,
        eos_token_id=1, bos_token_id=2, pad_token_id=0,
    )
    model = OlmoeForCausalLM(cfg).float()
    folder = tmp_path / "checkpoint"
    model.save_pretrained(folder)
    return model, folder


def _build(
    checkpoint, tmp_path, *, backend="grouped", dropout=0.0, budget=2**20,
    quant="none", double_quant=False,
):
    from peft import LoraConfig, TaskType

    from soup_cli.config.schema import ExpertStreamConfig
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        build_streamed_model,
        quantised_layer_suffixes,
        shard_layout_of,
    )

    model, source = checkpoint
    folder = tmp_path / ("cache-" + backend)
    suffixes = ()
    if quant == "nf4":
        shell = build_meta_skeleton(str(source), dtype="float32", quant=quant,
                                    double_quant=double_quant)
        suffixes = quantised_layer_suffixes(shell)
    index = shard_checkpoint(
        str(source), str(folder), dtype="float32", arch="olmoe",
        layout=shard_layout_of(model), expert_layout="per_expert",
        quant=quant, double_quant=double_quant, quant_device="cpu", quant_suffixes=suffixes,
    )
    options = ExpertStreamConfig(backend=backend, weight_budget_bytes=budget)
    lora = LoraConfig(r=2, lora_alpha=4, lora_dropout=dropout,
                      target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
    streamed, runtime = build_streamed_model(
        model_id=str(source), shard_dir=str(folder), index=index, lora_config=lora,
        device="cpu", dtype="float32", pin=False, expert_options=options, seed=19,
        quant=quant, double_quant=double_quant,
    )
    return streamed, runtime, index, folder


@pytest.mark.parametrize("double_quant", [False, True])
def test_nf4_public_builder_runs_backward_without_dense_experts(checkpoint, tmp_path, double_quant):
    import torch

    model, runtime, index, _folder = _build(checkpoint, tmp_path, quant="nf4",
                                           double_quant=double_quant)
    try:
        assert index.expert_quant_specs
        model.train()
        loss = model(input_ids=torch.tensor([[2, 3, 4, 5]]),
                     labels=torch.tensor([[2, 3, 4, 5]])).loss
        loss.backward()
        assert torch.isfinite(loss)
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert grads and all(x is not None and torch.isfinite(x).all() for x in grads)
        assert any(x.abs().sum() > 0 for x in grads)
        assert not any(".mlp.experts." in n for n, _ in model.named_parameters())
    finally:
        runtime.close()


def _reference(checkpoint, streamed):
    from peft import get_peft_model

    base = copy.deepcopy(checkpoint[0])
    base.name_or_path = str(checkpoint[1])
    base.config.name_or_path = str(checkpoint[1])
    cfg = copy.deepcopy(streamed.peft_config["default"])
    reference = get_peft_model(base, cfg)
    sources = dict(streamed.named_parameters())
    for name, parameter in reference.named_parameters():
        if parameter.requires_grad:
            parameter.data.copy_(sources[name].data)
    return reference


@pytest.mark.parametrize("backend", ["sequential", "grouped"])
def test_public_builder_loss_gradients_updates_and_close(checkpoint, tmp_path, backend):
    import torch

    streamed, runtime, _index, _folder = _build(checkpoint, tmp_path, backend=backend)
    try:
        reference = _reference(checkpoint, streamed)
        assert reference.config._attn_implementation == streamed.config._attn_implementation
        ids = torch.tensor([[1, 5, 9, 7, 3, 2]])
        optimizers = [torch.optim.SGD((p for p in m.parameters() if p.requires_grad), lr=.1)
                      for m in (reference, streamed)]
        initial = {n: p.clone() for n, p in streamed.named_parameters() if p.requires_grad}
        for step in range(2):
            results = []
            for model, optimizer in zip((reference, streamed), optimizers):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                output = model(input_ids=ids, labels=ids)
                output.loss.backward()
                grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
                results.append((output.logits.detach(), output.loss.detach(), grads))
                optimizer.step()
            want, got = results
            context = f"{backend}, step {step}, torch {torch.__version__}"
            assert _same_bits(want[0], got[0]), f"logits: {context}"
            assert _same_bits(want[1], got[1]), f"loss: {context}"
            for key in want[2]:
                assert _same_bits(want[2][key], got[2][key]), f"gradient {key}: {context}"
        assert any(not _same_bits(initial[n], p) for n, p in streamed.named_parameters()
                   if p.requires_grad)
        assert runtime.stats()["expert_fetches"] > 0
        assert not any(".mlp.experts." in n for n, _ in streamed.named_parameters())
    finally:
        runtime.close()
        runtime.close()
    assert runtime.expert_resources.closed


def test_budget_refusal_before_pool_allocation(checkpoint, tmp_path, monkeypatch):
    from soup_cli.utils import expert_group

    def forbidden(*args, **kwargs):
        raise AssertionError("allocated a pool before the budget refusal")

    monkeypatch.setattr(expert_group, "GroupedExpertPool", forbidden)
    with pytest.raises(ValueError, match="weight.*budget"):
        _build(checkpoint, tmp_path, budget=1)


def test_install_failure_closes_resources(checkpoint, tmp_path, monkeypatch):
    from soup_cli.utils import layer_stream_runtime as runtime_module

    resources = []
    from soup_cli.utils import expert_training

    original = expert_training.install_expert_training

    def record(*args, **kwargs):
        resource = original(*args, **kwargs)
        resources.append(resource)
        return resource

    def fail(*args, **kwargs):
        raise RuntimeError("injected layer-runtime failure")

    monkeypatch.setattr(expert_training, "install_expert_training", record)
    monkeypatch.setattr(runtime_module, "install_streaming", fail)
    with pytest.raises(RuntimeError, match="injected"):
        _build(checkpoint, tmp_path)
    assert len(resources) == 1 and resources[0].closed


def test_source_failure_still_closes_expert_readers(checkpoint, tmp_path, monkeypatch):
    _model, runtime, _index, _folder = _build(checkpoint, tmp_path)

    def fail():
        raise RuntimeError("injected source close failure")

    monkeypatch.setattr(runtime.source, "close", fail, raising=False)
    with pytest.raises(RuntimeError, match="source close"):
        runtime.close()
    assert runtime.expert_resources.closed
    assert runtime.hook is None


def test_last_layer_rejection_is_atomic_and_precedes_arena_allocation(
    checkpoint, tmp_path, monkeypatch,
):
    from soup_cli.config.schema import ExpertStreamConfig
    from soup_cli.utils import expert_group
    from soup_cli.utils.expert_training import install_expert_training

    _streamed, runtime, index, folder = _build(checkpoint, tmp_path)
    runtime.close()
    model = copy.deepcopy(checkpoint[0]).requires_grad_(False)
    originals = [layer.mlp.experts for layer in model.model.layers]
    originals[-1].down_proj.requires_grad_(True)

    def forbidden(*args, **kwargs):
        raise AssertionError("allocated before validating the last layer")

    monkeypatch.setattr(expert_group, "GroupedExpertPool", forbidden)
    with pytest.raises(ValueError, match="frozen-weight mismatch"):
        install_expert_training(model, str(folder), index, ExpertStreamConfig(),
                                device="cpu", dtype="float32", codes={})
    assert all(layer.mlp.experts is old for layer, old in zip(model.model.layers, originals))


def test_cuda_math_policy_ownership_and_error_restoration_without_cuda(monkeypatch):
    from contextlib import ExitStack

    import torch

    from soup_cli.utils.expert_training import _math_policy

    selected = ["original"]

    def select(value=None):
        if value is not None:
            selected[0] = value
        return selected[0]

    # Exercise the real torch flag setters on CPU. The split-K attribute has
    # only a getter; its value is written through the reduction tuple setter.
    matmul = torch.backends.cuda.matmul
    prior = (matmul.allow_tf32, matmul.allow_bf16_reduced_precision_reduction,
             matmul.allow_bf16_reduced_precision_reduction_split_k)
    monkeypatch.setattr(torch.backends.cuda, "preferred_blas_library", select)
    try:
        matmul.allow_tf32 = True
        matmul.allow_bf16_reduced_precision_reduction = (True, True)
        with pytest.raises(ValueError, match="injected"):
            with ExitStack() as owner:
                _math_policy(owner)
                assert selected[0] == "cublaslt" and not matmul.allow_tf32
                assert not matmul.allow_bf16_reduced_precision_reduction
                assert not matmul.allow_bf16_reduced_precision_reduction_split_k
                with ExitStack() as other, pytest.raises(RuntimeError, match="owns"):
                    _math_policy(other)
                raise ValueError("injected allocation failure")
        assert selected[0] == "original"
        assert matmul.allow_tf32 and matmul.allow_bf16_reduced_precision_reduction
        assert matmul.allow_bf16_reduced_precision_reduction_split_k
    finally:
        matmul.allow_tf32 = prior[0]
        matmul.allow_bf16_reduced_precision_reduction = prior[1:]


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_olmoe_l2l_matches_same_streamed_ga_with_ignored_labels(
    checkpoint, tmp_path, k, dropout,
):
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    model, runtime, _index, _folder = _build(checkpoint, tmp_path, dropout=dropout)
    try:
        ids = [torch.tensor([[1, 5, 9, 7, 3, 2]]) + i for i in range(k)]
        labels = [x.clone() for x in ids]
        labels[0][0, 2] = -100
        count = sum(int((y[:, 1:] != -100).sum()) for y in labels)
        model.train()
        model.zero_grad(set_to_none=True)
        rng_before = torch.get_rng_state().clone()
        reference_losses = []
        for x, y in zip(ids, labels):
            loss = model(input_ids=x, labels=y, num_items_in_batch=count).loss
            reference_losses.append(loss.detach())
            loss.backward()
        reference = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
        rng_after = torch.get_rng_state().clone()
        torch.set_rng_state(rng_before)
        model.zero_grad(set_to_none=True)
        step = L2LStep(model, runtime, capture_layer_call(model, ids[0]))
        store = ActivationStore(n_layers=2, k=k, chunk_shape=(1, 6, 32),
                                dtype=torch.float32, device="cpu", pin=False)
        try:
            losses = step.run(ids, store, n_items=count, labels=labels)
        finally:
            store.close()
        assert all(_same_bits(a, b) for a, b in zip(losses, reference_losses))
        assert _same_bits(torch.get_rng_state(), rng_after)
        for n, p in model.named_parameters():
            if p.requires_grad:
                assert _same_bits(reference[n], p.grad), n
    finally:
        runtime.close()


def test_opt_in_config_refuses_unsupported_training_before_model_load():
    from soup_cli.config.schema import ExpertStreamConfig, SoupConfig

    common = dict(base="local/olmoe", task="sft", data={"train": "local.jsonl"})
    with pytest.raises(ValueError, match="stream_experts.*stream_layers"):
        SoupConfig(**common, training={"stream_experts": {}})
    for width in [True, 0, 5]:
        with pytest.raises(ValueError):
            ExpertStreamConfig(group_width=width)


def test_router_auxiliary_loss_is_not_silently_dropped():
    from types import SimpleNamespace

    from soup_cli.config.schema import ExpertStreamConfig, SoupConfig
    from soup_cli.trainer.l2l_trainer import refuse_unsupported_l2l
    from soup_cli.utils.expert_training import validate_expert_training

    config = SimpleNamespace(model_type="olmoe", output_router_logits=True)
    with pytest.raises(ValueError, match="auxiliary"):
        validate_expert_training(config, ExpertStreamConfig(), device="cpu", dtype="float32")
    with pytest.raises(ValueError, match="auxiliary"):
        refuse_unsupported_l2l(config, n_gpu=0)
    with pytest.raises(ValueError, match="moe_aux_loss_coeff=0"):
        SoupConfig(base="local/olmoe", data={"train": "local.jsonl"}, training={
            "stream_layers": True, "stream_experts": {}, "batch_size": 1,
        })


def test_sft_setup_mixin_selects_expert_layout_from_the_config(checkpoint, tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.stream_setup import StreamingSetupMixin

    raw = Tokenizer(WordLevel({str(i): i for i in range(64)}, unk_token="0"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="0", eos_token="1")
    tokenizer.save_pretrained(checkpoint[1])
    config = SoupConfig(
        base=str(checkpoint[1]), task="sft", data={"train": str(tmp_path / "data.jsonl")},
        training={"backend": "transformers", "stream_layers": True, "stream_source": "ram",
                  "batch_size": 1,
                  "moe_aux_loss_coeff": 0.0,
                  "stream_experts": {"backend": "grouped"}, "quantization": "none",
                  "lora": {"r": 2, "alpha": 4, "dropout": 0.0,
                           "target_modules": ["q_proj", "v_proj"]}},
    )
    setup = StreamingSetupMixin()
    setup.device = "cpu"
    setup._trust_remote_code = False
    try:
        plain = config.model_copy(deep=True)
        plain.training.stream_experts = None
        with pytest.raises(ValueError, match="requires explicit"):
            setup._setup_streaming_transformers(plain, plain.training)
        setup._setup_streaming_transformers(config, config.training)
        assert setup._stream_runtime.expert_resources is not None
        assert setup._stream_runtime.stats()["expert_backend"] == "grouped"
        assert setup._stream_runtime.stats()["expert_source"] == "disk-buffered"
    finally:
        setup._close_stream_runtime()


@pytest.mark.parametrize("l2l", [False, True])
@pytest.mark.parametrize("codec", ["none", "nf4", "nf4-dq"])
def test_actual_sft_trainer_steps_and_adapter_roundtrip(checkpoint, tmp_path, l2l, codec):
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, default_data_collator
    from trl import SFTConfig, SFTTrainer

    from soup_cli.trainer.l2l_trainer import attach_l2l_state, make_l2l_trainer_class

    quant = "none" if codec == "none" else "nf4"
    model, runtime, _index, _folder = _build(
        checkpoint, tmp_path, quant=quant, double_quant=codec == "nf4-dq",
    )
    raw = Tokenizer(WordLevel({str(i): i for i in range(64)}, unk_token="0"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="0", eos_token="1")
    args = SFTConfig(
        output_dir=str(tmp_path / "train"), use_cpu=True, bf16=False, fp16=False,
        max_steps=2, per_device_train_batch_size=1, gradient_accumulation_steps=2,
        learning_rate=.01, optim="sgd", report_to="none", save_strategy="no",
        logging_strategy="no", disable_tqdm=True, gradient_checkpointing=False,
        dataset_kwargs={"skip_prepare_dataset": True}, dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    rows = [[2, 3, 4, 5, 6, 1], [4, 5, 6, 7, 8, 1], [3, 4, 5, 6, 7, 1]]
    dataset = Dataset.from_dict({
        "input_ids": rows, "labels": rows, "attention_mask": [[1] * len(x) for x in rows],
    })
    trainer_cls = make_l2l_trainer_class(SFTTrainer) if l2l else SFTTrainer
    try:
        trainer = trainer_cls(model=model, args=args, train_dataset=dataset,
                              processing_class=tokenizer, data_collator=default_data_collator)
        if l2l:
            attach_l2l_state(trainer, runtime, dtype=torch.float32)
        result = trainer.train()
        assert result.global_step == 2
        assert runtime.stats()["expert_fetches"] > 0
        model.eval()
        ids = torch.tensor([rows[0]])
        with torch.no_grad():
            wanted = model(input_ids=ids).logits.clone()
        target = tmp_path / "adapter"
        trainer.save_model(str(target))
        if codec == "none":
            reloaded = PeftModel.from_pretrained(copy.deepcopy(checkpoint[0]), target).eval()
            with torch.no_grad():
                assert _same_bits(reloaded(input_ids=ids).logits, wanted)
        else:
            from peft import get_peft_model_state_dict, set_peft_model_state_dict
            from safetensors.torch import load_file

            reloaded, other_runtime, _, _ = _build(
                checkpoint, tmp_path / "reload", quant=quant, double_quant=codec == "nf4-dq",
            )
            try:
                # Restore through the same Trainer setup: TRL casts QLoRA adapters
                # to BF16 during construction, even for a float32 CPU base. Loading
                # those factors into fresh float32 adapters changes the GEMM recipe.
                restored_trainer = SFTTrainer(
                    model=reloaded, args=copy.deepcopy(args), train_dataset=dataset,
                    processing_class=tokenizer, data_collator=default_data_collator,
                )
                reloaded = restored_trainer.model
                state = load_file(str(target / "adapter_model.safetensors"))
                assert state and all("lora_" in key for key in state)
                outcome = set_peft_model_state_dict(reloaded, state)
                assert not outcome.unexpected_keys, outcome.unexpected_keys
                actual_state = get_peft_model_state_dict(reloaded)
                assert set(actual_state) == set(state)
                for key in state:
                    assert actual_state[key].dtype == state[key].dtype
                    assert _same_bits(actual_state[key], state[key]), key
                reloaded.eval()
                with torch.no_grad():
                    assert _same_bits(reloaded(input_ids=ids).logits, wanted)
            finally:
                other_runtime.close()
        if l2l:
            assert trainer._soup_l2l.steps == 2
            trainer._soup_l2l.close()
    finally:
        runtime.close()


@pytest.mark.parametrize("l2l", [False, True])
def test_public_sft_wrapper_trains_saves_and_releases_experts(checkpoint, tmp_path, l2l):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.sft import SFTTrainerWrapper

    raw = Tokenizer(WordLevel({str(i): i for i in range(64)}, unk_token="0"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="0", eos_token="1")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )
    tokenizer.save_pretrained(checkpoint[1])
    output = tmp_path / "wrapper-adapter"
    config = SoupConfig(
        base=str(checkpoint[1]), output=str(output),
        data={"train": str(tmp_path / "data.jsonl"), "max_length": 64, "val_split": 0,
              "train_on_responses_only": False},
        training={"backend": "transformers", "stream_layers": True, "stream_source": "ram",
                  "batch_size": 1, "gradient_accumulation_steps": 2, "epochs": 1,
                  "optimizer": "sgd", "rewind_log": False, "packing": False,
                  "seed": 19, "l2l": l2l, "moe_aux_loss_coeff": 0,
                  "quantization": "none", "stream_experts": {"backend": "grouped"},
                  "lora": {"r": 2, "alpha": 4, "dropout": 0,
                           "target_modules": ["q_proj", "v_proj"]}},
    )
    wrapper = SFTTrainerWrapper(config, device="cpu")
    rows = [{"messages": [{"role": "user", "content": " ".join(
        str(3 + (i + j) % 60) for j in range(64)
    )}]} for i in range(3)]
    try:
        wrapper.setup({"train": rows, "val": []})
        runtime = wrapper._stream_runtime
        result = wrapper.train()
        assert result["total_steps"] == 2
        assert runtime.expert_resources.closed
        assert runtime.stats()["expert_fetches"] > 0
        assert (output / "adapter_model.safetensors").is_file()
        assert (output / "adapter_config.json").is_file()
        assert (output / "tokenizer.json").is_file()
    finally:
        wrapper._close_stream_runtime()
