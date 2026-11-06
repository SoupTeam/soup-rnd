"""Real Soup/TRL/HF CE-only path on CPU with Liger kernels substituted by a stand-in.

Plumbing only: Liger kernels substituted; not Liger/CUDA/NF4/8B validation. The real
PatchController (h2_operator_patch) is installed in each fresh subprocess over a stand-in
``liger_kernel.transformers.monkey_patch`` module that exists only in that subprocess. Its
``apply_liger_kernel_to_llama`` has the pinned parameter set; for ``cross_entropy=True`` it
assigns one fixed spy function to ``torch.nn.functional.cross_entropy``. The spy calls the stock
op saved once and records reduction, ignore_index and target count. Each arm runs in its own
subprocess because the rebinding is process-global; the pair comparison is a separate step.

Not covered: Liger/Triton kernels and their numerics, CUDA, NF4, the 8B model, the
128 frozen Llama-3.1 rows (needs the real tokenizer, H2-L2), PatchController.verify (its 32-layer
module checks) and kernel_audit.
"""

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
KEYS = ("input_ids", "attention_mask", "labels")
PINS = {"trl": "0.29.1", "transformers": "5.17.0"}


def _installed(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


pytestmark = pytest.mark.skipif(
    any(_installed(name) != version for name, version in PINS.items()),
    reason="Asserts pinned TRL 0.29.1 / transformers 5.17.0 dataset and loss behavior",
)

CHILD = r"""
import copy, importlib.machinery, json, sys, types, warnings
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as functional
import torch.optim.adam as adam_module
import trl
from datasets import Dataset

mode, arm, variant, root = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
KEYS = ('input_ids', 'attention_mask', 'labels')
USERS = ['hello', 'world', 'hello world', 'world hello']
ANSWERS = ['answer', 'answer seven', 'seven answer seven', 'answer seven world hello']
# Answer-major order: each accumulation window has a different useful-target count.
ROWS = [{'messages': [{'role': 'user', 'content': user},
                      {'role': 'assistant', 'content': answer}]}
        for answer in ANSWERS for user in USERS]
events = {'ce': [], 'patch': []}
stock_ce = functional.cross_entropy  # Saved once; the spy delegates to this.


def liger_cross_entropy(*args, **kwargs):
    # Grad-enabled calls are the training loss; Soup's rewind recorder adds a no-grad
    # per-row diagnostic call with reduction='none' (soup_cli/trainer/rewind_hf.py).
    target = kwargs.get('target', args[1] if len(args) > 1 else None)
    ignore = kwargs.get('ignore_index', -100)
    events['ce'].append({'reduction': kwargs.get('reduction', 'mean'), 'ignore_index': ignore,
                         'grad': torch.is_grad_enabled(),
                         'targets': int((target != ignore).sum())})
    if variant == 'mean' and torch.is_grad_enabled():
        kwargs['reduction'] = 'mean'
    return stock_ce(*args, **kwargs)


def apply_liger_kernel_to_llama(rope=True, cross_entropy=False, fused_linear_cross_entropy=True,
                                rms_norm=True, swiglu=True, model=None):
    events['patch'].append({
        'flags': dict(rope=rope, cross_entropy=cross_entropy, rms_norm=rms_norm, swiglu=swiglu,
                      fused_linear_cross_entropy=fused_linear_cross_entropy),
        'model': None if model is None else type(model).__name__})
    if rope or fused_linear_cross_entropy or rms_norm or swiglu:
        raise ValueError('Stand-in implements the cross_entropy flag only')
    if cross_entropy and variant != 'no_rebind':
        functional.cross_entropy = liger_cross_entropy


def standin(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    sys.modules[name] = module
    return module


class LigerRMSNorm: pass
class LigerSwiGLUMLP: pass
def liger_rotary_pos_emb(*args, **kwargs): raise NotImplementedError
def llama_lce_forward(*args, **kwargs): raise NotImplementedError


monkey_patch = standin(
    'liger_kernel.transformers.monkey_patch', liger_cross_entropy=liger_cross_entropy,
    apply_liger_kernel_to_llama=apply_liger_kernel_to_llama, LigerRMSNorm=LigerRMSNorm,
    LigerSwiGLUMLP=LigerSwiGLUMLP, liger_rotary_pos_emb=liger_rotary_pos_emb,
    llama_lce_forward=llama_lce_forward)
standin('liger_kernel.transformers', monkey_patch=monkey_patch)
standin('liger_kernel')

import soup_cli
from h2_ce_confirmation import validate_prepared_batches
from h2_ce_training import Recorder, recording_trainer
from h2_operator_patch import PatchController, validate_counts
from h2_trainer_smoke import build_local_model
from soup_cli.config.schema import SoupConfig
from soup_cli.trainer.sft import SFTTrainerWrapper
from transformers.trainer_utils import unwrap_peft_model

# The prepare mode builds one Liger-arm wrapper, so its controller arm is ce_only.
control = PatchController('ce_only' if mode == 'prepare' else arm)
if variant == 'noop_step':
    adam_module.adam = lambda *args, **kwargs: None
build_local_model(root / 'model')


def controller_state():
    now = control.bindings()
    names = {key: 'liger' if value is liger_cross_entropy else
             'stock' if value is control.original[key] else 'other' for key, value in now.items()}
    try:
        validate_counts(control.arm, control.counts)
        counts_ok = True
    except ValueError:
        counts_ok = False
    return {'arm': control.arm, 'counts': dict(control.counts), 'counts_ok': counts_ok,
            'bindings': names, 'patch': events['patch'], 'ce': events['ce'],
            'cross_entropy_is_spy': functional.cross_entropy is liger_cross_entropy}


def make_wrapper(use_liger, tag, shuffle=False):
    class CPUWrapper(SFTTrainerWrapper):
        @staticmethod
        def _as_sft_config(args, max_length, packing=False):
            args.use_cpu = True
            result = SFTTrainerWrapper._as_sft_config(args, max_length, packing)
            if shuffle:
                result.shuffle_dataset = True
            return result

    config = SoupConfig.model_validate({
        'base': str(root / 'model'), 'task': 'sft', 'backend': 'transformers',
        'data': {'train': 'in-memory', 'format': 'alpaca', 'max_length': 64},
        'output': str(root / ('output_' + tag)),
        'training': {'epochs': 1, 'batch_size': 1, 'gradient_accumulation_steps': 4,
                     'quantization': 'none', 'seed': 42, 'data_seed': 42, 'lr': 0.0002,
                     'optimizer': 'adamw_torch', 'scheduler': 'constant', 'warmup_ratio': 0.0,
                     'weight_decay': 0.0, 'gradient_checkpointing': True,
                     'auto_mixed_precision': False, 'use_liger': use_liger,
                     'logging_steps': 1, 'save_steps': 1000,
                     'lora': {'r': 2, 'alpha': 4, 'dropout': 0.0,
                              'target_modules': ['q_proj', 'v_proj']}}})
    wrapper = CPUWrapper(config, device='cpu', report_to='none')
    wrapper.setup({'train': copy.deepcopy(ROWS)})
    return wrapper


def failure(trainer_like, expected):
    try:
        validate_prepared_batches(trainer_like, expected)
    except Exception as error:
        return type(error).__name__ + ': ' + str(error)
    return None


def environment():
    import datasets, peft, transformers
    return {'executable': sys.executable, 'soup_cli': soup_cli.__file__,
            **{m.__name__: [m.__version__, m.__file__]
               for m in (transformers, datasets, trl, torch, peft)}}


def prepare():
    base = make_wrapper(False, 'baseline')
    liger = make_wrapper(True, 'ce_only')
    expected = [{k: list(row[k]) for k in KEYS} for row in base.trainer.train_dataset]
    trainer = liger.trainer
    args, collator = trainer.args, trainer.data_collator
    try:
        [{k: row[k] for k in KEYS} for row in trainer.train_dataset]
        prefix = None
    except KeyError as error:
        prefix = 'KeyError: ' + str(error)
    real = [copy.deepcopy(dict(trainer.train_dataset[i]))
            for i in range(len(trainer.train_dataset))]
    negatives = {}
    mutated = copy.deepcopy(real)
    mutated[0]['assistant_masks'][mutated[0]['assistant_masks'].index(1)] = 0
    negatives['assistant_mask'] = mutated
    mutated = copy.deepcopy(real)
    mutated[0]['input_ids'][1] = 5 if mutated[0]['input_ids'][1] != 5 else 4
    negatives['token'] = mutated
    negatives['order'] = [real[1], real[0], *real[2:]]
    base_real = [copy.deepcopy(dict(base.trainer.train_dataset[i])) for i in range(len(expected))]
    base_real[0]['labels'][-1] = 7 if base_real[0]['labels'][-1] != 7 else 8
    negatives['baseline_label'] = base_real
    negatives['missing'] = real[:-1]
    negatives['extra'] = [*real, real[0]]
    result = {name: failure(SimpleNamespace(
        train_dataset=rows, data_collator=base.trainer.data_collator if name == 'baseline_label'
        else collator), expected) for name, rows in negatives.items()}

    def float_labels(items):
        batch = collator(items)
        batch['labels'] = batch['labels'].to(torch.float32)
        return batch
    result['dtype'] = failure(SimpleNamespace(train_dataset=real, data_collator=float_labels),
                              expected)
    shuffled = make_wrapper(False, 'shuffled', shuffle=True)
    result['shuffle'] = failure(shuffled.trainer, expected)
    return {
        'environment': environment(), 'control': controller_state(),
        'columns': {'baseline': base.trainer.train_dataset.column_names,
                    'ce_only': trainer.train_dataset.column_names},
        'use_liger_kernel': {'baseline': base.trainer.args.use_liger_kernel,
                             'ce_only': args.use_liger_kernel},
        'args': {k: getattr(args, k) for k in (
            'shuffle_dataset', 'padding_free', 'pad_to_multiple_of', 'assistant_only_loss',
            'loss_type', 'completion_only_loss')},
        'resolved_completion_only_loss': trainer.completion_only_loss,
        'collator': {'class': type(collator).__module__ + '.' + type(collator).__qualname__,
                     'pad_token_id': collator.pad_token_id,
                     'padding_free': collator.padding_free,
                     'completion_only_loss': collator.completion_only_loss},
        'audit': {'baseline': validate_prepared_batches(base.trainer, expected),
                  'ce_only': validate_prepared_batches(trainer, expected)},
        'prefix_expression': prefix, 'rows': len(expected), 'negatives': result,
        'shuffled_order_changed': [dict(r)['input_ids'] for r in shuffled.trainer.train_dataset]
        != [dict(r)['input_ids'] for r in trainer.train_dataset],
    }


def train():
    holder = {}
    observed = {'top': [], 'decoder': [], 'gradient_checkpointing': []}

    trl.SFTTrainer = recording_trainer(trl.SFTTrainer, holder)
    wrapper = make_wrapper(arm == 'ce_only', arm)
    trainer = wrapper.trainer
    inner = unwrap_peft_model(wrapper.model)
    after_setup = dict(control.counts)

    def watch(key):
        def hook(module, args, kwargs):
            if module.training and torch.is_grad_enabled():
                observed[key].append({name: kwargs[name] for name in (
                    'skip_logits', 'return_token_accuracy', 'use_token_scaling') if name in kwargs})
                if key == 'top':
                    observed['gradient_checkpointing'].append(bool(inner.is_gradient_checkpointing))
        return hook
    # PEFT calls the wrapped model's .forward directly, so the decoder (called) is hooked too.
    inner.register_forward_pre_hook(watch('top'), with_kwargs=True)
    inner.model.register_forward_pre_hook(watch('decoder'), with_kwargs=True)
    # Real collator on the real (possibly pruned) dataset, as audited before capture.
    rows = []
    for index in range(len(trainer.train_dataset)):
        batch = trainer.data_collator([trainer.train_dataset[index]])
        rows.append({k: batch[k][0].tolist() for k in KEYS})
    work = [{k: [v] for k, v in row.items()} for row in rows]
    # Mirrors h2_ce_confirmation.run_child training setup (fixed single-row microbatches).
    trainer.train_dataset = Dataset.from_list(rows[:15] if variant == 'drop_batch' else rows)

    def collator(items):
        if len(items) != 1:
            raise ValueError('Fixed single-row microbatches required')
        return {k: torch.tensor([items[0][k]], dtype=torch.long) for k in items[0]}
    trainer.data_collator = collator
    names = [n for n, p in wrapper.model.named_parameters() if p.requires_grad]
    holder['recorder'] = Recorder(root / 'capture', wrapper.model, names, work,
                                  accumulation=4, compute_policy='fp32-cpu-test')
    receipt = {'environment': environment(), 'observed': observed, 'after_setup': after_setup,
               'work': work, 'use_liger_kernel': trainer.args.use_liger_kernel,
               'error': None}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            wrapper.train()
            holder['recorder'].finish(holder['optimizer'], global_step=trainer.state.global_step)
        except Exception as error:
            receipt['error'] = type(error).__name__ + ': ' + str(error)
    receipt['token_accuracy_warnings'] = sum('token_accuracy' in str(w.message) for w in caught)
    receipt['control'] = controller_state()
    return receipt


receipt = prepare() if mode == 'prepare' else train()
(root / 'receipt.json').write_text(json.dumps(receipt, indent=2, default=str))
"""


def run_child(root, mode, arm="none", variant="normal"):
    root.mkdir(parents=True)
    env = dict(
        os.environ,
        PYTHONPATH="src:benchmarks/harness",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        ACCELERATE_USE_CPU="true",
        ACCELERATE_TORCH_DEVICE="cpu",
        TOKENIZERS_PARALLELISM="false",
    )
    result = subprocess.run(
        [sys.executable, "-c", CHILD, mode, arm, variant, str(root)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    receipt = json.loads((root / "receipt.json").read_text())
    assert receipt["environment"]["soup_cli"].startswith(str(ROOT / "src")), receipt
    return receipt


def shifted_counts(work):
    return [sum(v != -100 for v in batch["labels"][0][1:]) for batch in work]


def require_arm_receipt(receipt, arm):
    """Reject a run whose patch points or substituted CE were not reached as on the GPU arm."""
    if receipt["error"] is not None:
        raise AssertionError("Training failed: " + receipt["error"])
    control, observed = receipt["control"], receipt["observed"]
    gradient = [call for call in control["ce"] if call["grad"]]
    diagnostics = [call for call in control["ce"] if not call["grad"]]
    stock = dict.fromkeys(control["bindings"], "stock")
    if arm == "ce_only":
        flags = {key: key == "cross_entropy" for key in ("rope", "rms_norm", "swiglu")}
        flags = dict(flags, fused_linear_cross_entropy=False, cross_entropy=True)
        patches = [
            {"flags": flags, "model": None},
            {"flags": flags, "model": "LlamaForCausalLM"},
        ]
        if (
            control["counts"] != {"class": 1, "instance": 1}
            or not control["counts_ok"]
            or receipt["after_setup"] != {"class": 1, "instance": 0}
            or control["patch"] != patches
            or control["bindings"] != dict(stock, cross_entropy="liger")
            or not control["cross_entropy_is_spy"]
            or receipt["use_liger_kernel"] is not True
            or len(gradient) != 16
            or any(call["reduction"] != "sum" or call["ignore_index"] != -100 for call in gradient)
            or [call["targets"] for call in gradient] != shifted_counts(receipt["work"])
            or any(call["reduction"] != "none" for call in diagnostics)
        ):
            raise AssertionError(
                "CE arm did not reach the rebound cross entropy 16/16: " + str(control)
            )
        injected = {"skip_logits": True, "return_token_accuracy": True, "use_token_scaling": False}
        if observed["top"] != [injected] * 16 or observed["decoder"] != [injected] * 16:
            raise AssertionError("TRL Liger kwargs not injected: " + str(observed))
    elif (
        control["counts"] != {"class": 0, "instance": 0}
        or not control["counts_ok"]
        or control["patch"] != []
        or control["ce"] != []
        or control["bindings"] != stock
        or control["cross_entropy_is_spy"]
        or receipt["use_liger_kernel"] is not False
        or observed["top"] != [{}] * 16
        or observed["decoder"] != [{}] * 16
    ):
        raise AssertionError("Baseline arm touched the Liger path: " + str(control))
    if observed["gradient_checkpointing"] != [True] * 16:
        raise AssertionError("Gradient checkpointing inactive during training")


def denominators(work):
    """Shifted useful targets of each 4-row accumulation window, repeated per microbatch."""
    counts = shifted_counts(work)
    return [sum(counts[index // 4 * 4 : index // 4 * 4 + 4]) for index in range(16)]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    return run_child(tmp_path_factory.mktemp("prepare") / "run", "prepare")


@pytest.fixture(scope="module")
def arms(tmp_path_factory):
    base = tmp_path_factory.mktemp("arms")
    return {
        arm: (base / arm, run_child(base / arm, "train", arm)) for arm in ("baseline", "ce_only")
    }


def test_real_liger_dataset_is_pruned_and_collated_audit_passes(prepared):
    assert set(prepared["columns"]["ce_only"]) == {"input_ids", "assistant_masks"}
    assert {"input_ids", "attention_mask", "labels"} <= set(prepared["columns"]["baseline"])
    assert prepared["use_liger_kernel"] == {"baseline": False, "ce_only": True}
    # One Liger-arm wrapper; setup only, so the class stage ran and the instance stage did not.
    assert prepared["control"]["counts"] == {"class": 1, "instance": 0}
    assert len(prepared["control"]["patch"]) == 1 and prepared["control"]["ce"] == []
    assert prepared["args"] == {
        "shuffle_dataset": False,
        "padding_free": False,
        "pad_to_multiple_of": None,
        "assistant_only_loss": False,
        "loss_type": "nll",
        "completion_only_loss": None,
    }
    assert prepared["resolved_completion_only_loss"] is False
    assert prepared["collator"]["padding_free"] is False
    assert prepared["rows"] == 16
    for arm in ("baseline", "ce_only"):
        audit = prepared["audit"][arm]
        assert audit["status"] == "PASS_EXACT_COLLATED_PREPARED_ROWS"
        assert audit["rows"] == 16 and audit["fields"] == sorted(KEYS)
        # R3b: the audit receipt records the real dataset columns and collator config.
        assert audit["dataset_columns"] == sorted(prepared["columns"][arm])
        assert audit["collator"] == {
            "class": prepared["collator"]["class"],
            "pad_token_id": prepared["collator"]["pad_token_id"],
            "padding_free": False,
            "completion_only_loss": False,
            "pad_to_multiple_of": None,
        }
    # The attempt014 failure, reproduced on the real pruned dataset.
    assert prepared["prefix_expression"] == "KeyError: 'attention_mask'"


@pytest.mark.parametrize(
    "mutation",
    ["assistant_mask", "baseline_label", "token", "order", "missing", "extra", "dtype", "shuffle"],
)
def test_real_prepared_rows_mutations_rejected(prepared, mutation):
    message = prepared["negatives"][mutation]
    assert message is not None and message.startswith("ValueError: "), message
    assert "prepared" in message
    if mutation == "shuffle":
        assert prepared["shuffled_order_changed"] is True


def test_ce_arm_training_is_bit_identical_to_baseline(arms):
    import torch
    from safetensors.torch import load_file

    sys.path.insert(0, str(ROOT / "benchmarks/harness"))
    from h2_ce_training import review_training

    (base_root, base), (ce_root, ce) = arms["baseline"], arms["ce_only"]
    require_arm_receipt(base, "baseline")
    require_arm_receipt(ce, "ce_only")
    assert ce["token_accuracy_warnings"] >= 1 and base["token_accuracy_warnings"] == 0
    assert base["work"] == ce["work"]
    first = json.loads((base_root / "capture/training.json").read_text())
    second = json.loads((ce_root / "capture/training.json").read_text())
    expected = denominators(base["work"])
    assert len(set(expected)) > 1  # Non-uniform windows distinguish normalization bugs.
    for record in (first, second):
        observed = [row["num_items_in_batch"] for row in record["batches"]]
        assert observed == expected and all(type(v) is int for v in observed)
    assert [row["loss"] for row in first["batches"]] == [row["loss"] for row in second["batches"]]
    names = ["initial", "optimizer_final"] + [
        f"{kind}_{step:02d}" for step in range(1, 5) for kind in ("gradient", "adapter")
    ]
    for name in names:
        left = load_file(str(base_root / "capture" / (name + ".safetensors")))
        right = load_file(str(ce_root / "capture" / (name + ".safetensors")))
        assert set(left) == set(right) and left
        assert all(torch.equal(left[key], right[key]) for key in left), name
    policy = json.loads((ROOT / "benchmarks/h2/numerical_policy.json").read_text())
    result = review_training(base_root / "capture", ce_root / "capture", policy, base["work"])
    assert result["status"] == "PASS_CE_TRAINING_TENSORS"


def test_ce_without_sum_reduction_is_rejected_by_pair_review(arms, tmp_path):
    sys.path.insert(0, str(ROOT / "benchmarks/harness"))
    from h2_ce_training import review_training

    base_root, base = arms["baseline"]
    receipt = run_child(tmp_path / "mean", "train", "ce_only", "mean")
    assert receipt["error"] is None
    loss_calls = [call for call in receipt["control"]["ce"] if call["grad"]]
    assert len(loss_calls) == 16
    assert all(call["reduction"] == "sum" for call in loss_calls)  # Spy observed 'sum' pre-forcing.
    policy = json.loads((ROOT / "benchmarks/h2/numerical_policy.json").read_text())
    with pytest.raises(ValueError, match="tolerance"):
        review_training(base_root / "capture", tmp_path / "mean/capture", policy, base["work"])


def test_unapplied_instance_patch_is_detected(tmp_path):
    receipt = run_child(tmp_path / "no_rebind", "train", "ce_only", "no_rebind")
    control = receipt["control"]
    assert control["counts"] == {"class": 1, "instance": 1} and control["ce"] == []
    assert control["bindings"]["cross_entropy"] == "stock"
    with pytest.raises(AssertionError, match="16/16"):
        require_arm_receipt(receipt, "ce_only")


@pytest.mark.parametrize(
    "variant,message",
    [("noop_step", "No adapter changed"), ("drop_batch", "normalization denominator")],
)
def test_real_trainer_faults_rejected_by_recorder(tmp_path, variant, message):
    receipt = run_child(tmp_path / variant, "train", "ce_only", variant)
    assert receipt["error"] is not None and receipt["error"].startswith("ValueError: ")
    assert message in receipt["error"]
