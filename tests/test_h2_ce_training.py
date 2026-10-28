"""Full tensor evidence and fault controls for prospective CE-only training."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'benchmarks/harness'))
from h2_ce_training import Recorder, review_training  # noqa: E402

ROOT = Path(__file__).parents[1]
POLICY = json.loads((ROOT / 'benchmarks/h2/numerical_policy.json').read_text())


def fixture_run(root, *, no_update=False):
    torch = pytest.importorskip('torch')
    torch.manual_seed(42)
    model = torch.nn.Linear(2, 2, bias=False)
    model.register_buffer('frozen', torch.ones(1))
    batch = {'input_ids': [[1, 2, 3]], 'attention_mask': [[1, 1, 1]],
             'labels': [[-100, 2, 3]]}
    work = [batch] * 16
    recorder = Recorder(root, model, ['weight'], work, accumulation=4,
                        compute_policy='fp32-cpu-test')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0)
    recorder.optimizer_created(optimizer, max_grad_norm=1.0)
    for index in range(16):
        loss = model(torch.tensor([[2.0, 3.0]])).square().sum() / 4
        inputs = {k: torch.tensor(v) for k, v in batch.items()}
        recorder.batch(inputs, loss, 8)
        loss.backward()
        recorder.backward()
        if (index + 1) % 4 == 0:
            recorder.preclip(scaler=None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not no_update:
                optimizer.step()
            recorder.poststep(optimizer)
            optimizer.zero_grad(set_to_none=True)
    recorder.finish(optimizer, global_step=4)
    return work


def test_pair_and_full_evidence(tmp_path):
    from safetensors.torch import load_file

    work = fixture_run(tmp_path / 'left')
    fixture_run(tmp_path / 'right')
    result = review_training(tmp_path / 'left', tmp_path / 'right', POLICY, work)
    assert result['status'] == 'PASS_CE_TRAINING_TENSORS'
    assert result['performance_verdict'] == 'NO_VERDICT'
    gradient = load_file(str(tmp_path / 'left/gradient_01.safetensors'))['weight']
    assert gradient.norm().item() > 1.0  # Captured before clipping.
    assert len(result['steps']) == 4


def test_no_optimizer_update_rejected(tmp_path):
    with pytest.raises(ValueError, match='No adapter changed'):
        fixture_run(tmp_path / 'bad', no_update=True)


@pytest.mark.parametrize('fault', ['bytes', 'gradient', 'delta', 'loss', 'order',
                                 'denominator', 'frozen', 'step', 'moment', 'missing'])
def test_tampered_or_divergent_evidence_rejected(tmp_path, fault):
    from h2_ce_training import seal
    from safetensors.torch import load_file, save_file

    work = fixture_run(tmp_path / 'left')
    fixture_run(tmp_path / 'right')
    right = tmp_path / 'right'
    if fault == 'bytes':
        with (right / 'gradient_01.safetensors').open('ab') as stream:
            stream.write(b'changed')
    elif fault in {'gradient', 'delta', 'moment'}:
        name = {'gradient': 'gradient_01', 'delta': 'adapter_01',
                'moment': 'optimizer_final'}[fault]
        path = right / (name + '.safetensors')
        tensors = load_file(str(path))
        key = next(iter(tensors))
        tensors[key] = tensors[key] + (10 if fault != 'moment' else float('nan'))
        save_file(tensors, str(path))
        seal(right)  # Rehashing an invalid observation must not make it valid.
    elif fault == 'missing':
        (right / 'gradient_04.safetensors').unlink()
        seal(right)
    else:
        path = right / 'training.json'
        data = json.loads(path.read_text())
        if fault == 'loss':
            data['batches'][0]['loss'] += 10
        elif fault == 'order':
            data['batches'][0]['inputs']['input_ids'][0][0] += 1
        elif fault == 'denominator':
            data['batches'][0]['num_items_in_batch'] += 1
        elif fault == 'frozen':
            data['after']['frozen']['buffer:frozen'] = '0' * 64
        elif fault == 'step':
            data['events'][0]['completed_batches'] = 3
        path.write_text(json.dumps(data))
        seal(right)
    with pytest.raises(ValueError):
        review_training(tmp_path / 'left', right, POLICY, work)


def test_symlink_rejected(tmp_path):
    work = fixture_run(tmp_path / 'left')
    (tmp_path / 'link').symlink_to(tmp_path / 'left', target_is_directory=True)
    with pytest.raises(ValueError):
        review_training(tmp_path / 'left', tmp_path / 'link', POLICY, work)


def test_small_update_divergence_is_not_hidden_by_large_initial_weights():
    torch = pytest.importorskip('torch')
    from h2_ce_training import compare_tensors

    initial = torch.tensor([1000.0], dtype=torch.float64)
    left, right = initial + 0.001, initial + 0.002
    # Comparing final adapters would hide a 2x update difference.
    compare_tensors({'adapter': left}, {'adapter': right}, POLICY)
    with pytest.raises(ValueError, match='tolerance'):
        compare_tensors({'adapter': left - initial}, {'adapter': right - initial}, POLICY)


def test_actual_trainer_accumulation_and_preclip(tmp_path):
    import os
    import subprocess

    for dependency in ['torch', 'transformers', 'trl', 'peft', 'datasets']:
        pytest.importorskip(dependency)
    code = '''
import json,sys,torch,trl
from pathlib import Path
from datasets import Dataset
from h2_trainer_smoke import build_local_model
from h2_ce_training import Recorder,recording_trainer,review_training
from soup_cli.config.schema import SoupConfig
from soup_cli.trainer.sft import SFTTrainerWrapper
root=Path(sys.argv[1]);build_local_model(root/'model')
holder={};trl.SFTTrainer=recording_trainer(trl.SFTTrainer,holder)
class CPUWrapper(SFTTrainerWrapper):
    @staticmethod
    def _as_sft_config(args,max_length,packing=False):
        args.use_cpu=True
        return SFTTrainerWrapper._as_sft_config(args,max_length,packing)
config=SoupConfig.model_validate({
 'base':str(root/'model'),'task':'sft','backend':'transformers',
 'data':{'train':'in-memory','format':'alpaca','max_length':64},
 'output':str(root/'output'),
 'training':{'epochs':1,'batch_size':1,'gradient_accumulation_steps':4,
 'quantization':'none','seed':42,'data_seed':42,'lr':0.0002,'optimizer':'adamw_torch',
 'scheduler':'constant','warmup_ratio':0.0,'weight_decay':0.0,
 'gradient_checkpointing':False,'auto_mixed_precision':False,
 'logging_steps':1,'save_steps':1000,'lora':{'r':2,'alpha':4,'dropout':0.0,
 'target_modules':['q_proj','v_proj']}}})
wrapper=CPUWrapper(config,device='cpu',report_to='none')
wrapper.setup({'train':[{'messages':[{'role':'user','content':'hello world'},
 {'role':'assistant','content':'answer seven'}]} for _ in range(16)]})
rows=[{k:list(row[k]) for k in ('input_ids','attention_mask','labels')}
      for row in wrapper.trainer.train_dataset]
assert len(rows)==16
work=[{k:[v] for k,v in row.items()} for row in rows]
wrapper.trainer.train_dataset=Dataset.from_list(rows)
def collator(items):
    assert len(items)==1
    return {k:torch.tensor([items[0][k]]) for k in items[0]}
wrapper.trainer.data_collator=collator
names=[n for n,p in wrapper.model.named_parameters() if p.requires_grad]
holder['recorder']=Recorder(root/'capture',wrapper.model,names,work,
 accumulation=4,compute_policy='fp32-cpu-test')
wrapper.train()
holder['recorder'].finish(holder['optimizer'],global_step=wrapper.trainer.state.global_step)
policy=json.loads(Path('benchmarks/h2/numerical_policy.json').read_text())
result=review_training(root/'capture',root/'capture',policy,work)
assert result['status']=='PASS_CE_TRAINING_TENSORS'
'''
    env = dict(os.environ, PYTHONPATH='src:benchmarks/harness', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', ACCELERATE_USE_CPU='true',
               ACCELERATE_TORCH_DEVICE='cpu', TOKENIZERS_PARALLELISM='false')
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], cwd=ROOT,
                            env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
