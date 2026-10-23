"""Fingerprint the explicit bitsandbytes NF4 state outside PyTorch buffers."""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from pathlib import Path
from typing import Any

from h2_update_oracle import tensor_sha

STATE_FIELDS = {'absmax', 'shape', 'code', 'dtype', 'blocksize', 'quant_type',
                'offset', 'state2', 'nested'}


def _state(value: Any, *, depth: int = 0) -> dict:
    import torch
    from bitsandbytes.functional import QuantState

    if type(value) is not QuantState or set(vars(value)) != STATE_FIELDS or depth > 1:
        raise ValueError('Unknown quantizer state schema; freeze/review this runtime first')
    if value.nested is not (value.state2 is not None):
        raise ValueError('Inconsistent nested quantizer state')
    if depth and value.state2 is not None:
        raise ValueError('Unexpected recursively nested quantizer')
    if type(value.blocksize) is not int or value.blocksize <= 0:
        raise ValueError('Invalid quantizer block size')
    if not isinstance(value.dtype, torch.dtype):
        raise ValueError('Missing quantizer source dtype')
    shape = value.shape
    if shape is not None and (not isinstance(shape, (tuple, list, torch.Size))
                              or not all(type(v) is int and v > 0 for v in shape)):
        raise ValueError('Invalid quantizer source shape')
    result = {'shape': list(shape) if shape is not None else None,
              'dtype': str(value.dtype), 'blocksize': value.blocksize,
              'quant_type': value.quant_type, 'nested': value.nested}
    for key in ('absmax', 'code', 'offset'):
        tensor = getattr(value, key)
        if tensor is None and key == 'offset' and not value.nested:
            result[key] = None
        else:
            if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all().item():
                raise ValueError('Missing/nonfinite quantizer tensor: ' + key)
            result[key] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                           'sha256': tensor_sha(tensor)}
    result['state2'] = _state(value.state2, depth=depth + 1) if value.nested else None
    return result


def quantizer_snapshot(model: Any) -> dict:
    """Reject uninitialized/unknown 4-bit weights; hash all known NF4 decode state.

    This covers bitsandbytes Params4bit/QuantState, not arbitrary quantizer plugins.
    The helper records its installed source identity; a candidate GPU runtime still
    needs separately pinned versions and live validation.
    """
    candidates = {name: param for name, param in model.named_parameters()
                  if hasattr(param, 'quant_state')}
    if not candidates:
        return {'kind': 'NO_PARAMS4BIT_OBSERVED', 'parameters': {}}
    import torch
    from bitsandbytes.functional import QuantState
    from bitsandbytes.nn import Params4bit

    states = {}
    for name, param in candidates.items():
        if (type(param) is not Params4bit or param.requires_grad
                or param.bnb_quantized is not True or param.quant_state is None
                or param.quant_type != 'nf4' or param.quant_storage is not torch.uint8
                or param.dtype is not torch.uint8):
            raise ValueError('Expected frozen, initialized uint8 NF4 Params4bit: ' + name)
        if (type(param.compress_statistics) is not bool
                or param.quant_state.nested is not param.compress_statistics
                or param.blocksize != param.quant_state.blocksize
                or param.quant_state.quant_type != param.quant_type):
            raise ValueError('Weight/quantizer metadata disagree: ' + name)
        states[name] = {'packed_weight_sha256': tensor_sha(param),
                        'compress_statistics': param.compress_statistics,
                        'quant_storage': str(param.quant_storage),
                        'state': _state(param.quant_state)}
    return {'kind': 'BITSANDBYTES_NF4_STATE', 'parameters': states,
            'bitsandbytes_version': importlib.metadata.version('bitsandbytes'),
            'implementation_sha256': {
                cls.__name__: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
                for cls in (Params4bit, QuantState)},
            'performance_verdict': 'NO_VERDICT'}
