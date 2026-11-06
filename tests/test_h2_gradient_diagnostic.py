"""Scaling must not hide directional errors or manufacture a parity verdict."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'benchmarks/harness'))
from h2_gradient_diagnostic import diagnose, scale_geometry  # noqa: E402


@pytest.mark.parametrize('scale', [1.0, 0.25, -2.0, 0.0])
def test_pure_scaling(scale):
    torch = pytest.importorskip('torch')
    a = torch.tensor([1., -3., 2.], dtype=torch.float64)
    before = a.clone()
    result = scale_geometry(a, scale * a)
    assert result['best_fit_candidate_scale'] == pytest.approx(scale)
    assert result['residual_over_reference_norm'] == pytest.approx(0)
    assert a.equal(before)


def test_directional_error_survives_fit():
    torch = pytest.importorskip('torch')
    result = scale_geometry(torch.tensor([1., 0.]), torch.tensor([2., 3.]))
    assert result['best_fit_candidate_scale'] == 2
    assert result['residual_over_reference_norm'] == 3


def test_chunk_boundary_matches_independent_norm():
    torch = pytest.importorskip('torch')
    a = torch.arange(65539, dtype=torch.float64) / 65539
    b = a * 0.75
    b[-1] += 2
    result = scale_geometry(a, b)
    scale = torch.dot(a, b) / torch.dot(a, a)
    expected = torch.linalg.vector_norm(b - scale * a) / torch.linalg.vector_norm(a)
    assert result['best_fit_candidate_scale'] == pytest.approx(scale.item())
    assert result['residual_over_reference_norm'] == pytest.approx(expected.item())


def test_zero_reference_has_no_fitted_scale():
    torch = pytest.importorskip('torch')
    for b in (torch.zeros(2), torch.ones(2)):
        result = scale_geometry(torch.zeros(2), b)
        assert result['best_fit_candidate_scale'] is None
        assert result['residual_over_reference_norm'] is None
        assert result['both_zero'] == bool(b.eq(0).all())


@pytest.mark.parametrize('kind', ['nan', 'inf', 'shape', 'dtype', 'empty', 'integer'])
def test_invalid_gradients_rejected(kind):
    torch = pytest.importorskip('torch')
    a, b = torch.ones(2), torch.ones(2)
    if kind in {'nan', 'inf'}:
        b[0] = float(kind)
    elif kind == 'shape':
        b = b[:1]
    elif kind == 'dtype':
        b = b.double()
    elif kind == 'empty':
        a, b = a[:0], b[:0]
    else:
        a, b = a.long(), b.long()
    with pytest.raises(ValueError):
        scale_geometry(a, b)


@pytest.mark.parametrize('mismatch', ['before', 'work', 'compute_policy', 'input', 'names'])
def test_diagnostic_rejects_incomparable_probes(monkeypatch, tmp_path, mismatch):
    torch = pytest.importorskip('torch')
    import h2_numerics

    metadata = {'schema': 'test', 'compute_policy': 'test', 'before': {'trainable': {'x': {}}},
                'work': {}}
    other = dict(metadata)
    left = {'input.ids': torch.tensor([1]), 'gradient.x': torch.ones(1)}
    right = dict(left)
    if mismatch == 'input':
        right['input.ids'] = torch.tensor([2])
    elif mismatch == 'names':
        right['extra'] = torch.ones(1)
    else:
        other[mismatch] = 'different'
    monkeypatch.setattr(h2_numerics, 'read_probe',
                        lambda path: (metadata, left) if path.name == 'ref' else (other, right))
    with pytest.raises(ValueError, match='mismatch'):
        diagnose(tmp_path / 'ref', tmp_path / 'candidate')
