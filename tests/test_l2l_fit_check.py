"""The A2-03 fit check's byte estimate, on SmolLM2-135M's published shape."""

from types import SimpleNamespace

from tests.test_l2l_probe_harness import _load

SMOLLM2_135M = SimpleNamespace(
    hidden_size=576, intermediate_size=1536, num_hidden_layers=30, vocab_size=49152,
    num_attention_heads=9, num_key_value_heads=3, tie_word_embeddings=True,
)


def test_weights_bytes_match_the_135m_parameter_count():
    parts = _load("l2l_fit_check").estimate_bytes(SMOLLM2_135M, seq=128, k=4)
    # 134.5 M parameters at 2 bytes each; the published count is 134,515,008
    assert abs(parts["weights"] - 2 * 134_515_008) < 2 * 1_000_000


def test_logits_grow_with_k():
    check = _load("l2l_fit_check")
    one = check.estimate_bytes(SMOLLM2_135M, seq=128, k=1)["logits"]
    four = check.estimate_bytes(SMOLLM2_135M, seq=128, k=4)["logits"]
    assert four == 4 * one
