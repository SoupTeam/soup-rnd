"""Kimi K2 tokenizer without Hub code (ticket 16).

``tests/fixtures/kimi_k2_tokenizer_reference.json`` holds token ids that Moonshot's own
``tokenization_kimi.py`` produced once (see its ``provenance`` field). The parity tests need
the real ``tiktoken.model`` (2.8 MB, not in the repo). Point ``SOUP_K2_TOKENIZER_DIR`` at a
folder holding ``tiktoken.model``, ``tokenizer_config.json`` and ``chat_template.jinja`` from
``moonshotai/Kimi-K2-Instruct`` at the fixture's revision. Without it those tests skip.
The synthetic tests always run.
"""

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from soup_cli.utils.kimi_k2 import load_kimi_k2_tokenizer  # noqa: E402

_FIXTURE = Path(__file__).parent / "fixtures" / "kimi_k2_tokenizer_reference.json"
_REFERENCE: Dict[str, Any] = json.loads(_FIXTURE.read_text(encoding="utf-8"))
_REAL_DIR = os.environ.get("SOUP_K2_TOKENIZER_DIR")


def _special(content: str, special: bool = True) -> Dict[str, Any]:
    return {
        "content": content, "lstrip": False, "normalized": False, "rstrip": False,
        "single_word": False, "special": special,
    }


def _tiny_folder(root: Path, base_tokens: int = 300) -> Path:
    """A K2-shaped folder: 256 byte tokens, a few merges, specials at K2's relative ids."""
    root.mkdir(parents=True, exist_ok=True)
    lines = [f"{base64.b64encode(bytes([i])).decode()} {i}" for i in range(256)]
    merges = [b"he", b"ll", b"hell", b"hello", b" w", b"or", b" wor", b"ld", b" world"]
    merges += [bytes([0xE4, 0xBD]), "你".encode()]
    merges += [f"x{i}".encode() for i in range(base_tokens - 256 - len(merges))]
    for offset, token in enumerate(merges):
        lines.append(f"{base64.b64encode(token).decode()} {256 + offset}")
    (root / "tiktoken.model").write_text("\n".join(lines) + "\n")
    first = base_tokens
    decoder = {
        str(first): _special("[BOS]"),
        str(first + 1): _special("[EOS]"),
        str(first + 2): _special("<|im_end|>"),
        str(first + 3): _special("<|im_user|>"),
        str(first + 5): _special("<|tool_call_begin|>", special=False),
        str(first + 254): _special("[UNK]"),
        str(first + 255): _special("[PAD]"),
    }
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": decoder,
                "additional_special_tokens": ["<|im_end|>", "<|im_user|>"],
                "bos_token": "[BOS]", "eos_token": "[EOS]", "unk_token": "[UNK]",
                "pad_token": "[PAD]", "tokenizer_class": "TikTokenTokenizer",
                "auto_map": {"AutoTokenizer": ["tokenization_kimi.TikTokenTokenizer", None]},
            }
        )
    )
    (root / "chat_template.jinja").write_text(
        "{% for m in messages %}<|im_user|>{{ m['content'] }}<|im_end|>{% endfor %}"
    )
    return root


# --- synthetic folder: structure, no Hub code ---


def test_loads_without_hub_code_or_tiktoken_package(tmp_path: Path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "tiktoken", None)  # import tiktoken raises ImportError
    tok = load_kimi_k2_tokenizer(str(_tiny_folder(tmp_path / "k2")))
    assert type(tok).__module__.startswith("transformers")


def test_special_tokens_sit_at_their_configured_ids(tmp_path: Path):
    tok = load_kimi_k2_tokenizer(str(_tiny_folder(tmp_path / "k2")))
    assert tok.convert_tokens_to_ids("[BOS]") == 300
    assert tok.convert_tokens_to_ids("<|im_end|>") == 302
    assert tok.convert_tokens_to_ids("<|tool_call_begin|>") == 305
    assert tok.convert_tokens_to_ids("[UNK]") == 554
    assert tok.convert_tokens_to_ids("[PAD]") == 555
    assert tok.bos_token == "[BOS]" and tok.eos_token == "[EOS]"
    assert tok.pad_token == "[PAD]" and tok.unk_token == "[UNK]"
    assert len(tok) == 556


def test_unlisted_ids_are_reserved_tokens(tmp_path: Path):
    tok = load_kimi_k2_tokenizer(str(_tiny_folder(tmp_path / "k2")))
    assert tok.convert_ids_to_tokens(304) == "<|reserved_token_304|>"
    assert tok.encode("<|reserved_token_304|>") == [304]


def test_special_text_is_one_token_and_round_trips(tmp_path: Path):
    tok = load_kimi_k2_tokenizer(str(_tiny_folder(tmp_path / "k2")))
    ids = tok.encode("hello world<|im_end|>你")
    assert 302 in ids
    assert tok.decode(ids) == "hello world<|im_end|>你"


def test_no_bos_is_added(tmp_path: Path):
    tok = load_kimi_k2_tokenizer(str(_tiny_folder(tmp_path / "k2")))
    assert 300 not in tok.encode("hello")


def test_chat_template_comes_from_the_jinja_file(tmp_path: Path):
    tok = load_kimi_k2_tokenizer(str(_tiny_folder(tmp_path / "k2")))
    text = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False)
    assert text == "<|im_user|>hi<|im_end|>"


def test_refuses_trust_remote_code_and_missing_files(tmp_path: Path):
    folder = _tiny_folder(tmp_path / "k2")
    with pytest.raises(ValueError, match="trust-remote-code"):
        load_kimi_k2_tokenizer(str(folder), trust_remote_code=True)
    (folder / "tiktoken.model").unlink()
    with pytest.raises(FileNotFoundError, match="tiktoken.model"):
        load_kimi_k2_tokenizer(str(folder))


def test_dispatch_sends_only_kimi_k2_to_the_native_loader(tmp_path: Path, monkeypatch):
    from soup_cli.utils import kimi_k2

    folder = _tiny_folder(tmp_path / "k2")
    (folder / "config.json").write_text(json.dumps({"model_type": "kimi_k2"}))
    other = tmp_path / "v3"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"model_type": "deepseek_v3"}))
    from transformers import AutoTokenizer

    seen = []

    def fake_from_pretrained(path, **_kwargs):
        seen.append(path)
        return "plain"

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fake_from_pretrained)
    assert kimi_k2.load_stream_tokenizer(str(other), False) == "plain"
    assert seen == [str(other)]
    tok = kimi_k2.load_stream_tokenizer(str(folder), False)
    assert tok.convert_tokens_to_ids("[BOS]") == 300
    assert seen == [str(other)]


def test_split_pattern_is_the_one_the_reference_was_made_with():
    """Runs in CI, where the real-file parity tests below skip."""
    from soup_cli.utils.kimi_k2 import _K2_SPLIT_PATTERN

    digest = hashlib.sha256(_K2_SPLIT_PATTERN.encode()).hexdigest()
    assert digest == _REFERENCE["provenance"]["split_pattern_sha256"]


# --- real tiktoken.model against Moonshot's own ids ---

_needs_real = pytest.mark.skipif(
    not _REAL_DIR or not (Path(_REAL_DIR or "") / "tiktoken.model").is_file(),
    reason="SOUP_K2_TOKENIZER_DIR does not hold the Kimi-K2-Instruct tokenizer files",
)


@pytest.fixture(scope="module")
def real_tokenizer():
    return load_kimi_k2_tokenizer(str(_REAL_DIR))


@_needs_real
def test_real_files_are_the_ones_the_reference_came_from():
    pinned = _REFERENCE["provenance"]["files_sha256"]
    for name in ("tiktoken.model", "chat_template.jinja", "tokenizer_config.json"):
        digest = hashlib.sha256((Path(_REAL_DIR) / name).read_bytes()).hexdigest()
        assert digest == pinned[name], name


@_needs_real
def test_vocab_size_and_special_ids_match(real_tokenizer):
    assert len(real_tokenizer) == _REFERENCE["vocab_size"]
    for token, index in _REFERENCE["special_ids"].items():
        assert real_tokenizer.convert_tokens_to_ids(token) == index, token


@_needs_real
@pytest.mark.parametrize("index", range(len(_REFERENCE["texts"])))
def test_text_ids_match_moonshots_tokenizer(real_tokenizer, index):
    case = _REFERENCE["texts"][index]
    assert real_tokenizer.encode(case["text"]) == case["ids"], repr(case["text"])


@_needs_real
@pytest.mark.parametrize("index", range(len(_REFERENCE["chats"])))
def test_chat_template_matches_moonshots_rendering(real_tokenizer, index):
    case = _REFERENCE["chats"][index]
    rendered = real_tokenizer.apply_chat_template(
        case["messages"], tokenize=False, add_generation_prompt=case["add_generation_prompt"]
    )
    assert rendered == case["text"]
    assert real_tokenizer.encode(rendered) == case["ids"]
