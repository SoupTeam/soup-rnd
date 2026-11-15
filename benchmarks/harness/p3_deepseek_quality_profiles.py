"""Immutable source, tokenizer and execution pins for the paired quality studies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class TokenizerPin:
    files: Mapping[str, Mapping[str, object]]
    vocabulary_size: int
    bos_id: int
    eos_id: int
    pad_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", MappingProxyType({
            name: MappingProxyType(dict(descriptor)) for name, descriptor in self.files.items()
        }))


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    model_id: str
    revision: str
    weight_files: frozenset[str]
    weight_bytes: int
    parameters: int
    shape: Mapping[str, object]
    lora_targets: tuple[str, ...]
    attention_heads: int
    qk_head_dim: int
    v_head_dim: int
    source_inventory_sha256: str
    attention_protocol_sha256: str
    tokenizer: TokenizerPin
    min_free_bytes: int
    min_ram_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", MappingProxyType(dict(self.shape)))
        object.__setattr__(self, "weight_files", frozenset(self.weight_files))
        object.__setattr__(self, "lora_targets", tuple(self.lora_targets))


_DEEPSEEK = ModelProfile(
    profile_id="deepseek-v3",
    model_id="deepseek-ai/DeepSeek-V3",
    revision="e815299b0bcbac849fa540c768ef21845365c9eb",
    weight_files=frozenset(f"model-{number:05d}-of-000163.safetensors" for number in range(1, 164)),
    weight_bytes=688586727753,
    parameters=671026419200,
    shape={
        "model_type": "deepseek_v3", "hidden_size": 7168, "num_hidden_layers": 61,
        "n_routed_experts": 256, "num_experts_per_tok": 8, "first_k_dense_replace": 3,
        "vocab_size": 129280, "q_lora_rank": 1536, "kv_lora_rank": 512,
    },
    lora_targets=("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"),
    attention_heads=128,
    qk_head_dim=192,
    v_head_dim=128,
    source_inventory_sha256="5677f32a218d96414334a1536d66a93d0bb87ac98786585faef41329f7ae7e1c",
    attention_protocol_sha256="fc887c5a996bd58536ef0325b61405cd73a2a83cf363fcc4156204ab180e1ed6",
    tokenizer=TokenizerPin(
        files={
            "tokenizer.json": {
                "bytes": 7847652,
                "sha256": "621ac2e32d0dba658404412318818aaa8ce8cda492e59830109d8da6b517fb41",
            },
            "tokenizer_config.json": {
                "bytes": 3128,
                "sha256": "637bcd1a08cf7c772ce6a383196b22930921c79c3c73223c5afc0c7f41545546",
            },
        },
        vocabulary_size=128815, bos_id=0, eos_id=1, pad_id=1,
    ),
    min_free_bytes=1_400_000_000_000,
    min_ram_bytes=64 * 1024**3,
)

_MOONLIGHT = ModelProfile(
    profile_id="moonlight-16b-a3b",
    model_id="moonshotai/Moonlight-16B-A3B",
    revision="476b36a473d4467f94469414bef6cee75c9c8172",
    weight_files=frozenset(f"model-{number}-of-27.safetensors" for number in range(1, 28)),
    weight_bytes=31920879456,
    # Model tensor elements, including correction bias, excluding regenerated rotary buffers.
    parameters=15960110208,
    shape={
        "model_type": "deepseek_v3", "hidden_size": 2048, "num_hidden_layers": 27,
        "n_routed_experts": 64, "num_experts_per_tok": 6, "first_k_dense_replace": 1,
        "vocab_size": 163840, "q_lora_rank": None, "kv_lora_rank": 512,
    },
    lora_targets=("q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"),
    attention_heads=16,
    qk_head_dim=192,
    v_head_dim=128,
    source_inventory_sha256="372baec885e52e89d26895f79759758b9600b67c518f80761c437e0f5e08a5ee",
    attention_protocol_sha256="b7ec8f56ca073a1397a7d4579d6b32550b18b148c67a9f1be2fbc50c873d1363",
    tokenizer=TokenizerPin(
        files={
            "tokenizer.json": {
                "bytes": 19592675,
                "sha256": "a5b137b65b40e0283a3c646472444078503dd3cb83c64b69379beca74288773d",
            },
            "tokenizer_config.json": {
                "bytes": 344,
                "sha256": "7e0f5761740d72e66d290afdce2cf9d10c598d018503554066ee55d5bf2a30a2",
            },
            "chat_template.jinja": {
                "bytes": 527,
                "sha256": "0af7f5845651f3ba85a82d70d8af968a1da0a60f4c7bb662a9c305cfc9cc224c",
            },
        },
        vocabulary_size=163840, bos_id=163584, eos_id=163585, pad_id=163838,
    ),
    min_free_bytes=150 * 10**9,
    min_ram_bytes=32 * 1024**3,
)

PROFILES: Mapping[str, ModelProfile] = MappingProxyType({
    profile.profile_id: profile for profile in (_DEEPSEEK, _MOONLIGHT)
})


def get_profile(profile_id: str) -> ModelProfile:
    try:
        return PROFILES[profile_id]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown quality model profile: {profile_id!r}") from error
