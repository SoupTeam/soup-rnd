"""Profile boundaries: immutable pins and refusal before consuming foreign artifacts."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from benchmarks.harness import p3_deepseek_quality_model as model
from benchmarks.harness import p3_deepseek_quality_prepare as prepare
from benchmarks.harness.p3_deepseek_quality_profiles import PROFILES, get_profile


@pytest.mark.parametrize("profile_id", ["deepseek-v3", "moonlight-16b-a3b"])
def test_profile_pins_cannot_be_changed_through_nested_references(profile_id):
    profile = get_profile(profile_id)
    with pytest.raises(FrozenInstanceError):
        profile.revision = "0" * 40
    with pytest.raises(TypeError):
        profile.shape["hidden_size"] = 1
    with pytest.raises(TypeError):
        profile.tokenizer.files["tokenizer.json"]["sha256"] = "0" * 64
    with pytest.raises(TypeError):
        PROFILES[profile_id] = profile


@pytest.mark.parametrize("profile_id", ["", "moonlight", "deepseek-v2"])
def test_unknown_profile_never_falls_back_to_deepseek(profile_id):
    with pytest.raises(ValueError, match="profile"):
        get_profile(profile_id)


@pytest.mark.parametrize("profile_id", ["deepseek-v3", "moonlight-16b-a3b"])
def test_changed_hub_shard_size_is_rejected_before_payload_download(profile_id):
    profile = get_profile(profile_id)
    rows = []
    for name, descriptor in model.source_inventory(profile_id).items():
        lfs = (SimpleNamespace(sha256=descriptor["hub_sha256"], size=descriptor["bytes"])
               if descriptor["hub_sha256"] is not None else None)
        rows.append(SimpleNamespace(rfilename=name, size=descriptor["bytes"], lfs=lfs,
                                    blob_id=descriptor["hub_etag"] if lfs is None else None))
    info = SimpleNamespace(sha=profile.revision, siblings=rows)
    assert prepare.hub_inventory(info, profile_id=profile_id) == model.source_inventory(profile_id)
    changed = next(row for row in rows if row.rfilename in profile.weight_files)
    changed.size -= 1
    changed.lfs.size -= 1
    with pytest.raises(ValueError, match="byte total"):
        prepare.hub_inventory(info, profile_id=profile_id)


@pytest.mark.parametrize("source_profile,selected_profile", [
    ("moonlight-16b-a3b", "deepseek-v3"),
    ("deepseek-v3", "moonlight-16b-a3b"),
])
def test_foreign_source_receipt_is_refused_before_cache_or_tensor_access(
    source_profile, selected_profile,
):
    profile = get_profile(source_profile)
    metadata = {
        "profile_id": source_profile,
        "production_qualified": True,
        "model_id": profile.model_id,
        "revision": profile.revision,
        "shape": dict(profile.shape),
        "weight_files": len(profile.weight_files),
        "weight_bytes": profile.weight_bytes,
        "source_verification": "hub_commit_etag_and_size_plus_cache_source_fingerprint",
        "receipt_sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="profile"):
        model.require_production_source(metadata, profile_id=selected_profile)


def test_moonlight_capacity_does_not_inherit_671b_requirements():
    profile = get_profile("moonlight-16b-a3b")
    memory = {"effective_bytes": profile.min_ram_bytes}
    prepare.require_capacity(profile.min_free_bytes, memory, profile_id=profile.profile_id)
    with pytest.raises(ValueError, match="NVMe"):
        prepare.require_capacity(profile.min_free_bytes - 1, memory, profile_id=profile.profile_id)
    with pytest.raises(ValueError, match="RAM"):
        prepare.require_capacity(profile.min_free_bytes,
                                 {"effective_bytes": profile.min_ram_bytes - 1},
                                 profile_id=profile.profile_id)
    with pytest.raises(ValueError, match="1.4TB"):
        prepare.require_capacity(profile.min_free_bytes, memory)


