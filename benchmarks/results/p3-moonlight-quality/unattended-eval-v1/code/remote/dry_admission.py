#!/usr/bin/env python
"""Dry admission: assert fresh-host source/cache/tokenizer/adapter identity == frozen protocol.
No scoring, no data. Exit 0 only if every check passes."""
import argparse, os, sys, traceback
from pathlib import Path

PROFILE = "moonlight-16b-a3b"
PROTOCOL_SHA256 = "d22871c398d29e25c614129a1a084903fbc0dce09a5bb5d2659c08d9102b2f82"

failures: list[str] = []


def diff(expected, observed, path="$"):
    if isinstance(expected, dict) and isinstance(observed, dict):
        for key in sorted(expected.keys() | observed.keys()):
            if key not in observed:
                yield f"{path}.{key}: MISSING live"
            elif key not in expected:
                yield f"{path}.{key}: UNEXPECTED live={observed[key]!r:.160}"
            else:
                yield from diff(expected[key], observed[key], f"{path}.{key}")
    elif expected != observed:
        yield f"{path}: protocol={expected!r:.160} live={observed!r:.160}"


def check(name, expected, observed):
    lines = list(diff(expected, observed, name))
    failures.extend(lines)
    print(f"[{'FAIL' if lines else ' ok '}] {name}")
    for line in lines:
        print("       " + line)


def stage(name, fn):
    try:
        return fn()
    except Exception as error:  # report and continue with later independent stages
        failures.append(f"{name}: {type(error).__name__}: {error}")
        print(f"[FAIL] {name}: {type(error).__name__}: {error}")
        traceback.print_exc(limit=3)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--tokenizer-dir", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--protocol-sha256", default=PROTOCOL_SHA256)
    ap.add_argument("--scoring-protocol", type=Path, required=True)
    ap.add_argument("--adapter", action="append", default=[], metavar="DIR:MANIFEST_SHA256",
                    help="repeat for each of the four final adapters")
    ap.add_argument("--no-gpu", action="store_true", help="skip stage 5 (open_session etc.)")
    ap.add_argument("--resident", action="store_true", help="stage 5 also runs prepare_resident")
    args = ap.parse_args()
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "export CUBLAS_WORKSPACE_CONFIG=:4096:8"

    from benchmarks.harness import p3_deepseek_quality_eval as ev
    from benchmarks.harness import p3_deepseek_quality_model as model_api
    from benchmarks.harness import p3_deepseek_quality_prepare as prep
    from benchmarks.harness import p3_deepseek_quality_residency as resident_storage
    from soup_cli import __version__ as soup_version
    from soup_cli.utils import layer_shard

    print("soup_cli", soup_version, "(index.json embeds this; expected 0.75.0)")

    # 0. protocol (same call as eval.py:635-636)
    reference, _ = ev._scoring_protocol(args.scoring_protocol)
    study = ev.admit_protocol(args.protocol, args.protocol_sha256, reference, profile_id=PROFILE)

    # 1. source identity, CPU only: what open_session puts in metadata["source"]
    source = Path(os.path.realpath(args.source))                       # model.py:546
    def source_stage():
        identity = prep.admit_source(source, profile_id=PROFILE)        # prepare.py:687-704
        config = __import__("json").loads((source / "config.json").read_text(encoding="utf-8"))
        _, directory = model_api._normalized_runtime_source(            # adds runtime_config_sha256
            source, config, identity, profile_id=PROFILE)
        directory.cleanup()
        check("source", study["source"], identity)
        return identity
    identity = stage("source", source_stage)

    # 2. cache identity, CPU only: what open_session puts in metadata["cache"] (model.py:644-652)
    cache = Path(os.path.realpath(args.cache))
    def cache_stage():
        index = layer_shard.read_shard_index(str(cache))
        observed = {
            "profile_id": PROFILE, "path": str(cache),
            "index_sha256": model_api.sha256(cache / "index.json"),
            "source_fingerprint": index.source_fingerprint, "dtype": index.dtype,
            "quant": index.quant, "double_quant": index.double_quant,
            "quant_device": index.quant_device, "expert_layout": index.expert_layout,
            "source_parameters": index.total_params, "n_layers": index.n_layers,
            "tier": "disk", "buffers": 2, "read_ahead": 1,
        }
        check("cache", study["cache"], observed)
        if identity is not None:
            check("cache.fingerprint_vs_live_source",
                  {"fp": identity["cache_source_fingerprint"]}, {"fp": index.source_fingerprint})
        # files inspect_shard_cache checks for existence (layer_shard.py:1208-1221)
        paths = [layer_shard.extras_shard_path(str(cache)),
                 *(layer_shard.large_shard_path(str(cache), key) for key in index.large_keys),
                 *layer_shard.layer_paths(str(cache), index),
                 *layer_shard.expert_paths(str(cache), index).values()]
        missing = [p for p in paths if not os.path.exists(p)]
        check("cache.files_present", {"missing": []}, {"missing": missing})
        print(f"       cache payload files: {len(paths)}, "
              f"bytes={sum(os.path.getsize(p) for p in paths if os.path.exists(p))}")
    stage("cache", cache_stage)

    # 3. tokenizer identity (eval.py:670-678); do not realpath: basename must equal the revision
    tokenizer = None
    def token_stage():
        nonlocal tokenizer
        tokenizer, token_identity = ev._load_tokenizer(args.tokenizer_dir, profile_id=PROFILE)
        check("tokenizer", study["tokenizer"], token_identity)
    stage("tokenizer", token_stage)

    # 4. adapters, static (eval.py:650-660, 677-678)
    adapters = []
    for spec in args.adapter:
        root, digest = spec.rsplit(":", 1)
        adapters.append((Path(root), digest))
        def adapter_stage(root=Path(root), digest=digest):
            manifest = ev.admit_adapter(root, digest, profile_id=PROFILE)
            ident = manifest["identity"]
            name = f"adapter[{root.name}]"
            for key in ("source", "cache", "tokenizer", "initial_adapter_sha256", "pack_sha256",
                        "attention_policy"):
                check(f"{name}.{key}", study[key], ident.get(key))
            check(f"{name}.protocol/steps/length",
                  {"p": args.protocol_sha256, "s": study["training"]["steps"], "in_lengths": True},
                  {"p": ident["protocol_sha256"], "s": ident["completed_steps"],
                   "in_lengths": ident["length"] in study["training"]["lengths"]})
        stage(f"adapter {root}", adapter_stage)

    # 5. GPU: the real admission path up to (not including) scoring
    if not args.no_gpu:
        session = None
        resident = None
        try:
            ev._admit_native_execution("cuda")
            session = model_api.open_session(
                source, cache, device="cuda", dtype="bfloat16", prepare=False, profile_id=PROFILE)
            model_api.require_production_source(session.metadata["source"], profile_id=PROFILE)
            check("session.source (eval.py:674)", study["source"], session.metadata["source"])
            check("session.cache  (eval.py:674)", study["cache"], session.metadata["cache"])
            if tokenizer is not None:
                ev.admit_runtime(session, tokenizer, profile_id=PROFILE)
                print("[ ok ] admit_runtime")
            for root, digest in adapters:
                ev.configure_adapter(session, root, digest, profile_id=PROFILE)
                print(f"[ ok ] configure_adapter {root.name}")
            if args.resident and tokenizer is not None and not failures:
                resident = resident_storage.prepare_resident(session)
                resident.admit_runtime(session, tokenizer, test_fixture=False, profile_id=PROFILE)
                print("[ ok ] prepare_resident + resident.admit_runtime")
        except Exception as error:
            failures.append(f"gpu admission: {type(error).__name__}: {error}")
            print(f"[FAIL] gpu admission: {type(error).__name__}: {error}")
            traceback.print_exc(limit=5)
        finally:
            if resident is not None:
                resident.close()
            if session is not None:
                session.close()

    print("\nDRY ADMISSION:", "PASS" if not failures else f"FAIL ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
