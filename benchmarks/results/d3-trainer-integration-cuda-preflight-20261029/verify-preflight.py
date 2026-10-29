"""Independent read-only replay of the synthetic development records."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path


def analyze(path):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert set(archive.namelist()) == set(manifest) | {"MANIFEST.json"}
        assert len(archive.namelist()) == len(set(archive.namelist()))
        for name, expected in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
        metadata = json.loads(archive.read("evidence/source-r2/DEVELOPMENT-SOURCE.json"))
        with zipfile.ZipFile(
            io.BytesIO(archive.read("tools/d3-trainer-dev-58145733-01.zip"))
        ) as original:
            assert original.testzip() is None
            originals = json.loads(original.read("DEVELOPMENT-SOURCE.json"))
            for name, expected in originals["files"].items():
                assert hashlib.sha256(original.read(name)).hexdigest() == expected, name
            with zipfile.ZipFile(
                io.BytesIO(archive.read("tools/d3-trainer-revision2.zip"))
            ) as overlay:
                assert overlay.testzip() is None
                for name, expected in metadata["files"].items():
                    payload = (
                        overlay.read(name) if name in overlay.namelist() else original.read(name)
                    )
                    assert hashlib.sha256(payload).hexdigest() == expected, name

        def record(codec, l2l, backend):
            version = "matrix-r3" if codec == "none" else "matrix-r4"
            base = f"evidence/{version}/{codec}-{int(l2l)}-{backend}/"
            data = json.loads(archive.read(base + "status.json"))
            assert data["status"] == "passed"
            assert data["source_files_verified"] == len(metadata["files"])
            assert data["formal_session_started"] is False
            assert data["result"]["total_steps"] == 2
            assert data["runtime"]["expert_fetches"] > 0
            assert data["runtime"]["expert_source"] == "disk-buffered"
            reload = data["reload_comparison"]
            assert reload["actual_dtype"] == reload["wanted_dtype"] == "torch.float32"
            assert reload["actual_sha256"] == reload["wanted_sha256"]
            if backend != "sequential":
                assert data["runtime"]["expert_gemm"]["grouped_matrices"] > 0
            return base, data

        cells = []
        cross_mode = []
        for codec in ["none", "nf4", "nf4-dq"]:
            for l2l in [False, True]:
                ref_base, ref = record(codec, l2l, "sequential")
                for backend in ["sequential", "grouped", "triton"]:
                    base, data = record(codec, l2l, backend)
                    assert data["final_parameters"] == ref["final_parameters"]
                    for name in [
                        "gradients-0.safetensors",
                        "gradients-1.safetensors",
                        "trained-logits.safetensors",
                    ]:
                        assert archive.read(base + name) == archive.read(ref_base + name), (
                            base + name
                        )
                    cells.append(dict(codec=codec, l2l=l2l, backend=backend, status="passed"))
            for backend in ["sequential", "grouped", "triton"]:
                _a, ordinary = record(codec, False, backend)
                _b, layerwise = record(codec, True, backend)
                assert ordinary["final_parameters"] == layerwise["final_parameters"]
                cross_mode.append(dict(codec=codec, backend=backend, exact_final_adapters=True))
        controls = json.loads(archive.read("evidence/instruments-01/controls/report.json"))
        assert controls["status"] == "controls_passed"
        assert [(c["case"], c["expected"], c["observed"]) for c in controls["controls"]] == [
            ("baseline", "ok", "ok"),
            ("own", "ok", "ok"),
            ("foreign", "void", "void"),
        ]
        foreign = json.loads(archive.read("evidence/instruments-01/controls/foreign-final.json"))
        assert foreign["pid"] != controls["pid"] and foreign["ppid"] != controls["pid"]
        fixture = json.loads(archive.read("evidence/instruments-01/fixture.json"))
        assert (
            hashlib.sha256(
                archive.read("evidence/instruments-01/fixture-first-block.bin")
            ).hexdigest()
            == fixture["block_sha256"]
        )
        failures = []
        for prefix in [
            "evidence/smoke-01/",
            "evidence/matrix-r2/none-0-sequential/",
            "evidence/matrix-r3/nf4-0-sequential/",
        ]:
            data = json.loads(archive.read(prefix + "status.json"))
            assert data["status"] == "failed"
            failures.append(dict(path=prefix, traceback=data["traceback"]))
        return dict(
            kind="synthetic CUDA public trainer development preflight",
            formal_pretrained_qualification=False,
            archive_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            manifest_entries_verified=len(manifest),
            development_source_files_verified=len(metadata["files"]),
            passed_cells=len(cells),
            exact_grouped_triton_comparisons=12,
            exact_adapter_reload_checks=18,
            cross_mode=cross_mode,
            cells=cells,
            instrument_controls=[
                dict(case=c["case"], outcome=c["observed"]) for c in controls["controls"]
            ],
            retained_failed_preflight_attempts=failures,
            counter_scope="guest-visible virtual disk; no physical host isolation certification",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze(args.archive)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8", newline="\n")
    else:
        from rich.console import Console

        Console().print(text, end="", markup=False, highlight=False)
