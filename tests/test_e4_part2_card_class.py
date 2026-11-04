"""E4 part 2: decide verifies the card named in each F record's box stamp (review of PR #21).

The bounds are the harness constants that define the rule's card names ("A100 40 GB" and
"A100 80 GB", sections 1 and 3); they change no threshold.
"""
# ruff: noqa: I001 (the support import must run before the harness imports)

import json
import os

import pytest

from tests.test_e4_part2_calibration import GOOD_PROV, GOOD_STAMP

pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")
pytest.importorskip("peft")

import e4_part2_calibration as cal  # noqa: E402
import e4_part2_common as common  # noqa: E402

pytestmark = pytest.mark.skipif(
    bool(common.library_versions()["mismatches"]),
    reason="the E4 part 2 harness is pinned to the rule's library versions",
)


@pytest.fixture(autouse=True)
def _own_gpu_context(monkeypatch):
    monkeypatch.setattr(common, "identify_own_gpu_pids_cuda", lambda gpu_index=0, **kw: {
        "own_pids": [os.getpid()], "method": "stub"})


def card(name="NVIDIA A100-SXM4-40GB", mib=40960, cap="8.0"):
    return dict(GOOD_STAMP, gpu_name=name, gpu_memory_total_mib=mib, gpu_compute_cap=cap)


A100_40 = card()
A100_40_PCIE = card("NVIDIA A100-PCIE-40GB", 40960)
A100_80 = card("NVIDIA A100-SXM4-80GB", 81920)


def write_f(out, prefix, stamp, reserved=21e9):
    journal = common.AttemptsJournal(out / "attempts.jsonl")
    for arm in cal.ARMS:
        cal.execute_attempt(
            journal, f"{prefix}_{arm}", "F", out / "runs",
            lambda: {"oom": False, "failed": False, "max_memory_reserved_bytes": reserved},
            stamp=lambda *_a: dict(stamp), prov=lambda: GOOD_PROV,
            identify_own=lambda: {"own_pids": [], "method": "stub"})


def decide(out, capsys):
    code = cal.main(["decide", "--out", str(out)])
    return code, capsys.readouterr()


class TestAcceptedCards:
    def test_a100_40gb_sxm_and_pcie_are_final_under_f(self, tmp_path, capsys):
        for i, stamp in enumerate((A100_40, A100_40_PCIE)):
            out = tmp_path / str(i)
            out.mkdir()
            write_f(out, "F", stamp)
            assert decide(out, capsys)[0] == 0
            assert json.loads((out / "decision.json").read_text())["decision"] == (
                "FINAL_A100_40GB")

    def test_a100_80gb_is_final_under_f80_after_a_40gb_f_over_the_limit(self, tmp_path, capsys):
        write_f(tmp_path, "F", A100_40, reserved=39e9)
        write_f(tmp_path, "F80", A100_80, reserved=39e9)
        assert decide(tmp_path, capsys)[0] == 0
        assert json.loads((tmp_path / "decision.json").read_text())["decision"] == (
            "FINAL_A100_80GB")

    def test_the_bounds_are_inclusive(self, tmp_path, capsys):
        for i, mib in enumerate((40000, 42000)):
            out = tmp_path / str(i)
            out.mkdir()
            write_f(out, "F", card(mib=mib))
            assert decide(out, capsys)[0] == 0


class TestRefusedCards:
    def refused(self, tmp_path, capsys, field_value):
        code, seen = decide(tmp_path, capsys)
        assert code != 0 and not (tmp_path / "decision.json").exists()
        assert field_value in seen.err + seen.out
        return seen

    def test_an_80gb_card_under_f_is_not_final_a100_40gb(self, tmp_path, capsys):
        """nurik's case: three A100-SXM4-80GB records under F, reserved < 36 GB."""
        write_f(tmp_path, "F", A100_80, reserved=21e9)
        self.refused(tmp_path, capsys, "gpu_memory_total_mib=81920")

    def test_a_40gb_card_under_f80_is_refused(self, tmp_path, capsys):
        write_f(tmp_path, "F", A100_40, reserved=39e9)
        write_f(tmp_path, "F80", A100_40, reserved=39e9)
        self.refused(tmp_path, capsys, "gpu_memory_total_mib=40960")

    def test_a_card_that_is_not_an_a100_is_refused(self, tmp_path, capsys):
        write_f(tmp_path, "F", card("NVIDIA H100 80GB HBM3", 81559, "9.0"))
        self.refused(tmp_path, capsys, "gpu_name=NVIDIA H100 80GB HBM3")

    def test_compute_capability_8_6_is_refused(self, tmp_path, capsys):
        write_f(tmp_path, "F", card(cap="8.6"))
        self.refused(tmp_path, capsys, "gpu_compute_cap=8.6")

    def test_a_stamp_without_total_memory_is_refused(self, tmp_path, capsys):
        write_f(tmp_path, "F", card(mib=None))
        self.refused(tmp_path, capsys, "gpu_memory_total_mib=None")

    def test_a_stamp_without_compute_capability_is_refused(self, tmp_path, capsys):
        write_f(tmp_path, "F", card(cap=None))
        self.refused(tmp_path, capsys, "gpu_compute_cap=None")

    def test_the_after_stamp_is_checked_too(self, tmp_path, capsys):
        journal = common.AttemptsJournal(tmp_path / "attempts.jsonl")
        for arm in cal.ARMS:
            stamps = iter([A100_40, A100_80])
            cal.execute_attempt(
                journal, f"F_{arm}", "F", tmp_path / "runs",
                lambda: {"oom": False, "failed": False, "max_memory_reserved_bytes": 1},
                stamp=lambda *_a: dict(next(stamps)), prov=lambda: GOOD_PROV,
                identify_own=lambda: {"own_pids": [], "method": "stub"})
        self.refused(tmp_path, capsys, "gpu_memory_total_mib=81920")


class TestStampCarriesTheCard:
    def test_box_stamp_records_total_memory_and_compute_capability(self, monkeypatch):
        answers = {"name": "NVIDIA A100-SXM4-40GB", "uuid": "GPU-1", "driver_version": "550",
                   "memory.total": "40960", "compute_cap": "8.0"}
        monkeypatch.setattr(common, "workload_gpu_id", lambda: "0")
        monkeypatch.setattr(common, "_nvidia_query", lambda field, gpu_id="0": answers[field])
        stamp = common.box_stamp(0, "42", readers={
            k: (lambda: None) for k in ("ram_available_bytes", "commit_charge", "gpu_processes",
                                        "gpu_clock", "gpu_memory_used_mib")})
        assert stamp["gpu_memory_total_mib"] == "40960" and stamp["gpu_compute_cap"] == "8.0"
