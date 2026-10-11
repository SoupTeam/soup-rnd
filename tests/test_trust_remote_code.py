"""Tests for ``--trust-remote-code`` opt-in (v0.36.0 Part B).

Replaces the previous unconditional ``trust_remote_code=True`` smell across
sft.py / chat.py / serve.py with an explicit, auditable opt-in flag plus a
trusted-org allowlist that suppresses noise on first-party models.
"""

from __future__ import annotations

import re
from io import StringIO

import pytest
from rich.console import Console

# Rich help renderer can split a flag like --trust-remote-code with ANSI
# colour escapes between `-`, `-trust`, `-remote-code` when the terminal
# is narrow (macOS CI runners hit this; Windows local does not). Strip
# ANSI so substring assertions are robust. Mirrors the helper in
# tests/test_log_level.py.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mK]")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)

# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------


class TestKnownSafePrefixes:
    def test_meta_llama_is_safe(self):
        from soup_cli.utils.trust_remote import is_known_safe

        assert is_known_safe("meta-llama/Llama-3.2-1B")

    def test_qwen_is_safe(self):
        from soup_cli.utils.trust_remote import is_known_safe

        assert is_known_safe("Qwen/Qwen2.5-7B")

    def test_mistral_is_safe(self):
        from soup_cli.utils.trust_remote import is_known_safe

        assert is_known_safe("mistralai/Mistral-7B-Instruct-v0.3")

    def test_random_org_not_safe(self):
        from soup_cli.utils.trust_remote import is_known_safe

        assert not is_known_safe("randomuser/SomeModel")

    def test_local_path_not_safe(self):
        from soup_cli.utils.trust_remote import is_known_safe

        assert not is_known_safe("./local-checkpoint")

    def test_partial_prefix_does_not_match(self):
        """`meta-llama-evil/...` must NOT match the `meta-llama/` prefix."""
        from soup_cli.utils.trust_remote import is_known_safe

        assert not is_known_safe("meta-llama-evil/SomeModel")

    def test_empty_string_not_safe(self):
        from soup_cli.utils.trust_remote import is_known_safe

        assert not is_known_safe("")

    def test_non_string_not_safe(self):
        from soup_cli.utils.trust_remote import is_known_safe

        assert not is_known_safe(None)
        assert not is_known_safe(123)


# ---------------------------------------------------------------------------
# resolve_trust_remote_code — main entry
# ---------------------------------------------------------------------------


class TestResolve:
    def test_default_off_for_safe_prefix_passes_silently(self):
        """Trusted org + flag off → returns False, no warning."""
        from soup_cli.utils.trust_remote import resolve_trust_remote_code

        buf = StringIO()
        console = Console(file=buf, force_terminal=False)
        out = resolve_trust_remote_code(
            "meta-llama/Llama-3.2-1B",
            requested=False,
            console=console,
            requires_remote_code=False,
        )
        assert out is False
        assert buf.getvalue() == ""

    def test_flag_enabled_warns_once(self):
        """User opted in → return True + warning panel."""
        from soup_cli.utils.trust_remote import resolve_trust_remote_code

        buf = StringIO()
        console = Console(file=buf, force_terminal=False)
        out = resolve_trust_remote_code(
            "shady-org/SomeModel",
            requested=True,
            console=console,
            requires_remote_code=True,
        )
        assert out is True
        output = buf.getvalue()
        assert "trust_remote_code" in output.lower() or "remote code" in output.lower()
        assert "shady-org/SomeModel" in output

    def test_flag_enabled_safe_prefix_suppresses_warning(self):
        """Trusted org doesn't ship custom code — suppress warning even when flag set."""
        from soup_cli.utils.trust_remote import resolve_trust_remote_code

        buf = StringIO()
        console = Console(file=buf, force_terminal=False)
        out = resolve_trust_remote_code(
            "meta-llama/Llama-3.2-1B",
            requested=True,
            console=console,
            requires_remote_code=False,
        )
        assert out is True
        # No noisy panel when the model is from a trusted prefix.
        assert "WARNING" not in buf.getvalue().upper()

    def test_default_off_for_unknown_with_remote_code_raises(self):
        """Model needs custom code + flag off → fail fast with actionable error."""
        from soup_cli.utils.trust_remote import resolve_trust_remote_code

        buf = StringIO()
        console = Console(file=buf, force_terminal=False)
        with pytest.raises(ValueError) as exc_info:
            resolve_trust_remote_code(
                "shady-org/CustomModel",
                requested=False,
                console=console,
                requires_remote_code=True,
            )
        msg = str(exc_info.value)
        assert "shady-org/CustomModel" in msg
        assert "--trust-remote-code" in msg

    def test_default_off_for_unknown_without_remote_code_passes(self):
        """Standard model + flag off → returns False, no error."""
        from soup_cli.utils.trust_remote import resolve_trust_remote_code

        buf = StringIO()
        console = Console(file=buf, force_terminal=False)
        out = resolve_trust_remote_code(
            "shady-org/StandardLlama",
            requested=False,
            console=console,
            requires_remote_code=False,
        )
        assert out is False

    def test_console_optional(self):
        """resolve_trust_remote_code must work when console is None."""
        from soup_cli.utils.trust_remote import resolve_trust_remote_code

        out = resolve_trust_remote_code(
            "meta-llama/Llama-3.2-1B",
            requested=True,
            console=None,
            requires_remote_code=False,
        )
        assert out is True

    def test_invalid_model_name_rejected(self):
        from soup_cli.utils.trust_remote import resolve_trust_remote_code

        with pytest.raises(ValueError, match="model_name"):
            resolve_trust_remote_code(
                "",
                requested=False,
                console=None,
                requires_remote_code=False,
            )


# ---------------------------------------------------------------------------
# model_requires_trust_remote_code (probe HF config for auto_map)
# ---------------------------------------------------------------------------


class TestRequiresProbe:
    def test_local_path_no_auto_map_returns_false(self, tmp_path, monkeypatch):
        """Local path with config.json lacking auto_map → False."""
        from soup_cli.utils.trust_remote import model_requires_trust_remote_code

        config = tmp_path / "config.json"
        config.write_text('{"model_type": "llama"}', encoding="utf-8")
        assert model_requires_trust_remote_code(str(tmp_path)) is False

    def test_local_path_with_auto_map_returns_true(self, tmp_path):
        from soup_cli.utils.trust_remote import model_requires_trust_remote_code

        config = tmp_path / "config.json"
        config.write_text(
            '{"model_type": "custom", "auto_map": '
            '{"AutoModelForCausalLM": "modeling.Custom"}}',
            encoding="utf-8",
        )
        assert model_requires_trust_remote_code(str(tmp_path)) is True

    def test_builtin_model_type_with_auto_map_returns_false(self, tmp_path):
        """transformers ignores auto_map for a built-in model_type when
        trust_remote_code is off, so the gate must not demand the flag."""
        from soup_cli.utils.trust_remote import model_requires_trust_remote_code

        config = tmp_path / "config.json"
        config.write_text(
            '{"model_type": "deepseek_v3", "auto_map": '
            '{"AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM"}}',
            encoding="utf-8",
        )
        assert model_requires_trust_remote_code(str(tmp_path)) is False

    def test_unknown_model_type_with_auto_map_returns_true(self, tmp_path):
        """kimi_k2 has no built-in class on transformers 5.19."""
        from soup_cli.utils.trust_remote import model_requires_trust_remote_code

        config = tmp_path / "config.json"
        config.write_text(
            '{"model_type": "kimi_k2", "auto_map": '
            '{"AutoModelForCausalLM": "modeling_kimi.KimiK2ForCausalLM"}}',
            encoding="utf-8",
        )
        assert model_requires_trust_remote_code(str(tmp_path)) is True

    @pytest.mark.parametrize("model_type_json", ["null", "7", '["llama"]', '""'])
    def test_auto_map_with_unusable_model_type_returns_true(self, tmp_path, model_type_json):
        from soup_cli.utils.trust_remote import model_requires_trust_remote_code

        config = tmp_path / "config.json"
        config.write_text(
            '{"model_type": ' + model_type_json + ', "auto_map": {"AutoConfig": "c.C"}}',
            encoding="utf-8",
        )
        assert model_requires_trust_remote_code(str(tmp_path)) is True

    def test_auto_map_without_model_type_returns_true(self, tmp_path):
        from soup_cli.utils.trust_remote import model_requires_trust_remote_code

        config = tmp_path / "config.json"
        config.write_text('{"auto_map": {"AutoConfig": "c.C"}}', encoding="utf-8")
        assert model_requires_trust_remote_code(str(tmp_path)) is True

    def test_builtin_probe_failure_stays_conservative(self, tmp_path, monkeypatch):
        """If the built-in lookup raises, the gate keeps demanding the flag."""
        from soup_cli.utils import trust_remote

        def boom(_name):
            raise RuntimeError("boom")

        monkeypatch.setattr(trust_remote, "_is_builtin_model_type", boom)
        config = tmp_path / "config.json"
        config.write_text(
            '{"model_type": "llama", "auto_map": {"AutoConfig": "c.C"}}', encoding="utf-8"
        )
        assert trust_remote.model_requires_trust_remote_code(str(tmp_path)) is True

    def test_module_import_does_not_load_torch_or_transformers(self):
        """AGENTS.md lazy-import rule: the built-in lookup runs inside the probe."""
        import subprocess
        import sys

        code = (
            "import sys; import soup_cli.utils.trust_remote; "
            "bad = [m for m in ('torch', 'transformers') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)"
        )
        assert subprocess.run([sys.executable, "-c", code]).returncode == 0

    def test_missing_config_returns_none(self, tmp_path):
        """Missing config.json → None (unknown — caller decides)."""
        from soup_cli.utils.trust_remote import model_requires_trust_remote_code

        out = model_requires_trust_remote_code(str(tmp_path))
        assert out is None

    def test_malformed_config_returns_none(self, tmp_path):
        from soup_cli.utils.trust_remote import model_requires_trust_remote_code

        config = tmp_path / "config.json"
        config.write_text("{this is not json", encoding="utf-8")
        out = model_requires_trust_remote_code(str(tmp_path))
        assert out is None

    def test_non_dict_root_returns_none(self, tmp_path):
        """Config with non-dict root (e.g. JSON array) → None."""
        from soup_cli.utils.trust_remote import model_requires_trust_remote_code

        config = tmp_path / "config.json"
        config.write_text("[1, 2, 3]", encoding="utf-8")
        out = model_requires_trust_remote_code(str(tmp_path))
        assert out is None

    def test_non_directory_path_returns_none(self):
        """Bare HF repo id (not a local dir) → None (unknown)."""
        from soup_cli.utils.trust_remote import model_requires_trust_remote_code

        out = model_requires_trust_remote_code("meta-llama/Llama-3.2-1B")
        assert out is None


class TestTokenizerProbe:
    def test_tokenizer_auto_map_returns_true(self, tmp_path):
        from soup_cli.utils.trust_remote import tokenizer_requires_trust_remote_code

        (tmp_path / "tokenizer_config.json").write_text(
            '{"auto_map": {"AutoTokenizer": ["tok.Tok", null]}}', encoding="utf-8"
        )
        assert tokenizer_requires_trust_remote_code(str(tmp_path)) is True

    @pytest.mark.parametrize("body", [None, "{}", "{not json", "[1]"])
    def test_no_tokenizer_auto_map_returns_false(self, tmp_path, body):
        from soup_cli.utils.trust_remote import tokenizer_requires_trust_remote_code

        if body is not None:
            (tmp_path / "tokenizer_config.json").write_text(body, encoding="utf-8")
        assert tokenizer_requires_trust_remote_code(str(tmp_path)) is False

    def test_export_callers_check_the_tokenizer_too(self):
        """Both quantize paths in export.py must combine the two probes."""
        from pathlib import Path

        import soup_cli.commands.export as export_mod

        source = Path(export_mod.__file__).read_text(encoding="utf-8")
        assert source.count("tokenizer_requires_trust_remote_code(str(source_path))") == 2


# ---------------------------------------------------------------------------
# CLI flag plumbing
# ---------------------------------------------------------------------------


class TestCLIPlumbing:
    """Smoke check that --trust-remote-code is a registered Typer option."""

    def test_train_help_lists_flag(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "--trust-remote-code" in _strip_ansi(result.output), result.output

    def test_chat_help_lists_flag(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "--trust-remote-code" in _strip_ansi(result.output), result.output

    def test_serve_help_lists_flag(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--trust-remote-code" in _strip_ansi(result.output), result.output
