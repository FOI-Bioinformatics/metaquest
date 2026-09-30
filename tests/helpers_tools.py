"""Fake external tools on PATH for the tool-table and doctor tests.

``fake_tool`` writes a small ``#!/bin/sh`` script under ``tmp_path/bin`` that prints fixed
text and exits with a fixed code, whatever arguments it is given. Callers put the returned
folder on ``PATH`` (``monkeypatch.setenv("PATH", str(bin_dir))``), so ``shutil.which`` and
``run_secure`` find the fake and never a real tool.
"""

import shlex
from pathlib import Path


def fake_tool(tmp_path: Path, name: str, stdout: str, stderr: str = "", rc: int = 0) -> Path:
    """Write an executable ``name`` into ``tmp_path/bin`` that prints ``stdout``/``stderr`` and exits ``rc``.

    Returns the ``bin`` folder, so several tools can share one folder on ``PATH``.
    """
    bin_dir = Path(tmp_path) / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / name
    lines = ["#!/bin/sh"]
    if stdout:
        lines.append(f"printf '%s\\n' {shlex.quote(stdout)}")
    if stderr:
        lines.append(f"printf '%s\\n' {shlex.quote(stderr)} >&2")
    lines.append(f"exit {int(rc)}")
    script.write_text("\n".join(lines) + "\n")
    script.chmod(0o755)
    return bin_dir
